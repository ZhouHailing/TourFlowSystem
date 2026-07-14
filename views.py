import cv2
import os
import time
import threading
import numpy as np
from datetime import datetime
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import ensure_csrf_cookie
from django.core.files.base import ContentFile
from ultralytics import YOLO
from .models import FaceCapture, FlowCount, FlowWarn
import json
from django.http import JsonResponse
from datetime import datetime, timedelta
import requests


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH =r"c:\Users\zhouhzai\Downloads\ScreenRecording_06-07-2026 10-28-36_1(2).MP4"
MODEL_PERSON = os.path.join(BASE_DIR, "yolov11_model", "yolo11n.pt")
MODEL_FACE = os.path.join(BASE_DIR, "yolov11_model", "yolo11n-face.pt")

WARN_LIMIT = 15
SPLIT_LINE_RATIO = 0.58

# 行人检测过滤：提高置信度 + 宽高比，减少行李/景物误检
PERSON_CONF = 0.4
MIN_BOX_H = 60
MIN_ASPECT = 1.25
MAX_ASPECT = 4.5
MIN_STABLE_FRAMES = 5
DETECT_INTERVAL = 2

global_in = 0
global_out = 0
last_stat_minute = ""
lock = threading.Lock()
last_detected_centroids = []
last_detected_boxes = []
_track_meta = {}
_warn_active = False

debug_info = {
    "detect_count": 0,
    "tracked_count": 0,
    "frame_count": 0,
    "last_detect_frame": 0,
    "centroids": [],
    "tracked_ids": [],
    "split_x": 0,
    "frame_w": 0,
    "frame_h": 0,
    "video_ok": False,
    "last_event": "",
}

yolo_person = None
yolo_face = None
_models_loaded = False
def get_chart_data(request):
    global global_in, global_out

    # 从全局变量读取真实计数
    current = max(0, global_in - global_out)
    trend_data = []
    for i in range(9):
        trend_data.append(global_in // 9 * (i + 1) if global_in > 0 else 0)
    
    return JsonResponse({
        'in_count': global_in,
        'out_count': global_out,
        'current': current,
        'trend_data': trend_data,
        'realtime_data': [max(0, current - i * 2) for i in range(7)],
        'ratio_data': [
            {'value': global_in or 1, 'name': '进场'},
            {'value': global_out or 1, 'name': '离场'},
            {'value': current or 1, 'name': '在园'}
        ]
    })

def _ensure_models():
    global yolo_person, yolo_face, _models_loaded
    if _models_loaded:
        return
    os.makedirs(os.path.dirname(MODEL_PERSON), exist_ok=True)
    if not os.path.exists(MODEL_PERSON):
        yolo_person = YOLO("yolo11n.pt")
        yolo_person.save(MODEL_PERSON)
    else:
        yolo_person = YOLO(MODEL_PERSON)
    if os.path.exists(MODEL_FACE):
        yolo_face = YOLO(MODEL_FACE)
    else:
        yolo_face = yolo_person
    _models_loaded = True


def _is_valid_person(x1, y1, x2, y2, conf):
    """过滤非行人目标：置信度、最小高度、人体宽高比"""
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0 or conf < PERSON_CONF:
        return False
    if bh < MIN_BOX_H:
        return False
    aspect = bh / bw
    return MIN_ASPECT <= aspect <= MAX_ASPECT


def _foot_point(x1, y1, x2, y2):
    """用脚底中心点做越线判定，比框中心更准确"""
    return ((x1 + x2) // 2, y2)


class CentroidTracker:
    """基于欧氏距离的简易质心追踪器"""

    def __init__(self, max_disappeared=20, max_distance=80):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.deregistered = []
        self.registered = []
        self.age = {}

    def _register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.age[self.next_id] = 0
        self.registered.append((self.next_id, centroid))
        self.next_id += 1

    def _deregister(self, oid):
        if oid in self.objects:
            self.deregistered.append((oid, self.objects[oid]))
            del self.objects[oid]
        if oid in self.disappeared:
            del self.disappeared[oid]
        if oid in self.age:
            del self.age[oid]

    def update(self, centroids):
        self.deregistered = []
        self.registered = []
        for oid in self.age:
            self.age[oid] += 1

        if len(centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self.objects

        if len(self.objects) == 0:
            for c in centroids:
                self._register(c)
            return self.objects

        object_ids = list(self.objects.keys())
        object_pos = list(self.objects.values())
        D = np.zeros((len(object_pos), len(centroids)))
        for i, op in enumerate(object_pos):
            for j, nc in enumerate(centroids):
                D[i, j] = np.sqrt((op[0] - nc[0]) ** 2 + (op[1] - nc[1]) ** 2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            self.objects[oid] = centroids[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_pos))) - used_rows
        unused_cols = set(range(len(centroids))) - used_cols
        for row in unused_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self._deregister(oid)
        for col in unused_cols:
            self._register(centroids[col])
        return self.objects


tracker = CentroidTracker(max_disappeared=25, max_distance=100)
def get_stats(request):
    """统一返回所有数据：计数 + 图表 + 时间"""
    current = max(0, global_in - global_out)
    # 从数据库或内存获取历史趋势数据
    trend_data = [12, 23, 45, 56, 78, 89, 102, 95, 67]  # 这里替换成真实历史数据
    
    return JsonResponse({
        'in_count': global_in,
        'out_count': global_out,
        'current': current,
        'threshold': WARN_LIMIT,
        'trend_data': trend_data,
        'realtime_data': [45, 52, 48, 55, 60, 58, 62],  # 最近7个时间点的在园人数
        'ratio_data': [
            {'value': global_in or 1, 'name': '进场'},
            {'value': global_out or 1, 'name': '离场'},
            {'value': current or 1, 'name': '在园'}
        ],
        'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

def _check_warn(current_people):
    """在园人数达到阈值时立即触发预警，降至阈值以下后重置"""
    global _warn_active
    if current_people >= WARN_LIMIT:
        if not _warn_active:
            FlowWarn.objects.create(
                warn_people=current_people,
                warn_threshold=WARN_LIMIT,
                warn_content=f"人数超限：{current_people}人",
            )
            _warn_active = True
    else:
        _warn_active = False


def _reset_tracking_state():
    """重置追踪器状态"""
    tracker.objects.clear()
    tracker.disappeared.clear()
    tracker.age.clear()
    tracker.next_id = 0
    _track_meta.clear()


def _reset_session():
    """刷新/重新进入页面时清零所有计数与预警"""
    global global_in, global_out, last_stat_minute, last_detected_centroids, last_detected_boxes, _warn_active
    with lock:
        global_in = 0
        global_out = 0
        last_stat_minute = ""
        _warn_active = False
        last_detected_centroids = []
        last_detected_boxes = []
        _reset_tracking_state()
        FlowWarn.objects.all().delete()
        debug_info.update({
            "detect_count": 0,
            "tracked_count": 0,
            "frame_count": 0,
            "last_detect_frame": 0,
            "centroids": [],
            "tracked_ids": [],
            "last_event": "",
        })


def _update_line_counts(tracked, split_x, frame_counter):
    """仅依据脚底越线计数：右→左=进场，左→右=离场，每个ID各计一次"""
    global global_in, global_out

    for tid, pos in tracked.items():
        fx, _ = pos
        side = "L" if fx < split_x else "R"
        meta = _track_meta.setdefault(tid, {
            "side": side,
            "in_counted": False,
            "out_counted": False,
        })
        age = tracker.age.get(tid, 0)

        if age < MIN_STABLE_FRAMES:
            meta["side"] = side
            continue

        old_side = meta["side"]
        if old_side != side:
            if old_side == "R" and side == "L" and not meta["in_counted"]:
                global_in += 1
                meta["in_counted"] = True
                debug_info["last_event"] = f"IN ID={tid} @ frame {frame_counter}"
            elif old_side == "L" and side == "R" and not meta["out_counted"]:
                global_out += 1
                meta["out_counted"] = True
                debug_info["last_event"] = f"OUT ID={tid} @ frame {frame_counter}"
            meta["side"] = side

    for tid, _ in tracker.deregistered:
        _track_meta.pop(tid, None)


@require_POST
def set_threshold(request):
    global WARN_LIMIT
    try:
        val = int(request.POST.get("threshold", WARN_LIMIT))
        if 1 <= val <= 9999:
            WARN_LIMIT = val
            return JsonResponse({"ok": True, "threshold": WARN_LIMIT})
    except (TypeError, ValueError):
        pass
    return JsonResponse({"ok": False, "msg": "无效阈值"}, status=400)


def get_count(request):
    global global_in, global_out, WARN_LIMIT
    
    current = max(0, global_in - global_out)
    
    # 生成趋势数据（模拟最近9个时间点的进场数据）
    # 如果 global_in 是 0，就返回全 0
    if global_in > 0:
        trend_data = []
        for i in range(1, 10):
            trend_data.append(int(global_in * i / 9))
    else:
        trend_data = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # 生成实时在园数据（最近7个点）
    if current > 0:
        realtime_data = []
        for i in range(7):
            val = max(0, current - i * 2 + (i % 3))
            realtime_data.append(val)
    else:
        realtime_data = [0, 0, 0, 0, 0, 0, 0]
    
    warns = FlowWarn.objects.all().order_by("-warn_time")[:5]
    warn_data = [{"warn_content": w.warn_content, "warn_time": str(w.warn_time)} for w in warns]
    
    return JsonResponse({
        "in_count": global_in,
        "out_count": global_out,
        "current": current,
        "threshold": WARN_LIMIT,
        "warns": warn_data,
        "trend_data": trend_data,
        "realtime_data": realtime_data,
        "ratio_data": [
            {"value": global_in or 1, "name": "进场"},
            {"value": global_out or 1, "name": "离场"},
            {"value": current or 1, "name": "在园"}
        ]
    })

@require_POST
def reset_session(request):
    _reset_session()
    return JsonResponse({"ok": True})


@ensure_csrf_cookie
def index(request):
    _reset_session()
    current = 0
    warns = []
    faces = FaceCapture.objects.all().order_by("-capture_time")[:10]
    return render(request, "index.html", {
        "in_count": 0,
        "out_count": 0,
        "current": current,
        "warn_list": warns,
        "face_list": faces,
        "threshold": WARN_LIMIT,
    })


def _error_stream(message):
    error_img = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(error_img, message, (80, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    while True:
        _, jpg = cv2.imencode('.jpg', error_img)
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n'
        time.sleep(0.04)


@never_cache
def video_stream(request):
    global global_in, global_out, last_stat_minute, last_detected_centroids, last_detected_boxes

    _reset_session()
    _ensure_models()

    if not os.path.exists(VIDEO_PATH):
        return StreamingHttpResponse(
            _error_stream("VIDEO NOT FOUND"),
            content_type='multipart/x-mixed-replace; boundary=frame',
        )

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        return StreamingHttpResponse(
            _error_stream("CANNOT OPEN VIDEO"),
            content_type='multipart/x-mixed-replace; boundary=frame',
        )

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 25
    play_fps = min(original_fps, 30)
    target_delay = 1.0 / play_fps
    frame_counter = 0
    face_capture_counter = 0
    FACE_CAPTURE_EVERY_N_FRAMES = 30

    def generate():
        nonlocal cap, frame_counter, face_capture_counter
        global global_in, global_out, last_stat_minute, last_detected_centroids, last_detected_boxes, _warn_active
        last_frame_time = time.time()
        last_boxes = []
        prev_frame_pos = 0
        tracked = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            cur_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if cur_frame_pos < prev_frame_pos - 10:
                with lock:
                    global_in = 0
                    global_out = 0
                    _warn_active = False
                    _reset_tracking_state()
                    last_detected_centroids = []
                    last_detected_boxes = []
            prev_frame_pos = cur_frame_pos

            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < target_delay:
                time.sleep(target_delay - elapsed)
            last_frame_time = time.time()

            h, w = frame.shape[:2]
            split_x = int(w * SPLIT_LINE_RATIO)
            debug_info["frame_w"] = w
            debug_info["frame_h"] = h
            debug_info["split_x"] = split_x
            debug_info["video_ok"] = True

            frame_counter += 1
            debug_info["frame_count"] = frame_counter

            if frame_counter % DETECT_INTERVAL == 0:
                try:
                    results = yolo_person(frame, conf=PERSON_CONF, classes=[0], verbose=False)
                except Exception as e:
                    print(f"YOLO person inference error: {e}")
                    results = []
                centroids = []
                last_boxes = []
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) != 0:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        if not _is_valid_person(x1, y1, x2, y2, conf):
                            continue
                        foot = _foot_point(x1, y1, x2, y2)
                        centroids.append(foot)
                        last_boxes.append((x1, y1, x2, y2))
                debug_info["detect_count"] = len(centroids)
                debug_info["centroids"] = centroids
                if centroids:
                    debug_info["last_detect_frame"] = frame_counter
                last_detected_centroids = centroids
                last_detected_boxes = last_boxes
            else:
                centroids = last_detected_centroids
                last_boxes = last_detected_boxes

            with lock:
                tracked = tracker.update(centroids)
                debug_info["tracked_count"] = len(tracked)
                debug_info["tracked_ids"] = list(tracked.keys())
                _update_line_counts(tracked, split_x, frame_counter)

            face_capture_counter += 1
            if face_capture_counter >= FACE_CAPTURE_EVERY_N_FRAMES:
                face_capture_counter = 0
                try:
                    face_res = yolo_face(frame, conf=0.5, verbose=False)
                    for r in face_res:
                        for box in r.boxes:
                            fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
                            fx1, fy1 = max(0, fx1), max(0, fy1)
                            fx2, fy2 = min(w, fx2), min(h, fy2)
                            if fx2 - fx1 > 30 and fy2 - fy1 > 30:
                                face = frame[fy1:fy2, fx1:fx2]
                                ok, jpg = cv2.imencode('.jpg', face)
                                if ok:
                                    name = f"face_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
                                    f = ContentFile(jpg.tobytes(), name=name)
                                    FaceCapture.objects.create(face_img=f, scenic_point="景区入口")
                except Exception as e:
                    print(f"Face capture error: {e}")

            now_dt = datetime.now()
            current_min = now_dt.strftime("%Y-%m-%d %H:%M")
            current_people = max(0, global_in - global_out)
            with lock:
                _check_warn(current_people)
                if current_min != last_stat_minute:
                    FlowCount.objects.create(
                        in_num=global_in,
                        out_num=global_out,
                        current_people=current_people,
                    )
                    last_stat_minute = current_min

            cv2.line(frame, (split_x, 0), (split_x, h), (0, 255, 0), 1)
            for (x1, y1, x2, y2) in last_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)

            ret, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n'

    response = StreamingHttpResponse(
        generate(),
        content_type='multipart/x-mixed-replace; boundary=frame',
    )
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response


def face_list(request):
    faces = FaceCapture.objects.all().order_by("-capture_time")
    return render(request, "face_list.html", {"face_all": faces})


def warn_list(request):
    warns = FlowWarn.objects.all().order_by("-warn_time")
    return render(request, "warn_list.html", {"warn_all": warns})
def get_weather(request):
    # 直接用 wttr.in
    url = "https://wttr.in/Emeishan?format=j1&lang=zh"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        current = data.get('current_condition', [{}])[0]
        return JsonResponse({
            'temp': current.get('temp_C', '--'),
            'text': current.get('weatherDesc', [{}])[0].get('value', '未知'),
            'humidity': current.get('humidity', '--'),
            'windSpeed': current.get('windspeedKmph', '--')
        })
    except:
        return JsonResponse({'error': '天气获取失败'})
def clear_faces(request):
    FaceCapture.objects.all().delete()
    return JsonResponse({'ok': True})
