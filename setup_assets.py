"""下载 YOLO 模型权重与模拟景点视频"""
import os
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "yolov11_model")
VIDEO_DIR = os.path.join(BASE_DIR, "sim_video")
VIDEO_PATH = os.path.join(VIDEO_DIR, "scenic_sim.mp4")

MODELS = {
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "yolo11n-face.pt": "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11n-face.pt",
}

# 景点人流示例视频（多个备用源）
VIDEO_URLS = [
    "https://cdn.jsdelivr.net/gh/intel-iot-devkit/sample-videos@master/people-detection.mp4",
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
]


def download(url, dest):
    if os.path.exists(dest):
        print(f"已存在，跳过: {dest}")
        return True
    print(f"正在下载: {url}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(dest, "wb") as f:
            f.write(resp.read())
    print(f"下载完成: {dest}")
    return True


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    for name, url in MODELS.items():
        download(url, os.path.join(MODEL_DIR, name))
    for url in VIDEO_URLS:
        try:
            download(url, VIDEO_PATH)
            break
        except Exception as e:
            print(f"下载失败 {url}: {e}")
    else:
        print("警告: 视频下载失败，请手动将 mp4 放入 sim_video/scenic_sim.mp4")
    print("所有资源准备完毕！")


if __name__ == "__main__":
    main()
