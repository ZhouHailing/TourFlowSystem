"""生成本地模拟景点客流视频（含行人移动，用于演示进出计数）"""
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "sim_video")
VIDEO_PATH = os.path.join(VIDEO_DIR, "scenic_sim.mp4")

W, H = 960, 540
FPS = 25
DURATION = 30
TOTAL = FPS * DURATION


def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (W, H))

    rng = np.random.default_rng(42)
    walkers = [
        {"y": 200 + i * 40, "speed": rng.integers(2, 5), "phase": i * 40, "h": 80}
        for i in range(8)
    ]

    for frame_idx in range(TOTAL):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = (40, 90, 40)
        cv2.rectangle(frame, (0, H - 120), (W, H), (60, 60, 60), -1)
        cv2.rectangle(frame, (80, 80), (300, 280), (80, 50, 30), -1)
        cv2.putText(frame, "Scenic Spot Entrance", (100, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 200, 0), 2)

        for i, w in enumerate(walkers):
            x = (frame_idx * w["speed"] + w["phase"]) % (W + 200) - 100
            y = w["y"]
            h = w["h"]
            cv2.rectangle(frame, (x, y), (x + 35, y + h), (200, 180, 160), -1)
            cv2.circle(frame, (x + 17, y + 18), 14, (220, 200, 180), -1)

        cv2.putText(frame, f"Frame {frame_idx}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        out.write(frame)

    out.release()
    print(f"视频已生成: {VIDEO_PATH} ({TOTAL} frames)")


if __name__ == "__main__":
    main()
