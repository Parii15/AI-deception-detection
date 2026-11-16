
import cv2
import numpy as np
from pathlib import Path

VIDEO_DIR = Path("data/DOLOS/bulk_processed/clips")
OUTPUT_DIR = Path("data/DOLOS/bulk_processed/face_frames")

TARGET_FRAMES = 64
SIZE = (224, 224)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

videos = list(VIDEO_DIR.rglob("*.mp4"))
success = 0
failed = []

print("Total videos:", len(videos))

for i, video in enumerate(videos, 1):
    output_folder = OUTPUT_DIR / video.stem
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened() or total <= 0:
        failed.append(video.name)
        cap.release()
        continue

    indices = np.linspace(
        0, total - 1, TARGET_FRAMES
    ).astype(int)

    ok = True

    for j, frame_index in enumerate(indices, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ret, frame = cap.read()

        if not ret:
            ok = False
            break

        frame = cv2.resize(frame, SIZE)

        cv2.imwrite(
            str(output_folder / f"frame_{j:04d}.jpg"),
            frame
        )

    cap.release()

    if ok:
        success += 1
    else:
        failed.append(video.name)

    if i % 25 == 0:
        print(f"Processed {i}/{len(videos)}")

print("\nFRAME EXTRACTION COMPLETE")
print("Successful videos:", success)
print("Failed videos:", len(failed))
print("Output:", OUTPUT_DIR)
