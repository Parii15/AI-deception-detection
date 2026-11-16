
import csv
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

from yt_dlp import YoutubeDL

ROOT = Path("data/DOLOS")
CSV_FILE = ROOT / "dolos_timestamps.csv"

OUT = ROOT / "bulk_processed"
SOURCES = OUT / "source_videos"
CLIPS = OUT / "clips"
REPORTS = OUT / "reports"


def to_seconds(value):
    value = value.strip()
    parts = value.split(":")
    if len(parts) == 1:
        return int(parts[0]) * 60
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    raise ValueError("Invalid timestamp: " + value)


def download_video(video_id):
    existing = list(SOURCES.glob(video_id + ".*"))
    existing = [p for p in existing if p.is_file()
                and not p.name.endswith((".part", ".ytdl"))]

    if existing:
        return existing[0]

    options = {
        "format": "best[ext=mp4]/best",
        "outtmpl": str(SOURCES / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "retries": 3,
        "quiet": False,
    }

    with YoutubeDL(options) as ydl:
        ydl.download([
            "https://www.youtube.com/watch?v=" + video_id
        ])

    existing = list(SOURCES.glob(video_id + ".*"))
    existing = [p for p in existing if p.is_file()
                and not p.name.endswith((".part", ".ytdl"))]

    if not existing:
        raise RuntimeError("Downloaded video not found")

    return existing[0]


def extract_clip(source, row):
    target = CLIPS / (row["file_name"] + ".mp4")

    if target.exists() and target.stat().st_size > 0:
        return "already_exists"

    start = to_seconds(row["start_time"])
    end = to_seconds(row["end_time"])

    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(source),
        "-t", str(end - start),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-c:a", "aac",
        str(target)
    ], check=True, capture_output=True)

    return "created"


def main():
    for folder in (SOURCES, CLIPS, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)

    with CSV_FILE.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    grouped = defaultdict(list)
    invalid = []

    for row in rows:
        try:
            start = to_seconds(row["start_time"])
            end = to_seconds(row["end_time"])

            if end <= start:
                raise ValueError("End time must be after start time")

            if row["label"].strip().lower() not in (
                "truth", "deception", "lie"
            ):
                raise ValueError("Unknown label")

            grouped[row["YT_Video_ID"].strip()].append(row)

        except Exception as e:
            invalid.append({
                "file_name": row.get("file_name", ""),
                "reason": str(e)
            })

    source_ids = sorted(grouped)


    results = []

    print("Total CSV rows:", len(rows))
    print("Valid source videos selected:", len(source_ids))
    print("Invalid rows:", len(invalid))

    for index, video_id in enumerate(source_ids, 1):
        print(f"\n[{index}/{len(source_ids)}] Processing {video_id}")

        try:
            source = download_video(video_id)
            print("Video downloaded:", source.name)

            for row in grouped[video_id]:
                try:
                    status = extract_clip(source, row)
                    print(row["file_name"], status)

                    results.append({
                        "clip": row["file_name"],
                        "video_id": video_id,
                        "label": row["label"],
                        "status": status,
                        "error": ""
                    })

                except Exception as e:
                    results.append({
                        "clip": row["file_name"],
                        "video_id": video_id,
                        "label": row["label"],
                        "status": "clip_error",
                        "error": str(e)
                    })

        except Exception as e:
            print("DOWNLOAD ERROR:", e)

            for row in grouped[video_id]:
                results.append({
                    "clip": row["file_name"],
                    "video_id": video_id,
                    "label": row["label"],
                    "status": "source_error",
                    "error": str(e)
                })

    with (REPORTS / "download_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip", "video_id", "label", "status", "error"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    with (REPORTS / "invalid_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_name", "reason"]
        )
        writer.writeheader()
        writer.writerows(invalid)

    print("\nTEST FINISHED")
    print("Report:", REPORTS / "download_report.csv")
    print("Extracted clips:", CLIPS)


if __name__ == "__main__":
    main()
