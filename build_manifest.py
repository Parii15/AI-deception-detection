
import pandas as pd
from pathlib import Path

ROOT = Path("data/DOLOS/bulk_processed")

csv_path = Path("data/DOLOS/dolos_timestamps_validated.csv")
video_dir = ROOT / "clips"
audio_dir = ROOT / "audio"
frame_dir = ROOT / "face_frames"

df = pd.read_csv(csv_path)

# Normalize clip names and labels
df["file_name"] = df["file_name"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.lower()

df = df[df["label"].isin(["truth", "deception"])].copy()

# Build exact filename lookups
videos = {p.stem: p for p in video_dir.rglob("*.mp4")}
audios = {p.stem: p for p in audio_dir.rglob("*.wav")}

records = []

for _, row in df.iterrows():
    name = row["file_name"]
    video = videos.get(name)
    audio = audios.get(name)
    frames = frame_dir / name

    frame_files = (
        list(frames.glob("*.jpg")) if frames.is_dir() else []
    )

    records.append({
        "clip": name,
        "youtube_id": row["YT_Video_ID"],
        "label": row["label"],
        "video": str(video) if video else "",
        "audio": str(audio) if audio else "",
        "frame_folder": str(frames) if len(frame_files) == 64 else "",
        "frame_count": len(frame_files),
        "matched": bool(video and audio and len(frame_files) == 64)
    })

manifest = pd.DataFrame(records)

out_dir = Path("data/DOLOS/bulk_processed/manifests")
out_dir.mkdir(parents=True, exist_ok=True)

manifest.to_csv(out_dir / "all_clips_manifest.csv", index=False)

matched = manifest[manifest["matched"]]
unmatched = manifest[~manifest["matched"]]

matched.to_csv(out_dir / "matched_clips.csv", index=False)
unmatched.to_csv(out_dir / "unmatched_clips.csv", index=False)

print("\n--- MATCHING REPORT ---")
print("Validated labelled rows:", len(manifest))
print("Fully matched clips:", len(matched))
print("Unmatched clips:", len(unmatched))
print("\n--- LABEL DISTRIBUTION ---")
print(matched["label"].value_counts())
print("\nManifest saved:", out_dir)
