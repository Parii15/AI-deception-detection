
import pandas as pd
from pathlib import Path

source = Path("data/DOLOS/dolos_timestamps.csv")
output = Path("data/DOLOS/dolos_download.csv")

df = pd.read_csv(source, skipinitialspace=True)
df.columns = df.columns.str.strip()

def to_seconds(value):
    parts = str(value).strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return float(value)

result = pd.DataFrame({
    "yt_video_id": df["YT_Video_ID"].astype(str).str.strip(),
    "file_name": df["file_name"].astype(str).str.strip(),
    "start_time": df["start_time"].apply(to_seconds),
    "end_time": df["end_time"].apply(to_seconds),
})

result.to_csv(output, index=False)
print("Converted CSV saved to:", output)
print(result.head())
print("Total clips:", len(result))
