import argparse
import csv
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def select_mp4_members(zip_path, limit):
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            info for info in zf.infolist()
            if info.filename.lower().endswith(".mp4")
            and not info.filename.startswith("__MACOSX/")
            and info.file_size > 0
        ]
    members.sort(key=lambda info: (-info.file_size, info.filename))
    return members[:limit]


def run_command(cmd):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def count_video_frames(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        result = run_command(cmd)
    except subprocess.CalledProcessError:
        return 0
    value = result.stdout.strip().splitlines()[0] if result.stdout.strip() else "0"
    try:
        return int(value)
    except ValueError:
        return 0


def transcode_video(src, dst, height, width, frames, fps):
    vf = (
        f"fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"trim=end_frame={frames},"
        "setpts=PTS-STARTPTS"
    )
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src),
        "-vf", vf,
        "-frames:v", str(frames),
        "-an",
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    run_command(cmd)


def prepare_smoke_dataset(zip_path, output_dir, num_samples, height, width, frames, fps, prompt):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required to prepare the smoke dataset.")

    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    selected = select_mp4_members(zip_path, limit=max(num_samples * 8, num_samples))
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            for info in selected:
                if len(rows) >= num_samples:
                    break
                src = tmp_dir / Path(info.filename).name
                with zf.open(info) as source, src.open("wb") as target:
                    shutil.copyfileobj(source, target)

                out_name = f"smoke_{len(rows):03d}.mp4"
                dst = video_dir / out_name
                try:
                    transcode_video(src, dst, height=height, width=width, frames=frames, fps=fps)
                except subprocess.CalledProcessError:
                    if dst.exists():
                        dst.unlink()
                    continue
                if count_video_frames(dst) < frames:
                    dst.unlink(missing_ok=True)
                    continue
                rows.append({"video": f"videos/{out_name}", "prompt": prompt})

    if len(rows) == 0:
        raise RuntimeError("No usable videos were found in the zip archive.")

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "prompt"])
        writer.writeheader()
        writer.writerows(rows)
    return metadata_path, rows


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a small 480x832@81 HQ-VSR smoke dataset for Wan BSA training.")
    parser.add_argument("--zip_path", default="/Users/zonghuiliu/Downloads/HQ-VSR.zip")
    parser.add_argument("--output_dir", default="outputs/hq_vsr_bsa_smoke")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--prompt", default="A high quality video of a toy rotating horizontally 360 degrees with stable color and fine details.")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_path, rows = prepare_smoke_dataset(
        zip_path=args.zip_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        height=args.height,
        width=args.width,
        frames=args.frames,
        fps=args.fps,
        prompt=args.prompt,
    )
    print(f"Prepared {len(rows)} videos")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
