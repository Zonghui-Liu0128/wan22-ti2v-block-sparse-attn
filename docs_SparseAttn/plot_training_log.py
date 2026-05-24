import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def series(rows, key):
    points = []
    for row in rows:
        if key in row and row[key] is not None:
            points.append((float(row["step"]), float(row[key])))
    return points


def polyline(points, width, height, margin):
    if not points:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1e-12)
    coords = []
    for x, y in points:
        px = margin + (x - x_min) / x_span * (width - 2 * margin)
        py = height - margin - (y - y_min) / y_span * (height - 2 * margin)
        coords.append(f"{px:.2f},{py:.2f}")
    return " ".join(coords)


def write_svg(points, title, ylabel, output_path: Path):
    width, height, margin = 900, 420, 52
    path_points = polyline(points, width, height, margin)
    if not path_points:
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="{margin}" font-size="20">{title}: no data</text>
</svg>
"""
    else:
        y_values = [p[1] for p in points]
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#555"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#555"/>
<polyline points="{path_points}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
<text x="{margin}" y="30" font-size="20" font-family="sans-serif">{title}</text>
<text x="{margin}" y="{height - 14}" font-size="12" font-family="sans-serif">step</text>
<text x="14" y="{margin}" font-size="12" font-family="sans-serif">{ylabel}</text>
<text x="{width - margin - 140}" y="30" font-size="12" font-family="sans-serif">min={min(y_values):.6g}, max={max(y_values):.6g}</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.log_file)
    output_dir = args.output_dir or args.log_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    charts = [
        ("loss", "Training Loss", "loss", output_dir / "loss_curve.svg"),
        ("seconds_per_step", "Seconds Per Step", "seconds", output_dir / "seconds_per_step_curve.svg"),
        ("frames_per_second", "Frames Per Second", "frames/s", output_dir / "frames_per_second_curve.svg"),
        ("sparsity", "Block Sparse Sparsity", "sparsity", output_dir / "sparsity_curve.svg"),
    ]
    for key, title, ylabel, path in charts:
        write_svg(series(rows, key), title, ylabel, path)
        print(path)


if __name__ == "__main__":
    main()
