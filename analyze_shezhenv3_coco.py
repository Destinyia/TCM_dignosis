#!/usr/bin/env python3
import json
import math
import os
import random
from collections import Counter, defaultdict

try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


ROOT = "/mnt/d/workspace/DM/TCM_dignosis/datasets/shezhenv3-coco"
OUT_DIR = "/mnt/d/workspace/DM/TCM_dignosis/runs/shezhenv3_coco_analysis_2026-02-04"
SPLITS = ["train", "val", "test"]
RANDOM_SEED = 42


def load_coco(split):
    ann_path = os.path.join(ROOT, split, "annotations", f"{split}.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_categories(data):
    cats = data.get("categories", [])
    return {c["id"]: c.get("name", str(c["id"])) for c in cats}


def collect_stats():
    per_split = {}
    overall_counts = Counter()
    overall_bbox_areas = []
    overall_bbox_w = []
    overall_bbox_h = []

    for split in SPLITS:
        data = load_coco(split)
        cat_map = get_categories(data)
        counts = Counter()
        bbox_areas = []
        bbox_w = []
        bbox_h = []

        for ann in data.get("annotations", []):
            cid = ann.get("category_id")
            cname = cat_map.get(cid, f"unknown_{cid}")
            counts[cname] += 1
            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4:
                _, _, w, h = bbox
                if w > 0 and h > 0:
                    bbox_areas.append(w * h)
                    bbox_w.append(w)
                    bbox_h.append(h)

        per_split[split] = {
            "counts": counts,
            "bbox_areas": bbox_areas,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "cat_map": cat_map,
        }

        overall_counts.update(counts)
        overall_bbox_areas.extend(bbox_areas)
        overall_bbox_w.extend(bbox_w)
        overall_bbox_h.extend(bbox_h)

    return per_split, overall_counts, overall_bbox_areas, overall_bbox_w, overall_bbox_h


def svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def svg_header(w, h):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'


def svg_footer():
    return "</svg>"


def write_svg(out_path, content):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def svg_barh(labels, values, title, out_path):
    items = list(zip(labels, values))
    n = len(items)
    width = 1200
    bar_h = 18
    gap = 6
    left = 240
    right = 40
    top = 60
    bottom = 40
    height = top + bottom + n * (bar_h + gap)
    maxv = max(values) if values else 1

    lines = [svg_header(width, height)]
    lines.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    lines.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" fill="#222">{svg_escape(title)}</text>')
    lines.append(f'<line x1="{left}" y1="{top-8}" x2="{left}" y2="{height-bottom+8}" stroke="#888" stroke-width="1"/>')

    for i, (label, value) in enumerate(items):
        y = top + i * (bar_h + gap)
        bar_w = (width - left - right) * (value / maxv) if maxv else 0
        lines.append(
            f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="{bar_h}" fill="#4C78A8"/>'
        )
        lines.append(
            f'<text x="{left-8}" y="{y+3}" text-anchor="end" font-size="12" dominant-baseline="hanging" fill="#222">{svg_escape(label)}</text>'
        )
        lines.append(
            f'<text x="{left+bar_w+6}" y="{y+3}" text-anchor="start" font-size="11" dominant-baseline="hanging" fill="#444">{value}</text>'
        )

    lines.append(svg_footer())
    write_svg(out_path, "\n".join(lines))


def svg_hist_log(values, title, out_path, bins=40):
    vals = [v for v in values if v > 0]
    if not vals:
        return
    min_v = max(min(vals), 1.0)
    max_v = max(vals)
    log_min = math.log10(min_v)
    log_max = math.log10(max_v)
    log_range = max(log_max - log_min, 1e-9)

    counts = [0] * bins
    for v in vals:
        lv = math.log10(v)
        idx = int((lv - log_min) / log_range * bins)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        counts[idx] += 1

    width = 1200
    height = 500
    left = 80
    right = 40
    top = 60
    bottom = 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    max_c = max(counts) if counts else 1

    lines = [svg_header(width, height)]
    lines.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    lines.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" fill="#222">{svg_escape(title)}</text>')
    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#888" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#888" stroke-width="1"/>')

    for i in range(bins):
        x0 = left + ((i) / bins) * plot_w
        x1 = left + ((i + 1) / bins) * plot_w
        bar_w = max(1.0, x1 - x0)
        bar_h = (counts[i] / max_c) * plot_h if max_c else 0
        y0 = top + plot_h - bar_h
        lines.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#F58518" opacity="0.85"/>'
        )

    # X axis ticks (log)
    for t in range(5):
        lv = log_min + t * (log_range / 4)
        v = 10 ** lv
        x = left + ((lv - log_min) / log_range) * plot_w
        lines.append(f'<line x1="{x:.2f}" y1="{top+plot_h}" x2="{x:.2f}" y2="{top+plot_h+6}" stroke="#666" stroke-width="1"/>')
        label = f"{int(v)}"
        lines.append(f'<text x="{x:.2f}" y="{top+plot_h+10}" text-anchor="middle" font-size="11" dominant-baseline="hanging" fill="#444">{label}</text>')

    lines.append(f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" font-size="12" fill="#444">bbox area (log scale)</text>')
    lines.append(f'<text x="16" y="{top+plot_h/2}" text-anchor="middle" font-size="12" fill="#444" transform="rotate(-90 16 {top+plot_h/2})">count</text>')
    lines.append(svg_footer())
    write_svg(out_path, "\n".join(lines))


def svg_hist_linear(values, title, out_path, bins=40):
    vals = [v for v in values if v > 0]
    if not vals:
        return
    min_v = min(vals)
    max_v = max(vals)
    rng = max_v - min_v if max_v != min_v else 1.0

    counts = [0] * bins
    for v in vals:
        idx = int((v - min_v) / rng * bins)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        counts[idx] += 1

    width = 1200
    height = 500
    left = 80
    right = 40
    top = 60
    bottom = 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    max_c = max(counts) if counts else 1

    lines = [svg_header(width, height)]
    lines.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    lines.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" fill="#222">{svg_escape(title)}</text>')
    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#888" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#888" stroke-width="1"/>')

    for i in range(bins):
        x0 = left + (i / bins) * plot_w
        x1 = left + ((i + 1) / bins) * plot_w
        bar_w = max(1.0, x1 - x0)
        bar_h = (counts[i] / max_c) * plot_h if max_c else 0
        y0 = top + plot_h - bar_h
        lines.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#54A24B" opacity="0.85"/>'
        )

    # X axis ticks
    for t in range(5):
        v = min_v + t * (rng / 4)
        x = left + (t / 4) * plot_w
        lines.append(f'<line x1="{x:.2f}" y1="{top+plot_h}" x2="{x:.2f}" y2="{top+plot_h+6}" stroke="#666" stroke-width="1"/>')
        label = f"{int(v)}"
        lines.append(f'<text x="{x:.2f}" y="{top+plot_h+10}" text-anchor="middle" font-size="11" dominant-baseline="hanging" fill="#444">{label}</text>')

    lines.append(f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" font-size="12" fill="#444">pixels</text>')
    lines.append(f'<text x="16" y="{top+plot_h/2}" text-anchor="middle" font-size="12" fill="#444" transform="rotate(-90 16 {top+plot_h/2})">count</text>')
    lines.append(svg_footer())
    write_svg(out_path, "\n".join(lines))


def load_image_annotations(split):
    data = load_coco(split)
    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)
    cat_map = get_categories(data)
    return images, anns_by_image, cat_map


def collect_class_samples(class_names):
    samples = []
    for split in SPLITS:
        data = load_coco(split)
        images = {img["id"]: img for img in data.get("images", [])}
        cat_map = get_categories(data)
        name_to_id = {v: k for k, v in cat_map.items()}
        class_ids = {name_to_id.get(n) for n in class_names}
        class_ids.discard(None)

        anns_by_image = defaultdict(list)
        for ann in data.get("annotations", []):
            if ann.get("category_id") in class_ids:
                anns_by_image[ann["image_id"]].append(ann)

        for image_id, anns in anns_by_image.items():
            img = images.get(image_id)
            if not img:
                continue
            samples.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "file_name": img.get("file_name"),
                    "width": img.get("width"),
                    "height": img.get("height"),
                    "anns": anns,
                    "cat_map": cat_map,
                }
            )
    return samples


def build_color_map(cat_map):
    random.seed(RANDOM_SEED)
    colors = {}
    for cid, name in sorted(cat_map.items()):
        r = random.randint(50, 230)
        g = random.randint(50, 230)
        b = random.randint(50, 230)
        colors[cid] = (r, g, b)
    return colors


def draw_sample_grid_svg(split, out_path, grid=5, cell_size=320):
    images, anns_by_image, cat_map = load_image_annotations(split)
    colors = build_color_map(cat_map)

    # Filter images that have at least 1 annotation
    img_ids = [img_id for img_id in images if anns_by_image.get(img_id)]
    if len(img_ids) < grid * grid:
        sample_ids = img_ids
    else:
        random.seed(RANDOM_SEED)
        sample_ids = random.sample(img_ids, grid * grid)

    grid_w = grid * cell_size
    grid_h = grid * cell_size
    lines = [svg_header(grid_w, grid_h)]
    lines.append(f'<rect width="{grid_w}" height="{grid_h}" fill="#141414"/>')

    for idx, img_id in enumerate(sample_ids):
        img_meta = images[img_id]
        file_name = img_meta["file_name"]
        img_w = img_meta.get("width", 1)
        img_h = img_meta.get("height", 1)
        img_rel = os.path.join("datasets", "shezhenv3-coco", split, "images", file_name)

        scale = min(cell_size / img_w, cell_size / img_h)
        new_w = max(1, int(round(img_w * scale)))
        new_h = max(1, int(round(img_h * scale)))

        col = idx % grid
        row = idx // grid
        x0 = col * cell_size + (cell_size - new_w) // 2
        y0 = row * cell_size + (cell_size - new_h) // 2

        lines.append(
            f'<image x="{x0}" y="{y0}" width="{new_w}" height="{new_h}" href="{svg_escape(img_rel)}" preserveAspectRatio="xMidYMid meet"/>'
        )

        for ann in anns_by_image[img_id]:
            cid = ann.get("category_id")
            cname = cat_map.get(cid, f"unknown_{cid}")
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            bx, by, bw, bh = bbox
            sx = x0 + bx * scale
            sy = y0 + by * scale
            sw = bw * scale
            sh = bh * scale
            color = colors.get(cid, (255, 255, 0))
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            lines.append(
                f'<rect x="{sx:.2f}" y="{sy:.2f}" width="{sw:.2f}" height="{sh:.2f}" fill="none" stroke="{color_hex}" stroke-width="2"/>'
            )
            label_x = sx + 2
            label_y = sy + 2
            lines.append(
                f'<rect x="{label_x-1:.2f}" y="{label_y-1:.2f}" width="{6 + 6*len(cname):.2f}" height="12" fill="#000000" opacity="0.6"/>'
            )
            lines.append(
                f'<text x="{label_x:.2f}" y="{label_y:.2f}" font-size="10" fill="{color_hex}" dominant-baseline="hanging">{svg_escape(cname)}</text>'
            )

    lines.append(svg_footer())
    write_svg(out_path, "\n".join(lines))


def draw_sample_grid_png(split, out_path, grid=5, cell_size=320):
    if not HAVE_PIL:
        raise RuntimeError("Pillow is required for PNG output. Install with: pip install Pillow")

    images, anns_by_image, cat_map = load_image_annotations(split)
    colors = build_color_map(cat_map)

    img_ids = [img_id for img_id in images if anns_by_image.get(img_id)]
    if len(img_ids) < grid * grid:
        sample_ids = img_ids
    else:
        random.seed(RANDOM_SEED)
        sample_ids = random.sample(img_ids, grid * grid)

    grid_w = grid * cell_size
    grid_h = grid * cell_size
    canvas = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    font = ImageFont.load_default()

    for idx, img_id in enumerate(sample_ids):
        img_meta = images[img_id]
        file_name = img_meta["file_name"]
        img_path = os.path.join(ROOT, split, "images", file_name)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")

        w, h = img.size
        scale = min(cell_size / w, cell_size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        col = idx % grid
        row = idx // grid
        x0 = col * cell_size + (cell_size - new_w) // 2
        y0 = row * cell_size + (cell_size - new_h) // 2
        canvas.paste(img_resized, (x0, y0))

        draw = ImageDraw.Draw(canvas)
        for ann in anns_by_image[img_id]:
            cid = ann.get("category_id")
            cname = cat_map.get(cid, f"unknown_{cid}")
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            bx, by, bw, bh = bbox
            sx = x0 + bx * scale
            sy = y0 + by * scale
            sw = bw * scale
            sh = bh * scale
            color = colors.get(cid, (255, 255, 0))
            draw.rectangle([sx, sy, sx + sw, sy + sh], outline=color, width=2)
            tx = sx + 2
            ty = sy + 2
            draw.rectangle([tx - 1, ty - 1, tx + 6 + 6 * len(cname), ty + 12], fill=(0, 0, 0))
            draw.text((tx, ty), cname, fill=color, font=font)

    canvas.save(out_path, "PNG")


def draw_class_grid_png(class_names, out_path, grid=5, cell_size=320):
    if not HAVE_PIL:
        raise RuntimeError("Pillow is required for PNG output. Install with: pip install Pillow")

    samples = collect_class_samples(class_names)
    if not samples:
        raise RuntimeError("No samples found for requested classes")

    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    total_cells = grid * grid
    if len(samples) < total_cells:
        # Repeat samples to fill the grid
        reps = (total_cells + len(samples) - 1) // len(samples)
        samples = (samples * reps)[:total_cells]
    else:
        samples = samples[:total_cells]

    grid_w = grid * cell_size
    grid_h = grid * cell_size
    canvas = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    font = ImageFont.load_default()

    for idx, sample in enumerate(samples):
        split = sample["split"]
        file_name = sample["file_name"]
        img_path = os.path.join(ROOT, split, "images", file_name)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")

        w, h = img.size
        scale = min(cell_size / w, cell_size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        col = idx % grid
        row = idx // grid
        x0 = col * cell_size + (cell_size - new_w) // 2
        y0 = row * cell_size + (cell_size - new_h) // 2
        canvas.paste(img_resized, (x0, y0))

        draw = ImageDraw.Draw(canvas)
        cat_map = sample["cat_map"]
        colors = build_color_map(cat_map)
        for ann in sample["anns"]:
            cid = ann.get("category_id")
            cname = cat_map.get(cid, f"unknown_{cid}")
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            bx, by, bw, bh = bbox
            sx = x0 + bx * scale
            sy = y0 + by * scale
            sw = bw * scale
            sh = bh * scale
            color = colors.get(cid, (255, 255, 0))
            draw.rectangle([sx, sy, sx + sw, sy + sh], outline=color, width=2)
            tx = sx + 2
            ty = sy + 2
            draw.rectangle([tx - 1, ty - 1, tx + 6 + 6 * len(cname), ty + 12], fill=(0, 0, 0))
            draw.text((tx, ty), cname, fill=color, font=font)

        # small corner label with split
        draw.rectangle([x0 + 2, y0 + 2, x0 + 44, y0 + 14], fill=(0, 0, 0))
        draw.text((x0 + 4, y0 + 3), split, fill=(255, 255, 255), font=font)

    canvas.save(out_path, "PNG")


def main():
    ensure_dir(OUT_DIR)

    per_split, overall_counts, overall_bbox_areas, overall_bbox_w, overall_bbox_h = collect_stats()

    svg_barh(
        [k for k, _ in overall_counts.most_common()],
        [v for _, v in overall_counts.most_common()],
        "Class Distribution (Overall)",
        os.path.join(OUT_DIR, "class_distribution_overall.svg"),
    )

    for split in SPLITS:
        counts = per_split[split]["counts"].most_common()
        svg_barh(
            [k for k, _ in counts],
            [v for _, v in counts],
            f"Class Distribution ({split})",
            os.path.join(OUT_DIR, f"class_distribution_{split}.svg"),
        )

    svg_hist_log(
        overall_bbox_areas,
        "BBox Area Histogram (Overall, Log Scale)",
        os.path.join(OUT_DIR, "bbox_area_hist_overall.svg"),
    )

    for split in SPLITS:
        svg_hist_log(
            per_split[split]["bbox_areas"],
            f"BBox Area Histogram ({split}, Log Scale)",
            os.path.join(OUT_DIR, f"bbox_area_hist_{split}.svg"),
        )

    svg_hist_linear(
        overall_bbox_w,
        "BBox Width Histogram (Overall)",
        os.path.join(OUT_DIR, "bbox_width_hist_overall.svg"),
    )

    svg_hist_linear(
        overall_bbox_h,
        "BBox Height Histogram (Overall)",
        os.path.join(OUT_DIR, "bbox_height_hist_overall.svg"),
    )

    draw_sample_grid_png(
        "train",
        os.path.join(OUT_DIR, "sample_5x5_train.png"),
        grid=5,
        cell_size=320,
    )

    draw_class_grid_png(
        ["xinfeitu", "shenqutu", "gandantu"],
        os.path.join(OUT_DIR, "sample_5x5_xinfeitu_shenqutu_gandantu.png"),
        grid=5,
        cell_size=320,
    )

    print("saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
