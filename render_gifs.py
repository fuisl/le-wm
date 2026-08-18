"""Render the ground-truth and imagined-rollout schematic as two side-by-side
comparable GIFs, from the same data traffic-imagination.html's artifact uses
(traffic_data_sumo/rollout_viz.json). Pure PIL, no browser needed.
"""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DATA_PATH = "traffic_data_sumo/rollout_viz.json"
OUT_DIR = Path("traffic_data_sumo")
FONT_DIR = Path(__file__).parent / ".venv/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf"

W, H = 420, 470
CX, CY = 210, 260
JCT_HALF, LEG_LEN, LEG_HALF = 26, 118, 15
Y_MAX = 45
FRAME_MS = 480

# light-theme tokens, matching the HTML artifact
BG = "#F1F4F6"
ROAD = "#E7ECEF"
JUNCTION = "#DCE3E7"
BORDER = "#D7DEE3"
INK = "#172026"
INK_2 = "#47555F"
INK_3 = "#7C8991"
SERIES_GT = "#2a78d6"
SERIES_IM = "#eb6834"
SIGNAL_GO = "#2E9E4F"
SIGNAL_STOP = "#C4453D"

LEG_DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # top, right, bottom, left


def font(name, size):
    return ImageFont.truetype(str(FONT_DIR / name), size)

F_TITLE = font("DejaVuSansMono-Bold.ttf", 17)
F_SUB = font("DejaVuSans.ttf", 12)
F_LABEL = font("DejaVuSansMono-Bold.ttf", 12)
F_EDGE = font("DejaVuSansMono.ttf", 9)
F_VAL = font("DejaVuSansMono-Bold.ttf", 11)
F_FOOT = font("DejaVuSansMono.ttf", 12)


def bar_polygon(base, d, perp, offset_perp, length, thickness):
    bx, by = base[0] + perp[0] * offset_perp, base[1] + perp[1] * offset_perp
    ex, ey = bx + d[0] * length, by + d[1] * length
    hx, hy = perp[0] * thickness / 2, perp[1] * thickness / 2
    return [(bx - hx, by - hy), (ex - hx, ey - hy), (ex + hx, ey + hy), (bx + hx, by + hy)]


def draw_frame(frame, meta, values, series_color, series_label):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # header
    d.text((20, 18), "TRAFFIC IMAGINATION", font=F_TITLE, fill=INK)
    d.text((20, 42), f"cologne1 · {meta['controller']} · {series_label}", font=F_SUB, fill=INK_2)

    # roads + junction
    for dx, dy in LEG_DIRS:
        px, py = -dy, dx
        x1, y1 = CX + dx * JCT_HALF, CY + dy * JCT_HALF
        x2, y2 = CX + dx * (JCT_HALF + LEG_LEN), CY + dy * (JCT_HALF + LEG_LEN)
        poly = [
            (x1 + px * LEG_HALF, y1 + py * LEG_HALF), (x2 + px * LEG_HALF, y2 + py * LEG_HALF),
            (x2 - px * LEG_HALF, y2 - py * LEG_HALF), (x1 - px * LEG_HALF, y1 - py * LEG_HALF),
        ]
        d.polygon(poly, fill=ROAD, outline=BORDER)
    d.rounded_rectangle(
        [CX - JCT_HALF, CY - JCT_HALF, CX + JCT_HALF, CY + JCT_HALF],
        radius=4, fill=JUNCTION, outline=BORDER,
    )

    # bars, signal dots, labels
    for i, (dx, dy) in enumerate(LEG_DIRS):
        px, py = -dy, dx
        base = (CX + dx * JCT_HALF, CY + dy * JCT_HALF)
        length = min(max(values[i], 0) / Y_MAX, 1) * LEG_LEN
        d.polygon(bar_polygon(base, (dx, dy), (px, py), 0, length, 9), fill=series_color)

        dot_c = (base[0] + dx * 8, base[1] + dy * 8)
        dot_color = SIGNAL_GO if frame["green"][i] else SIGNAL_STOP
        d.ellipse([dot_c[0] - 5, dot_c[1] - 5, dot_c[0] + 5, dot_c[1] + 5], fill=dot_color, outline=BG, width=2)

        label_x = CX + dx * (JCT_HALF + LEG_LEN + 14)
        label_y = CY + dy * (JCT_HALF + LEG_LEN + 14)
        anchor = "mm"
        d.text((label_x, label_y - 6), meta["approach_labels"][i], font=F_LABEL, fill=INK, anchor=anchor)
        d.text((label_x, label_y + 8), meta["edge_ids"][i], font=F_EDGE, fill=INK_3, anchor=anchor)

        val_x = CX + dx * (JCT_HALF + LEG_LEN * 0.55) + px * (LEG_HALF + 12)
        val_y = CY + dy * (JCT_HALF + LEG_LEN * 0.55) + py * (LEG_HALF + 12)
        d.text((val_x, val_y), f"{values[i]:.0f}", font=F_VAL, fill=INK_2, anchor="mm")

    # footer readout
    region = frame["region"].replace("_", "-")
    d.rectangle([0, H - 44, W, H], fill="#FFFFFF", outline=BORDER)
    d.text((20, H - 30), f"t = {frame['sim_time_s']:>3d}s", font=F_FOOT, fill=INK)
    badge_color = SERIES_IM if frame["region"] == "imagined" else INK_3
    d.text((160, H - 30), region.upper(), font=F_FOOT, fill=badge_color)
    d.text((300, H - 30), f"phase {frame['phase']}", font=F_FOOT, fill=INK_3)

    return img


def build_gif(frames, meta, key, series_color, series_label, out_path):
    imgs = []
    for f in frames:
        values = f[key] if f[key] is not None else f["ground_truth"]
        imgs.append(draw_frame(f, meta, values, series_color, series_label))
    imgs[0].save(
        out_path, save_all=True, append_images=imgs[1:], duration=FRAME_MS, loop=0, optimize=False,
    )
    print(f"wrote {out_path} ({len(imgs)} frames)")


def main():
    data = json.loads(Path(DATA_PATH).read_text())
    meta = data["meta"]
    meta["edge_ids"] = ["-32038056#3", "23429231#1", "28198821#3", "27115123#3"]
    frames = data["frames"]

    build_gif(frames, meta, "ground_truth", SERIES_GT, "ground truth", OUT_DIR / "ground_truth.gif")
    build_gif(frames, meta, "imagined", SERIES_IM, "model's imagination", OUT_DIR / "imagined_rollout.gif")


if __name__ == "__main__":
    main()
