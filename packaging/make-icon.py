"""Иконка Feynt: финт — призрачный шеврон идёт впереди сплошного.

Черновик предлагает ход, целевая модель его подтверждает. Рисуется на скруглённом
квадрате в стиле macOS, один цвет плюс полупрозрачная копия.
"""
from pathlib import Path
import sys

from PIL import Image, ImageDraw

S = 1024
BG_TOP = (24, 26, 34)
BG_BOTTOM = (13, 14, 19)
ACCENT = (108, 168, 255)

img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
grad = Image.new("RGBA", (S, S))
gd = ImageDraw.Draw(grad)
for y in range(S):
    t = y / (S - 1)
    gd.line([(0, y), (S, y)], fill=tuple(
        int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)) + (255,))

# Скруглённый квадрат-маска: пропорции macOS Big Sur — радиус около 22% стороны.
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.225), fill=255)
img.paste(grad, (0, 0), mask)

def chevron(draw, cx, cy, half_w, half_h, thickness, colour):
    """Шеврон ">" как две толстые линии со скруглёнными стыками."""
    draw.line([(cx - half_w, cy - half_h), (cx + half_w, cy)],
              fill=colour, width=thickness, joint="curve")
    draw.line([(cx + half_w, cy), (cx - half_w, cy + half_h)],
              fill=colour, width=thickness, joint="curve")
    for point in ((cx - half_w, cy - half_h), (cx + half_w, cy), (cx - half_w, cy + half_h)):
        r = thickness // 2
        draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r], fill=colour)

layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(layer)
half_w, half_h, thick = int(S * 0.115), int(S * 0.175), int(S * 0.075)
# Призрачный шеврон впереди — предложенный ход, ещё не подтверждённый.
chevron(d, int(S * 0.60), S // 2, half_w, half_h, thick, ACCENT + (140,))
# Сплошной позади — то, что подтвердила целевая модель.
chevron(d, int(S * 0.42), S // 2, half_w, half_h, thick, ACCENT + (255,))
img.alpha_composite(layer)

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/private/tmp/feynt-icon/icon_1024.png")
img.save(out)
print("написано", out)
