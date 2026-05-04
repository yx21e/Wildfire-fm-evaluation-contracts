from __future__ import annotations

import math
from pathlib import Path


def esc(text: object) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def mix(c0: tuple[float, float, float], c1: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    t = clamp(t)
    return tuple(c0[i] * (1.0 - t) + c1[i] * t for i in range(3))


def text_width(text: str, size: float, bold: bool = False) -> float:
    width = 0.0
    for ch in text:
        if ch == " ":
            width += 0.28
        elif ch in "ilI.,:;!|":
            width += 0.25
        elif ch in "mwMW@":
            width += 0.82
        elif ch.isupper():
            width += 0.62
        else:
            width += 0.52
    if bold:
        width *= 1.05
    return width * size


class PdfCanvas:
    def __init__(self, width: float = 900, height: float = 560) -> None:
        self.width = width
        self.height = height
        self.ops: list[str] = []

    def _color(self, color: tuple[float, float, float], stroke: bool = False) -> str:
        return f"{color[0]:.4f} {color[1]:.4f} {color[2]:.4f} {'RG' if stroke else 'rg'}"

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: tuple[float, float, float] | None = None,
        stroke: tuple[float, float, float] | None = None,
        lw: float = 0.7,
    ) -> None:
        if fill is not None and stroke is None:
            self.ops.append(f"{self._color(fill)} {x:.2f} {y:.2f} {w:.2f} {h:.2f} re f")
        elif fill is None and stroke is not None:
            self.ops.append(f"{lw:.2f} w {self._color(stroke, True)} {x:.2f} {y:.2f} {w:.2f} {h:.2f} re S")
        elif fill is not None and stroke is not None:
            self.ops.append(
                f"{lw:.2f} w {self._color(fill)} {self._color(stroke, True)} "
                f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re B"
            )

    def line(
        self,
        points: list[tuple[float, float]],
        color: tuple[float, float, float] = (0, 0, 0),
        lw: float = 1.0,
    ) -> None:
        if len(points) < 2:
            return
        cmds = [f"{points[0][0]:.2f} {points[0][1]:.2f} m"]
        cmds.extend(f"{x:.2f} {y:.2f} l" for x, y in points[1:])
        self.ops.append(f"{lw:.2f} w {self._color(color, True)} " + " ".join(cmds) + " S")

    def text(
        self,
        x: float,
        y: float,
        text: object,
        size: float = 9,
        color: tuple[float, float, float] = (0, 0, 0),
        align: str = "left",
        bold: bool = False,
    ) -> None:
        s = str(text)
        w = text_width(s, size, bold)
        if align == "center":
            x -= w / 2.0
        elif align == "right":
            x -= w
        font = "F2" if bold else "F1"
        self.ops.append(
            f"BT /{font} {size:.2f} Tf {self._color(color)} "
            f"1 0 0 1 {x:.2f} {y:.2f} Tm ({esc(s)}) Tj ET"
        )

    def text_rotated(
        self,
        x: float,
        y: float,
        text: object,
        angle_deg: float,
        size: float = 9,
        color: tuple[float, float, float] = (0, 0, 0),
        align: str = "left",
        bold: bool = False,
    ) -> None:
        s = str(text)
        w = text_width(s, size, bold)
        angle = math.radians(angle_deg)
        if align == "center":
            x -= math.cos(angle) * w / 2.0
            y -= math.sin(angle) * w / 2.0
        elif align == "right":
            x -= math.cos(angle) * w
            y -= math.sin(angle) * w
        a = math.cos(angle)
        b = math.sin(angle)
        c = -math.sin(angle)
        d = math.cos(angle)
        font = "F2" if bold else "F1"
        self.ops.append(
            f"BT /{font} {size:.2f} Tf {self._color(color)} "
            f"{a:.5f} {b:.5f} {c:.5f} {d:.5f} {x:.2f} {y:.2f} Tm ({esc(s)}) Tj ET"
        )

    def save(self, path: Path) -> None:
        content = "\n".join(self.ops).encode("latin-1", errors="replace")
        objects: list[bytes] = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:.0f} {self.height:.0f}] "
                f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
            ).encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        ]
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for i, obj in enumerate(objects, start=1):
            offsets.append(len(out))
            out.extend(f"{i} 0 obj\n".encode())
            out.extend(obj)
            out.extend(b"\nendobj\n")
        xref_pos = len(out)
        out.extend(f"xref\n0 {len(objects)+1}\n".encode())
        out.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            out.extend(f"{offset:010d} 00000 n \n".encode())
        out.extend(
            f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
        )
        path.write_bytes(bytes(out))


def draw_axes(c: PdfCanvas, x0: float, y0: float, w: float, h: float, ymax: float, ticks: list[float]) -> None:
    c.line([(x0, y0), (x0, y0 + h)], color=(0.15, 0.15, 0.15), lw=0.8)
    c.line([(x0, y0), (x0 + w, y0)], color=(0.15, 0.15, 0.15), lw=0.8)
    for tick in ticks:
        y = y0 + h * tick / ymax
        c.line([(x0 - 4, y), (x0 + w, y)], color=(0.86, 0.86, 0.86), lw=0.45)
        c.text(x0 - 8, y - 3, f"{tick:g}", size=7, color=(0.25, 0.25, 0.25), align="right")
