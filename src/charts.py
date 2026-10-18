"""
charts.py
---------
Dashboard ke charts inline SVG me banata hai.

Kyun SVG? Isse koi JavaScript library ya internet (CDN) nahi chahiye, chart
page ke saath hi render ho jaata hai aur colors CSS variables se aate hain.
Saare labels html.escape se guzarte hain, isliye database ka text chart me
kuch bigaad nahi sakta.
"""
import html
import math

from markupsafe import Markup

NO_DATA = '<p class="muted">Not enough data to draw this chart yet.</p>'


def _esc(value):
    return html.escape(str(value), quote=True)


def _fmt(value, decimals=0):
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------- line chart
def line_chart(values, labels=None, width=640, height=240, color="var(--blue)",
               y_min=None, y_max=None, unit="", decimals=0, thresholds=None,
               area=True, aria="Trend chart"):
    """
    Simple line chart (axes + gridlines ke saath).
    thresholds = [(value, "label", "color"), ...] dashed horizontal lines.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n < 2:
        return Markup(NO_DATA)

    ml, mr, mt, mb = 50, 14, 12, 30
    pw, ph = width - ml - mr, height - mt - mb

    lo = min(vals) if y_min is None else float(y_min)
    hi = max(vals) if y_max is None else float(y_max)
    if hi - lo < 1e-9:
        hi += 1
        if y_min is None:
            lo -= 1
    pad = (hi - lo) * 0.08
    if y_min is None:
        lo -= pad
    if y_max is None:
        hi += pad

    def sx(i):
        return ml + pw * i / (n - 1)

    def sy(v):
        return mt + ph * (1 - (v - lo) / (hi - lo))

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(aria)}">']

    # gridlines + y labels
    for k in range(5):
        val = lo + (hi - lo) * k / 4
        y = sy(val)
        parts.append(f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{width - mr}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end">{_fmt(val, decimals)}{_esc(unit)}</text>')

    # x labels (max 5)
    if labels:
        for i in sorted({round(k * (n - 1) / 4) for k in range(5)}):
            # aakhri label right-aligned: warna chart ke kinare par kat jaata hai
            anchor = "end" if i == n - 1 else "middle"
            parts.append(f'<text x="{sx(i):.1f}" y="{height - 9}" text-anchor="{anchor}">{_esc(labels[i])}</text>')

    # threshold lines
    for t_val, t_label, t_color in (thresholds or []):
        if lo <= t_val <= hi:
            y = sy(t_val)
            parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width - mr}" y2="{y:.1f}" '
                         f'style="stroke:{t_color};stroke-dasharray:5 4;stroke-width:1.2"/>')
            parts.append(f'<text x="{width - mr - 2}" y="{y - 4:.1f}" text-anchor="end" '
                         f'style="fill:{t_color}">{_esc(t_label)}</text>')

    points = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
    if area:
        base = sy(lo)
        parts.append(f'<polygon points="{ml},{base:.1f} {points} {sx(n - 1):.1f},{base:.1f}" '
                     f'style="fill:{color};opacity:.10"/>')
    parts.append(f'<polyline points="{points}" style="fill:none;stroke:{color};stroke-width:2;'
                 f'stroke-linejoin:round"/>')

    # dots + hover text
    if n <= 40:
        for i, v in enumerate(vals):
            tip = f"{labels[i]}: " if labels else ""
            parts.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="3" style="fill:{color}">'
                         f'<title>{_esc(tip)}{_fmt(v, max(decimals, 1))}{_esc(unit)}</title></circle>')

    parts.append("</svg>")
    return Markup("".join(parts))


# ---------------------------------------------------------------- sparkline
def sparkline(values, width=200, height=48, color="var(--blue)"):
    """Chhota trend line (axes ke bina), machine page ke sensor cards ke liye."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n < 2:
        return Markup(NO_DATA)
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    pad = 5

    def sx(i):
        return pad + (width - 2 * pad) * i / (n - 1)

    def sy(v):
        return pad + (height - 2 * pad) * (1 - (v - lo) / (hi - lo))

    points = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
    svg = (f'<svg class="spark" viewBox="0 0 {width} {height}" role="img" aria-label="Sensor trend">'
           f'<polyline points="{points}" style="fill:none;stroke:{color};stroke-width:2;stroke-linejoin:round"/>'
           f'<circle cx="{sx(n - 1):.1f}" cy="{sy(vals[-1]):.1f}" r="3.5" style="fill:{color}"/></svg>')
    return Markup(svg)


# ---------------------------------------------------------------- donut
def donut(items, size=190, thickness=30, center_label="total"):
    """
    items = [(label, value, color), ...]
    Donut ke saath legend bhi (counts + percent) return karta hai.
    """
    total = sum(v for _, v, _ in items)
    if total <= 0:
        return Markup(NO_DATA)

    r = (size - thickness) / 2
    c = 2 * math.pi * r
    mid = size / 2
    parts = [f'<svg class="donut-svg" viewBox="0 0 {size} {size}" role="img" aria-label="Distribution chart">',
             f'<circle cx="{mid}" cy="{mid}" r="{r}" style="fill:none;stroke:var(--border);stroke-width:{thickness}"/>']
    offset = 0.0
    for label, value, color in items:
        if value <= 0:
            continue
        dash = value / total * c
        parts.append(
            f'<circle cx="{mid}" cy="{mid}" r="{r}" transform="rotate(-90 {mid} {mid})" '
            f'style="fill:none;stroke:{color};stroke-width:{thickness};'
            f'stroke-dasharray:{dash:.2f} {c - dash:.2f};stroke-dashoffset:{-offset:.2f}">'
            f'<title>{_esc(label)}: {value}</title></circle>')
        offset += dash
    parts.append(f'<text class="donut-num" x="{mid}" y="{mid + 4}" text-anchor="middle">{total:g}</text>')
    parts.append(f'<text x="{mid}" y="{mid + 22}" text-anchor="middle">{_esc(center_label)}</text>')
    parts.append("</svg>")

    legend = ['<ul class="legend">']
    for label, value, color in items:
        pct = value / total * 100
        legend.append(f'<li><span class="sw" style="background:{color}"></span>'
                      f'{_esc(label)} <b>{value:g}</b> <span class="muted">({pct:.0f}%)</span></li>')
    legend.append("</ul>")
    return Markup(f'<div class="donut">{"".join(parts)}{"".join(legend)}</div>')


# ---------------------------------------------------------------- bars
def hbar(items, width=560, bar_h=22, gap=10, color="var(--blue)"):
    """Horizontal bars. items = [(label, value) ya (label, value, color)]"""
    items = [(i[0], float(i[1]), i[2] if len(i) > 2 else color) for i in items]
    if not items or max(v for _, v, _ in items) <= 0:
        return Markup(NO_DATA)

    left = min(210, max(len(label) for label, _, _ in items) * 7 + 14)
    right = 46
    max_v = max(v for _, v, _ in items)
    height = len(items) * (bar_h + gap)
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Bar chart">']
    for i, (label, value, col) in enumerate(items):
        y = i * (bar_h + gap)
        w = (width - left - right) * value / max_v
        parts.append(f'<text x="{left - 8}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end">{_esc(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{max(w, 2):.1f}" height="{bar_h}" rx="3" style="fill:{col}">'
                     f'<title>{_esc(label)}: {value:g}</title></rect>')
        parts.append(f'<text class="val" x="{left + max(w, 2) + 6:.1f}" y="{y + bar_h / 2 + 4:.1f}">{value:g}</text>')
    parts.append("</svg>")
    return Markup("".join(parts))


def vbar(items, width=460, height=240, color="var(--blue)"):
    """Vertical bars. items = [(label, value) ya (label, value, color)]"""
    items = [(i[0], float(i[1]), i[2] if len(i) > 2 else color) for i in items]
    if not items or max(v for _, v, _ in items) <= 0:
        return Markup(NO_DATA)

    ml, mr, mt, mb = 10, 10, 22, 30
    pw, ph = width - ml - mr, height - mt - mb
    max_v = max(v for _, v, _ in items)
    slot = pw / len(items)
    bw = min(70, slot * 0.6)
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Bar chart">',
             f'<line class="grid" x1="{ml}" y1="{mt + ph}" x2="{width - mr}" y2="{mt + ph}"/>']
    for i, (label, value, col) in enumerate(items):
        h = ph * value / max_v
        x = ml + slot * i + (slot - bw) / 2
        y = mt + ph - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h, 1):.1f}" rx="3" style="fill:{col}">'
                     f'<title>{_esc(label)}: {value:g}</title></rect>')
        parts.append(f'<text class="val" x="{x + bw / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle">{value:g}</text>')
        parts.append(f'<text x="{x + bw / 2:.1f}" y="{height - 10}" text-anchor="middle">{_esc(label)}</text>')
    parts.append("</svg>")
    return Markup("".join(parts))
