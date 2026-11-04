"""Ve lai ba so do Chuong 3 (Hinh 3.1-3.3): ve CO CHE, khong ve ten goi.

Ba hinh cu la anh tinh tu ban v11 va sai so voi he thong da dung lai ("GCN 2 lop",
"BPR co trong so", node "metadata" khong ton tai, nhan tieng Anh). Ban ve lai dau
tien bang matplotlib dung noi dung nhung chi la "o chu noi mui ten" -- doc chi thay
ten cac buoc. Ban nay ve thu that su xay ra:

* Hinh 3.1 -- hai lop cua he thong (truy vet / chieu) tach ra tu CUNG mot tap
  huan luyen, voi guard kiem tong trong so (docs/KG_DESIGN.md muc 1).
* Hinh 3.2 -- mot do thi mau nho: do day canh ti le voi trong so W(u,i) tinh
  bang cong thuc (3.6)-(3.8) tai T_train; mot canh gop hai su kien.
* Hinh 3.3 -- truc thoi gian ve DUNG TI LE 96,6 / 13,8 / 27,6 ngay voi duong
  suy giam e^(-lambda*dt) neo tai T_train, va moi doan dung de lam gi.

SVG viet tay, chuyen sang PNG bang Chrome headless (--force-device-scale-factor 3).
Font Liberation Serif: cung so do voi Times New Roman cua than de an.

Chay:
    python scripts/18_ve_hinh_chuong3.py      # ghi docs/figures/fig_3_*.png
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
SPLIT = ROOT / "data" / "interim" / "original" / "split.json"

INK = "#1f2328"
MUTED = "#5b6068"
LINE = "#8a9099"
FILL = "#f4f5f7"
ACCENT = "#b3261e"          # mot mau duy nhat: co che hanh vi-thoi gian
ACCENT_BG = "#fbeceb"
FONT = "Liberation Serif, 'Times New Roman', serif"
LAMBDA = 0.05
ALPHA = {"view": 1.0, "addtocart": 2.0, "transaction": 3.0}


# ── SVG nho gon ─────────────────────────────────────────────────────────


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def t(x, y, s, size=16, anchor="middle", weight="normal", fill=INK, style="normal", halo=False):
    """Mot dong chu; `s` co the chua tspan da dung san (khong escape lai).

    halo=True ve vien trang duoi chu, de nhan dat tren duong ke van doc duoc.
    """
    h = ' paint-order="stroke" stroke="#fff" stroke-width="5" stroke-linejoin="round"' if halo else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-style="{style}" fill="{fill}"{h}>{s}</text>')


def lines(x, y, rows, size=16, gap=1.25, **kw):
    return "".join(t(x, y + i * size * gap, r, size=size, **kw) for i, r in enumerate(rows))


def sup(s):
    return f'<tspan baseline-shift="super" font-size="70%">{s}</tspan>'


def sub(s):
    return f'<tspan baseline-shift="sub" font-size="70%">{s}</tspan>'


def rect(x, y, w, h, fill=FILL, stroke=INK, sw=1.2, r=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arrow(x1, y1, x2, y2, color=INK, sw=1.3, dash=None, head="a"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d} marker-end="url(#{head})"/>')


def path(d, color=INK, sw=1.3, dash=None, end=True, fill="none"):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    me = ' marker-end="url(#a)"' if end else ""
    return f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}"{da}{me}/>'


def svg(w, h, body, label):
    defs = (
        '<defs>'
        f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>'
        f'<marker id="r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{ACCENT}"/></marker>'
        f'<marker id="g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>'
        '</defs>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}" '
            f'font-family="{FONT}"><rect width="{w}" height="{h}" fill="#fff"/>{defs}{body}</svg>')


# ── Bieu tuong nho, ve dung vat the ─────────────────────────────────────


def doc_icon(x, y, w, h, name, note):
    fold = 12
    shape = (f'<path d="M{x},{y} h{w - fold} l{fold},{fold} v{h - fold} h{-w} z" '
             f'fill="#fff" stroke="{INK}" stroke-width="1.2"/>'
             f'<path d="M{x + w - fold},{y} v{fold} h{fold}" fill="none" stroke="{INK}" stroke-width="1.2"/>')
    return (shape + t(x + w / 2, y + h / 2 - 2, name, size=15, weight="bold")
            + t(x + w / 2, y + h / 2 + 17, note, size=13.5, fill=MUTED))


def cylinder(x, y, w, h):
    ry = 7
    return (f'<path d="M{x},{y + ry} v{h - 2 * ry} a{w / 2},{ry} 0 0 0 {w},0 v{-(h - 2 * ry)}" '
            f'fill="{FILL}" stroke="{INK}" stroke-width="1.2"/>'
            f'<ellipse cx="{x + w / 2}" cy="{y + ry}" rx="{w / 2}" ry="{ry}" fill="#fff" '
            f'stroke="{INK}" stroke-width="1.2"/>')


def matrix_icon(x, y, n=6, c=9):
    """Ma tran thua: o to dam = canh co trong so lon, o nhat = trong so nho."""
    cells = [(0, 2, 1.0), (0, 4, .25), (1, 3, .6), (2, 0, 1.0), (2, 5, .15), (3, 1, .6),
             (4, 0, .25), (5, 2, .15), (1, 5, .4), (5, 1, .4), (3, 4, .8), (4, 3, .8)]
    out = [f'<rect x="{x}" y="{y}" width="{n * c}" height="{n * c}" fill="#fff" stroke="{INK}" stroke-width="1"/>']
    for r, k, a in cells:
        out.append(f'<rect x="{x + k * c + 1}" y="{y + r * c + 1}" width="{c - 2}" height="{c - 2}" '
                   f'fill="{ACCENT}" fill-opacity="{a}"/>')
    return "".join(out)


def layers_icon(x, y, w=74, h=10, n=4, gap=9):
    out = []
    for i in range(n):
        yy = y + i * (h + gap)
        out.append(rect(x, yy, w, h, fill="#fff" if i else FILL, r=2))
        if i:
            out.append(arrow(x + w / 2, yy - gap + 1, x + w / 2, yy - 1, sw=1))
    return "".join(out)


# ── Hinh 3.1 ────────────────────────────────────────────────────────────


def hinh_3_1() -> str:
    W, H = 900, 440
    b = []
    # Cot nguon
    b.append(t(95, 34, "DỮ LIỆU GỐC", size=13, weight="bold", fill=MUTED))
    for i, (name, note) in enumerate([("events.csv", "2.756.101 sự kiện"),
                                      ("item_properties", "thuộc tính theo thời gian"),
                                      ("category_tree", "cây danh mục")]):
        b.append(doc_icon(20, 52 + i * 78, 150, 58, name, note))
    # Tien xu ly
    bx, by, bw, bh = 222, 128, 150, 132
    for i in range(3):
        b.append(path(f"M170,{81 + i * 78} H196 V{by + bh / 2} H{bx - 2}", end=(i == 1)))
    b.append(rect(bx, by, bw, bh, fill="#fff"))
    b.append(lines(bx + bw / 2, by + 28, ["Tiền xử lý"], size=16, weight="bold"))
    b.append(lines(bx + bw / 2, by + 56, ["chỉ giữ dữ liệu", "t ≤ T" + sub("train")], size=14, gap=1.4, fill=MUTED))
    b.append(lines(bx + bw / 2, by + 100, ["ánh xạ định danh", "chỉ từ tập huấn luyện"], size=14, fill=MUTED))
    # Hai lan
    lx = 420
    b.append(rect(lx, 18, 462, 150, fill="none", stroke=LINE, dash="5 4", r=10))
    b.append(t(lx + 12, 38, "LỚP TRUY VẾT", size=13, anchor="start", weight="bold", fill=MUTED))
    b.append(rect(lx, 214, 462, 168, fill="none", stroke=LINE, dash="5 4", r=10))
    b.append(t(lx + 12, 234, "LỚP CHIẾU — DÙNG ĐỂ HUẤN LUYỆN", size=13, anchor="start", weight="bold", fill=MUTED))
    b.append(path(f"M{bx + bw},{by + 30} H396 V96 H{lx + 26}"))
    b.append(path(f"M{bx + bw},{by + bh - 30} H396 V318 H{lx + 26}"))
    # Truy vet: Neo4j -> ung dung
    b.append(cylinder(lx + 20, 56, 140, 82))
    b.append(lines(lx + 90, 94, ["Neo4j"], size=16, weight="bold"))
    b.append(lines(lx + 90, 113, ["mỗi sự kiện", "là một node"], size=13, fill=MUTED))
    b.append(rect(lx + 300, 62, 142, 70, fill="#fff"))
    b.append(lines(lx + 371, 91, ["Ứng dụng", "truy vết"], size=16, weight="bold"))
    b.append(arrow(lx + 160, 97, lx + 298, 97))
    b.append(t(lx + 224, 88, "truy vấn Cypher", size=13.5, fill=MUTED))
    # Chieu: A -> A_hat/GCN -> Top-K
    b.append(matrix_icon(lx + 40, 262))
    b.append(lines(lx + 67, 340, ["Ma trận kề A"], size=15, weight="bold"))
    b.append(lines(lx + 67, 362, ["W(u,i) = Σ α" + sub("b") + "·e" + sup("−λΔt")], size=14, fill=ACCENT))
    b.append(arrow(lx + 100, 289, lx + 186, 289))
    b.append(t(lx + 143, 280, "D" + sup("−½") + "AD" + sup("−½"), size=13.5, fill=MUTED))
    b.append(layers_icon(lx + 192, 254))
    b.append(lines(lx + 229, 340, ["GCN 3 lớp"], size=15, weight="bold"))
    b.append(lines(lx + 229, 362, ["học E" + sub("0") + " bằng BPR"], size=14, fill=MUTED))
    b.append(arrow(lx + 270, 289, lx + 338, 289))
    b.append(t(lx + 304, 280, "z" + sub("u") + "·z" + sub("i"), size=13.5, fill=MUTED))
    b.append(rect(lx + 342, 258, 100, 64, fill="#fff"))
    for k in range(4):
        b.append(f'<line x1="{lx + 356}" y1="{272 + k * 12}" x2="{lx + 428 - k * 10}" y2="{272 + k * 12}" '
                 f'stroke="{INK}" stroke-width="1.6"/>')
    b.append(lines(lx + 392, 340, ["Top-K"], size=15, weight="bold"))
    b.append(lines(lx + 392, 362, ["topk.csv"], size=14, fill=MUTED))
    # Top-K -> ung dung
    b.append(arrow(lx + 392, 256, lx + 392, 134))
    b.append(t(lx + 400, 196, "phát lại", size=13.5, anchor="start", fill=MUTED))
    # Guard nhat quan giua hai lop
    b.append(f'<line x1="{lx + 90}" y1="140" x2="{lx + 90}" y2="254" stroke="{LINE}" stroke-width="1.2" '
             f'stroke-dasharray="3 3" marker-start="url(#g)" marker-end="url(#g)"/>')
    b.append(lines(lx + 98, 186, ["tổng trọng số", "hai lớp khớp nhau"], size=13, anchor="start", fill=MUTED, style="italic"))
    b.append(lines(W / 2, 418, ["Màu đỏ: phần đóng góp của đề án (trọng số cạnh). Kiến trúc lan truyền giữ nguyên LightGCN."],
                   size=13.5, fill=MUTED, style="italic"))
    return svg(W, H, "".join(b), "Kiến trúc hai lớp: lớp truy vết Neo4j và lớp chiếu dùng để huấn luyện, "
                                 "cùng sinh từ tập huấn luyện đã cắt tại T_train.")


# ── Hinh 3.2 ────────────────────────────────────────────────────────────


def w_cua(su_kien):
    """Cong thuc (3.6)-(3.8) tai T_train: W = tong alpha_b * e^(-lambda * tuoi)."""
    return sum(ALPHA[b] * math.exp(-LAMBDA * d) for b, d in su_kien)


def so(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def nhan_theo_canh(x1, y1, x2, y2, text, at=0.5, off=-11, size=13.5, fill=ACCENT):
    """Nhan chay song song voi canh, lech vuong goc `off` don vi -- khong de len net."""
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    nx, ny = -(y2 - y1), (x2 - x1)
    n = math.hypot(nx, ny)
    x = x1 + (x2 - x1) * at + nx / n * off
    y = y1 + (y2 - y1) * at + ny / n * off
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="middle" fill="{fill}" '
            f'transform="rotate({ang:.1f} {x:.1f} {y:.1f})">{text}</text>')


def hinh_3_2() -> str:
    W, H = 900, 520
    b = []
    u = {"u1": (90, 150), "u2": (90, 330)}
    i = {"i1": (390, 90), "i2": (390, 240), "i3": (390, 390)}
    canh = [  # (u, i, su kien [(hanh vi, tuoi ngay)], nhan)
        ("u1", "i1", [("transaction", 1)], "mua, 1 ngày trước"),
        ("u1", "i2", [("view", 60)], "xem, 60 ngày trước"),
        ("u2", "i2", [("addtocart", 3)], "thêm giỏ, 3 ngày trước"),
        ("u2", "i3", [("view", 1), ("view", 10)], "xem 2 lần: 1 và 10 ngày"),
    ]
    for uu, ii, ev, nhan in canh:
        (x1, y1), (x2, y2) = u[uu], i[ii]
        w = w_cua(ev)
        sw = max(0.8, 1.6 * w)
        b.append(f'<line x1="{x1 + 30}" y1="{y1}" x2="{x2 - 34}" y2="{y2}" stroke="{ACCENT}" '
                 f'stroke-width="{sw:.2f}" stroke-linecap="round"/>')
        # Canh di trai -> phai: phap tuyen duong luon huong XUONG tren man hinh (truc y
        # cua SVG di xuong), nen off < 0 = phia tren, off > 0 = phia duoi.
        # Canh di len: nhan phia tren; canh di xuong: nhan phia duoi. Hai canh cung do
        # vao i2 tao thanh mot nem hep dan ve phia i2 -> dat nhan lui ve phia khach hang.
        len_ = y2 < y1
        at = 0.4 if (uu, ii) == ("u1", "i2") else 0.52
        off = -(sw / 2 + 7) if len_ else (sw / 2 + 16)
        b.append(nhan_theo_canh(x1 + 30, y1, x2 - 34, y2, f"{nhan} → W = {so(w)}", at=at, off=off))
    for k, (x, y) in u.items():
        b.append(f'<circle cx="{x}" cy="{y}" r="30" fill="#fff" stroke="{INK}" stroke-width="1.4"/>')
        b.append(t(x, y + 5, k.replace("u", "u" + sub("")[0:0]), size=16, weight="bold"))
    for k, (x, y) in i.items():
        b.append(rect(x - 34, y - 24, 68, 48, fill=FILL, r=4))
        b.append(t(x, y + 5, k, size=16, weight="bold"))
    # Tri thuc san pham
    c = {"c1": (620, 120), "c2": (620, 260), "c0": (810, 190)}
    pv = {"p1": (640, 405)}

    def diamond(x, y, r=26):
        return (f'<path d="M{x},{y - r} L{x + r},{y} L{x},{y + r} L{x - r},{y} z" fill="#fff" '
                f'stroke="{INK}" stroke-width="1.4"/>')

    for (ii, cc) in [("i1", "c1"), ("i2", "c1"), ("i3", "c2")]:
        (x1, y1), (x2, y2) = i[ii], c[cc]
        b.append(arrow(x1 + 34, y1, x2 - 28, y2, color=INK, sw=1.1))
        if ii == "i1":
            b.append(nhan_theo_canh(x1 + 34, y1, x2 - 28, y2, "HAS_CATEGORY", off=-8,
                                    size=12.5, fill=MUTED))
    for cc in ("c1", "c2"):
        (x1, y1), (x2, y2) = c[cc], c["c0"]
        b.append(arrow(x1 + 26, y1, x2 - 28, y2, sw=1.1))
        if cc == "c1":
            b.append(nhan_theo_canh(x1 + 26, y1, x2 - 28, y2, "PARENT_CATEGORY", off=-8,
                                    size=12.5, fill=MUTED))
    (x1, y1), (x2, y2) = i["i3"], pv["p1"]
    b.append(arrow(x1 + 34, y1 + 6, x2 - 52, y2, sw=1.1))
    b.append(nhan_theo_canh(x1 + 34, y1 + 6, x2 - 52, y2, "HAS_PROPERTY", off=16,
                            size=12.5, fill=MUTED))
    for k, (x, y) in c.items():
        b.append(diamond(x, y))
        b.append(t(x, y + 5, k, size=15, weight="bold"))
    for k, (x, y) in pv.items():
        b.append(rect(x - 50, y - 18, 100, 36, fill="#fff", r=18))
        b.append(t(x, y + 5, "màu = đỏ", size=14))
    b.append(t(236, 40, "INTERACTED_WITH — độ dày tỉ lệ với W(u,i)", size=13.5, weight="bold", fill=ACCENT))
    # Chu giai hinh dang
    ly = 480
    b.append(f'<circle cx="40" cy="{ly - 5}" r="9" fill="#fff" stroke="{INK}"/>' + t(56, ly, "Khách hàng", 14, "start"))
    b.append(rect(150, ly - 14, 18, 18, r=2) + t(176, ly, "Sản phẩm", 14, "start"))
    b.append(diamond(268, ly - 5, 10) + t(284, ly, "Danh mục", 14, "start"))
    b.append(rect(372, ly - 13, 30, 16, fill="#fff", r=8) + t(410, ly, "Thuộc tính = giá trị", 14, "start"))
    b.append(lines(560, ly, ["W tính tại T" + sub("train") + " với α = 1/2/3, λ = 0,05"], 14, anchor="start", fill=MUTED))
    return svg(W, H, "".join(b), "Đồ thị mẫu: cạnh khách–sản phẩm có độ dày tỉ lệ với trọng số hành vi-thời gian; "
                                 "sản phẩm nối với danh mục, danh mục cha và thuộc tính.")


# ── Hinh 3.3 ────────────────────────────────────────────────────────────


def ngay(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%d/%m/%Y")


def hinh_3_3() -> str:
    s = json.loads(SPLIT.read_text())
    d = 86_400_000
    tr = (s["t_train"] - s["t_min"]) / d
    va = (s["t_valid_end"] - s["t_train"]) / d
    te = (s["t_max"] - s["t_valid_end"]) / d
    W, H = 900, 470
    x0, x1 = 40, 860
    k = (x1 - x0) / (tr + va + te)
    xt, xv = x0 + tr * k, x0 + (tr + va) * k
    yb = 250
    b = []
    # Duong suy giam neo tai T_train
    top, base = 70, 210
    pts = []
    for n in range(0, 101):
        age = tr * n / 100
        pts.append(f"{xt - age * k:.1f},{base - (base - top) * math.exp(-LAMBDA * age):.1f}")
    b.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{ACCENT}" stroke-width="2"/>')
    b.append(f'<line x1="{x0}" y1="{base}" x2="{xt}" y2="{base}" stroke="{LINE}" stroke-width="0.8"/>')
    half = math.log(2) / LAMBDA
    xh = xt - half * k
    yh = base - (base - top) * 0.5
    b.append(f'<line x1="{xh:.1f}" y1="{yh:.1f}" x2="{xh:.1f}" y2="{base}" stroke="{ACCENT}" stroke-dasharray="3 3"/>')
    b.append(t(xh - 6, yh - 8, f"còn ½ sau ≈ {so(half, 1)} ngày", 13.5, "end", fill=ACCENT))
    b.append(t(x0 + 4, base - 8, f"đầu kỳ: ≈ {so(math.exp(-LAMBDA * tr), 3)}", 13.5, "start", fill=ACCENT))
    b.append(t(xt - 8, top - 10, "e" + sup("−λΔt") + " = 1 tại T" + sub("train"), 13.5, "end", fill=ACCENT))
    b.append(t(x0, 40, "Trọng số một sự kiện theo tuổi Δt, λ = 0,05", 15, "start", weight="bold"))
    # Truc thoi gian ti le
    for xa, xb, fill, ten, dai in [(x0, xt, FILL, "Huấn luyện", tr), (xt, xv, "#fff", "Xác thực", va),
                                   (xv, x1, "#fff", "Kiểm tra", te)]:
        b.append(f'<rect x="{xa}" y="{yb}" width="{xb - xa}" height="46" fill="{fill}" stroke="{INK}" stroke-width="1.2"/>')
        b.append(t((xa + xb) / 2, yb + 20, ten, 14.5, weight="bold"))
        b.append(t((xa + xb) / 2, yb + 38, f"{so(dai, 1)} ngày", 13, fill=MUTED))
    for x, ms, nhan in [(x0, s["t_min"], ""), (xt, s["t_train"], "T" + sub("train")),
                        (xv, s["t_valid_end"], ""), (x1, s["t_max"], "")]:
        b.append(f'<line x1="{x}" y1="{yb - 8}" x2="{x}" y2="{yb + 52}" stroke="{INK}" stroke-width="1.2"/>')
        b.append(t(x, yb + 70, ngay(ms), 13, fill=MUTED))
        if nhan:
            b.append(t(x, yb - 16, nhan, 15, weight="bold", halo=True))
    # Moi doan dung de lam gi
    yu = 360
    b.append(lines((x0 + xt) / 2, yu, ["Dựng đồ thị và huấn luyện BPR, 3 seed (2020, 2021, 2022);",
                                        "mẫu âm chỉ lấy từ sản phẩm xuất hiện trong đoạn này."], 14))
    b.append(lines(x1, yu, ["Xác thực: NDCG@20 mỗi 5 epoch,", "dừng sau 20 lần không cải thiện",
                            "(Active: 50), giữ epoch tốt nhất."], 13.5, anchor="end"))
    b.append(lines(x1, yu + 66, ["Kiểm tra: đánh giá một lần,", "Recall, NDCG, HitRate, Coverage."],
                   13.5, anchor="end", fill=MUTED))
    return svg(W, H, "".join(b), "Trục thời gian chia 70/10/20 vẽ đúng tỉ lệ; trọng số sự kiện suy giảm "
                                 "theo tuổi tính từ T_train; mỗi đoạn dùng cho một việc.")


# ── Xuat ────────────────────────────────────────────────────────────────


def render(svg_text: str, path: Path) -> Path:
    import re
    w, h = (int(v) for v in re.search(r'width="(\d+)" height="(\d+)"', svg_text).groups())
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "f.html"
        html.write_text(f'<!doctype html><meta charset="utf-8"><body style="margin:0;background:#fff">'
                        f"{svg_text}</body>", encoding="utf-8")
        subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu",
                        "--hide-scrollbars", "--force-device-scale-factor=3",
                        f"--window-size={w},{h}", f"--screenshot={path}", html.as_uri()],
                       check=True, capture_output=True, timeout=120)
    path.with_suffix(".svg").write_text(svg_text, encoding="utf-8")
    return path


# Hinh 3.2 va 3.3 nay lay tu scripts/19_drawio_chuong3.py (file .drawio de chinh tay);
# hinh_3_2/hinh_3_3 o day giu lai lam lich su, KHONG xuat de khoi ghi de.
HINH = {"Hình 3.1.": (hinh_3_1, "fig_3_1_kien_truc.png")}


def main() -> dict[str, Path]:
    out = {cap: render(ve(), OUT / ten) for cap, (ve, ten) in HINH.items()}
    for cap, path in out.items():
        print(f"{cap} -> {path}")
    return out


if __name__ == "__main__":
    main()
