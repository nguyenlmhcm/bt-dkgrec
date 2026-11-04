"""Hinh 3.2 va 3.3 dang draw.io: file .drawio de chinh tay + PNG xuat tu chinh file do.

Nguoi dung can sua hinh trong draw.io, nen nguon cua hai hinh nay la XML draw.io
(docs/figures/*.drawio). PNG duoc render tu CHINH file do bang draw.io viewer
trong Chrome headless -- mo file ra sua thi thay dung hinh trong quyen.

Moi duong noi la orthogonalEdgeStyle (chi doan ngang/doc), khong co duong cheo.
Trong so canh khach-san pham tinh bang cong thuc (3.6)-(3.8) tai T_train.

Chay:
    python scripts/19_drawio_chuong3.py   # ghi fig_3_2_schema.{drawio,png}, fig_3_3_quy_trinh.{drawio,png}
"""

from __future__ import annotations

import html
import json
import math
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import quoteattr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
SPLIT = ROOT / "data" / "interim" / "original" / "split.json"
VIEWER = "https://viewer.diagrams.net/js/viewer-static.min.js"

INK = "#1F2328"
MUTED = "#5B6068"
FILL = "#F4F5F7"
ACCENT = "#B3261E"
FONT = "fontFamily=Times New Roman;fontColor=#1F2328;"
LAMBDA = 0.05
ALPHA = {"view": 1.0, "addtocart": 2.0, "transaction": 3.0}


class Do:
    """Bo dung XML draw.io toi gian: vertex, edge, text."""

    def __init__(self):
        self.cells: list[str] = []
        self.n = 0

    def _id(self, id_=None, prefix="n"):
        self.n += 1
        return id_ or f"{prefix}{self.n}"

    def node(self, x, y, w, h, value, style, id_=None):
        i = self._id(id_)
        self.cells.append(
            f'<mxCell id="{i}" value={quoteattr(value)} style={quoteattr(FONT + style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def text(self, x, y, w, h, value, size=13, align="center", color=INK, extra=""):
        return self.node(x, y, w, h, value,
                         f"text;html=1;strokeColor=none;fillColor=none;align={align};"
                         f"verticalAlign=middle;whiteSpace=wrap;fontSize={size};fontColor={color};{extra}")

    def edge(self, src, dst, value="", style="", points=(), lx=0.0, ly=0.0, lpos=0.0):
        """Duong vuong goc. lpos: vi tri nhan doc theo canh (-1..1); lx, ly: lech them."""
        i = self._id(prefix="e")
        pts = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points)
        arr = f'<Array as="points">{pts}</Array>' if points else ""
        base = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;endSize=7;"
                "labelBackgroundColor=#FFFFFF;fontSize=12.5;")
        self.cells.append(
            f'<mxCell id="{i}" value={quoteattr(value)} style={quoteattr(FONT + base + style)} '
            f'edge="1" parent="1" source="{src}" target="{dst}">'
            f'<mxGeometry x="{lpos}" y="{ly}" relative="1" as="geometry">'
            f'<mxPoint x="{lx}" y="0" as="offset"/>{arr}</mxGeometry></mxCell>')
        return i

    def xml(self, name: str) -> str:
        body = "".join(self.cells)
        return (f'<mxfile host="bt-dkgrec"><diagram name="{name}" id="{name.replace(" ", "_")}">'
                f'<mxGraphModel grid="1" gridSize="10" guides="1" page="0" math="0" shadow="0">'
                f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root>'
                f'</mxGraphModel></diagram></mxfile>')


def so(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def w_cua(su_kien):
    return sum(ALPHA[b] * math.exp(-LAMBDA * d) for b, d in su_kien)


# ── Hinh 3.2 ────────────────────────────────────────────────────────────


def hinh_3_2() -> str:
    g = Do()
    BOX = "rounded=1;whiteSpace=wrap;html=1;arcSize=12;strokeColor=#1F2328;strokeWidth=1.2;"
    CIRCLE = ("ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#FFFFFF;strokeColor=#1F2328;"
              "strokeWidth=1.3;fontSize=16;fontStyle=1;")
    ITEM = BOX + f"fillColor={FILL};fontSize=16;fontStyle=1;"
    DIAM = "rhombus;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#1F2328;strokeWidth=1.3;fontSize=15;fontStyle=1;"
    PILL = BOX + "arcSize=50;fillColor=#FFFFFF;fontSize=14;"

    for x, w, t in [(30, 90, "Khách hàng"), (380, 90, "Sản phẩm"), (620, 100, "Danh mục"),
                    (790, 100, "Danh mục cha")]:
        g.text(x, 20, w, 24, t, size=13.5, color=MUTED, extra="fontStyle=1;")

    u1 = g.node(40, 120, 70, 70, "u1", CIRCLE, "u1")
    u2 = g.node(40, 280, 70, 70, "u2", CIRCLE, "u2")
    i1 = g.node(385, 63, 80, 64, "i1", ITEM, "i1")
    i2 = g.node(385, 203, 80, 64, "i2", ITEM, "i2")
    i3 = g.node(385, 343, 80, 64, "i3", ITEM, "i3")
    c1 = g.node(630, 125, 80, 70, "c1", DIAM, "c1")
    c2 = g.node(630, 340, 80, 70, "c2", DIAM, "c2")   # tam y = 375 = tam i3 -> canh thang
    c0 = g.node(800, 230, 80, 70, "c0", DIAM, "c0")
    p1 = g.node(610, 460, 120, 40, "màu = đỏ", PILL, "p1")

    # Canh hanh vi-thoi gian: do day ti le voi W. Hai canh cung vao i2 vao o hai
    # diem khac nhau (entryY 0.3 / 0.7) de khong chong len nhau.
    # Tren canh chi ghi so thu tu; noi dung nam trong bang chu giai ben duoi --
    # nhoi chu len duong gap khuc lam cac nhan de nhau o cho hai canh cung vao i2.
    canh = [
        (u1, i1, [("transaction", 1)], "mua, 1 ngày trước", "exitX=1;exitY=0.35;entryX=0;entryY=0.5;"),
        (u1, i2, [("view", 60)], "xem, 60 ngày trước", "exitX=1;exitY=0.65;entryX=0;entryY=0.2;"),
        (u2, i2, [("addtocart", 3)], "thêm giỏ, 3 ngày trước", "exitX=1;exitY=0.35;entryX=0;entryY=0.8;"),
        (u2, i3, [("view", 1), ("view", 10)], "xem 2 lần, 1 và 10 ngày trước", "exitX=1;exitY=0.65;entryX=0;entryY=0.5;"),
    ]
    SO = "1234"   # font Times khong co ky tu ①②③④ -> so thuong trong o vien do
    bang = []
    for k, (src, dst, ev, nhan, ports) in enumerate(canh):
        w = w_cua(ev)
        g.edge(src, dst, SO[k],
               f"{ports}strokeColor={ACCENT};fontColor={ACCENT};strokeWidth={1 + 1.4 * w:.1f};endArrow=none;"
               f"fontSize=13;fontStyle=1;labelBorderColor={ACCENT};spacingLeft=4;spacingRight=4;", lpos=0.75)
        bang.append(f"<b>({SO[k]})</b> {nhan}: W = <b>{so(w)}</b>")

    g.edge(i1, c1, "", "exitX=1;exitY=0.5;entryX=0;entryY=0.35;strokeWidth=1.1;")
    g.edge(i2, c1, "", "exitX=1;exitY=0.5;entryX=0;entryY=0.65;strokeWidth=1.1;")
    g.edge(i3, c2, "", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;strokeWidth=1.1;")
    g.edge(c1, c0, "", "exitX=1;exitY=0.5;entryX=0.5;entryY=0;strokeWidth=1.1;")
    g.edge(c2, c0, "", "exitX=1;exitY=0.5;entryX=0.5;entryY=1;strokeWidth=1.1;")
    g.edge(i3, p1, "", "exitX=1;exitY=0.8;entryX=0;entryY=0.5;strokeWidth=1.1;",
           points=[(545, 390), (545, 480)])
    # Nhan quan he: o chu dat co dinh canh duong cua no (keo chinh duoc trong draw.io)
    g.text(475, 352, 150, 20, "HAS_CATEGORY", size=12, color=MUTED)
    g.text(712, 138, 126, 20, "PARENT_CATEGORY", size=11.5, color=MUTED)
    g.text(552, 418, 110, 20, "HAS_PROPERTY", size=12, color=MUTED, align="left")

    # Lop truy vet: mot su kien cua u1 duoi dang node
    g.node(30, 430, 470, 110, "", "rounded=1;html=1;fillColor=none;strokeColor=#8A9099;dashed=1;arcSize=6;")
    g.text(40, 436, 330, 22, "Lớp truy vết (Neo4j): mỗi sự kiện là một node", size=12.5, align="left",
           color=MUTED, extra="fontStyle=1;")
    t1 = g.node(50, 475, 50, 50, "u1", CIRCLE + "fontSize=14;")
    te = g.node(200, 478, 130, 44, "Event<br>mua · 06/08/2015",
                BOX + "fillColor=#FFFFFF;dashed=1;fontSize=12.5;strokeColor=#5B6068;")
    ti = g.node(420, 476, 60, 46, "i1", ITEM + "fontSize=14;")
    g.edge(t1, te, "PERFORMED", "strokeColor=#5B6068;dashed=1;fontColor=#5B6068;fontSize=11.5;", ly=-11)
    g.edge(te, ti, "TARGETS", "strokeColor=#5B6068;dashed=1;fontColor=#5B6068;fontSize=11.5;", ly=-11)

    g.text(30, 560, 860, 22,
           "Nét đỏ: INTERACTED_WITH, độ dày tỉ lệ với W(u,i) = Σ α<sub>b</sub>·e<sup>−λΔt</sup>, "
           "tính tại T<sub>train</sub> với α = 1 / 2 / 3, λ = 0,05.", size=12.5, align="left", color=INK,
           extra="fontStyle=1;")
    for k, dong in enumerate(bang):
        g.text(30 + (k % 2) * 430, 586 + (k // 2) * 22, 420, 22, dong, size=13, align="left", color=ACCENT)
    return g.xml("Hinh 3.2")


# ── Hinh 3.3 ────────────────────────────────────────────────────────────


def ngay(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%d/%m/%Y")


def hinh_3_3() -> str:
    s = json.loads(SPLIT.read_text())
    d = 86_400_000
    tr = (s["t_train"] - s["t_min"]) / d
    va = (s["t_valid_end"] - s["t_train"]) / d
    te = (s["t_max"] - s["t_valid_end"]) / d
    g = Do()
    x0, x1, yb, hb = 40, 840, 70, 46
    k = (x1 - x0) / (tr + va + te)
    xt, xv = round(x0 + tr * k), round(x0 + (tr + va) * k)
    SEG = "rounded=0;whiteSpace=wrap;html=1;strokeColor=#1F2328;strokeWidth=1.2;fontSize=13.5;"
    sa = g.node(x0, yb, xt - x0, hb, f"<b>Huấn luyện</b> · {so(tr, 1)} ngày", SEG + f"fillColor={FILL};", "sa")
    sb = g.node(xt, yb, xv - xt, hb, f"<b>Xác thực</b><br>{so(va, 1)} ngày", SEG + "fillColor=#FFFFFF;fontSize=12;", "sb")
    sc = g.node(xv, yb, x1 - xv, hb, f"<b>Kiểm tra</b> · {so(te, 1)} ngày", SEG + "fillColor=#FFFFFF;", "sc")
    # T_train mot hang rieng phia tren: moc 07/08 va 21/08 chi cach nhau 78 don vi.
    g.text(xt - 90, yb - 58, 140, 22, f"T<sub>train</sub> = {ngay(s['t_train'])}", size=13,
           align="right", extra="fontStyle=1;")
    g.edge(g.text(xt - 2, yb - 36, 4, 4, ""), sa, "", "endArrow=none;strokeColor=#1F2328;exitX=0.5;exitY=1;"
           "entryX=1;entryY=0;strokeWidth=1.2;")
    for x, ms in [(x0, s["t_min"]), (xv, s["t_valid_end"]), (x1, s["t_max"])]:
        g.text(x - 60, yb - 30, 120, 22, ngay(ms), size=12.5, color=MUTED)

    BOX = "rounded=1;whiteSpace=wrap;html=1;arcSize=6;strokeColor=#1F2328;strokeWidth=1.2;fontSize=13;verticalAlign=middle;"
    ya, hbox = 200, 120
    a = g.node(40, ya, 280, hbox, "<b>Huấn luyện</b><br>dựng đồ thị tại T<sub>train</sub><br>"
               "BPR, 3 seed: 2020, 2021, 2022<br>mẫu âm chỉ từ sản phẩm trong đoạn này",
               BOX + f"fillColor={FILL};", "a")
    b = g.node(390, ya, 220, hbox, "<b>Xác thực</b><br>đo NDCG@20<br>dừng sau 20 lần không cải thiện<br>(nhóm Active: 50)",
               BOX + "fillColor=#FFFFFF;", "b")
    c = g.node(690, ya, 150, hbox, "<b>Kiểm tra</b><br>đánh giá một lần<br>Recall, NDCG,<br>HitRate, Coverage",
               BOX + "fillColor=#FFFFFF;", "c")
    NOI = "endArrow=none;strokeColor=#8A9099;dashed=1;strokeWidth=1;"
    g.edge(sa, a, "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;" + NOI)
    g.edge(sb, b, "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;" + NOI)
    g.edge(sc, c, "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;" + NOI)
    g.edge(a, b, "mỗi 5 epoch", "exitX=1;exitY=0.4;entryX=0;entryY=0.4;strokeWidth=1.3;", ly=-12)
    g.edge(b, a, "chưa dừng: huấn luyện tiếp", "exitX=0.5;exitY=1;entryX=0.5;entryY=1;strokeWidth=1.3;",
           points=[(500, ya + hbox + 36), (180, ya + hbox + 36)], ly=12)
    g.edge(b, c, "epoch tốt nhất", "exitX=1;exitY=0.4;entryX=0;entryY=0.4;strokeWidth=1.3;", ly=-12)
    return g.xml("Hinh 3.3")


# ── Xuat: luu .drawio va render PNG tu chinh file do ────────────────────


def render(xml: str, path: Path, w: int, h: int) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".drawio").write_text(xml, encoding="utf-8")
    cfg = json.dumps({"highlight": "none", "nav": False, "resize": False, "toolbar": "",
                      "lightbox": False, "border": 10, "xml": xml}, ensure_ascii=False)
    page = (f'<!doctype html><meta charset="utf-8"><body style="margin:0;background:#fff">'
            f'<div class="mxgraph" data-mxgraph="{html.escape(cfg, quote=True)}"></div>'
            f'<script src="{VIEWER}"></script></body>')
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "f.html"
        f.write_text(page, encoding="utf-8")
        subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu",
                        "--hide-scrollbars", "--force-device-scale-factor=3", "--virtual-time-budget=20000",
                        f"--window-size={w},{h}", f"--screenshot={path}", f.as_uri()],
                       check=True, capture_output=True, timeout=180)
    return path


HINH = {"Hình 3.2.": (hinh_3_2, "fig_3_2_schema.png", 960, 660),
        "Hình 3.3.": (hinh_3_3, "fig_3_3_quy_trinh.png", 960, 430)}


def main() -> dict[str, Path]:
    out = {cap: render(ve(), OUT / ten, w, h) for cap, (ve, ten, w, h) in HINH.items()}
    for cap, path in out.items():
        print(f"{cap} -> {path} (+ .drawio)")
    return out


if __name__ == "__main__":
    main()
