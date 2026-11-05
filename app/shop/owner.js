// Cổng nhà cung cấp: đồ thị tri thức động và cơ chế mô hình, theo Chương 3 của đề án.
import { $, api, esc, count, clock, dataTime, dataDate, ico, toast, errorState, emptyState, consoleShell, whoAmI, loginPage, bindLogout } from "./core.js";

const TABS = [["", "Đồ thị theo khách"], ["toan-do-thi", "Toàn đồ thị"], ["mo-hinh", "Mô hình và huấn luyện"], ["hieu-qua", "Hiệu quả mô hình"], ["phat-lai", "Phát lại dữ liệu thật"], ["he-thong", "Hệ thống và API"]];
const KIND = { customer: "khách khác", category: "danh mục", property: "thuộc tính", item: "sản phẩm" };
let timer = null;
let visitor = 962057;
let selected = null;

const short = (text, n = 30) => (text.length > n ? `${text.slice(0, n - 1)}…` : text);

function drawGraph(view, paths) {
  const touched = view.edges.slice(0, 6);
  const vias = paths.map((p) => p.via.item_id);
  vias.forEach((id) => { if (!touched.some((e) => e.item_id === id)) { const e = view.edges.find((x) => x.item_id === id); if (e) touched.push(e); } });
  const middles = [];
  paths.forEach((p) => { if (!middles.some((m) => m.id === p.middle.id)) middles.push(p.middle); });
  const rows = Math.max(touched.length, middles.length, 3);
  const h = Math.max(300, rows * 62 + 40);
  const W = 1120;
  const X = [56, 330, 690, 980];
  const y = (i, n) => 20 + ((h - 40) * (i + 0.5)) / n;
  const yT = Object.fromEntries(touched.map((e, i) => [e.item_id, y(i, touched.length)]));
  const yM = Object.fromEntries(middles.map((m, i) => [m.id, y(i, middles.length)]));
  const topW = Math.max(...touched.map((e) => e.W), 1e-6);
  const topS = Math.max(...paths.map((p) => p.strength), 1e-12);
  const curve = (x1, y1, x2, y2) => { const m = (x1 + x2) / 2; return `M${x1} ${y1} C${m} ${y1} ${m} ${y2} ${x2} ${y2}`; };
  const thick = (s) => 1.5 + 6 * Math.sqrt(s / topS);
  const parts = [];
  touched.forEach((e) => {
    parts.push(`<path class="edge w" d="${curve(X[0] + 52, h / 2, X[1] - 96, yT[e.item_id])}" stroke-width="${1 + 9 * Math.sqrt(e.W / topW)}"><title>W(u,i) = ${e.W}</title></path>`);
  });
  paths.forEach((p) => {
    const yv = yT[p.via.item_id], ym = yM[p.middle.id];
    if (yv === undefined) return;
    parts.push(`<path class="edge k-${p.kind}" d="${curve(X[1] + 96, yv, X[2] - 86, ym)}" stroke-width="${thick(p.strength)}"/>`);
    parts.push(`<path class="edge k-${p.kind}" d="${curve(X[2] + 86, ym, X[3] - 96, h / 2)}" stroke-width="${thick(p.strength)}"/>`);
  });
  const box = (cx, cy, w, cls, text, sub) => `<g class="node ${cls}"><rect x="${cx - w / 2}" y="${cy - 24}" width="${w}" height="48" rx="6"/><text x="${cx}" y="${sub ? cy - 4 : cy + 5}" text-anchor="middle" class="t1">${esc(short(text, Math.floor(w / 7.4)))}<title>${esc(text)}</title></text>${sub ? `<text x="${cx}" y="${cy + 14}" text-anchor="middle" class="t2">${esc(short(sub, Math.floor(w / 6.4)))}</text>` : ""}</g>`;
  parts.push(box(X[0], h / 2, 104, "n-me", view.name, `Khách ${view.visitor_id}`));
  touched.forEach((e) => parts.push(box(X[1], yT[e.item_id], 192, "n-item", e.name, `W = ${e.W}`)));
  middles.forEach((m) => parts.push(box(X[2], yM[m.id], 172, `n-${m.kind}`, m.label, m.detail)));
  const target = view.recommended.find((i) => i.item_id === selected);
  if (target) parts.push(box(X[3], h / 2, 192, "n-target", target.name, "sản phẩm gợi ý"));
  return `<svg class="kg" viewBox="0 0 ${W} ${h}" role="img" aria-label="Đồ thị của ${esc(view.name)}: lịch sử, nút trung gian và sản phẩm gợi ý">${parts.join("")}</svg>
    <p class="legend small"><span class="k k-w"></span>cạnh hành vi W(u,i)<span class="k k-customer"></span>qua khách khác<span class="k k-category"></span>qua danh mục<span class="k k-property"></span>qua thuộc tính</p>`;
}

async function graphTab(pane, notice = "") {
  try {
    const [people, view] = await Promise.all([api("/owner/customers"), api(`/owner/graph/${visitor}`)]);
    if (!view.recommended.some((i) => i.item_id === selected)) selected = view.recommended[0]?.item_id ?? null;
    const paths = selected ? await api(`/owner/graph/${visitor}/paths/${selected}`) : [];
    if (!pane.isConnected) return;
    pane.innerHTML = `<div class="toolbar">
      <label class="small">Khách <select class="field" id="who">${people.map((p) => `<option value="${p.visitor_id}" ${p.visitor_id === visitor ? "selected" : ""}>${esc(p.name)} (${p.visitor_id})</option>`).join("")}</select></label>
      <span class="small"><b>Đồng hồ mô hình τ:</b> ${dataTime(view.tau)}</span>
      <button class="btn primary sm" id="next">${ico("play")}Sự kiện thật tiếp theo</button>
      <button class="btn sm" id="back">Về mốc kết thúc huấn luyện ${dataDate(view.t_train)}</button></div>
      ${notice ? `<div class="note wide-note">${notice}</div>` : ""}
      <div class="kg-wrap"><aside class="topk"><h3>Top-K của khách</h3><ol>${view.recommended.map((i) => `<li><button data-pick="${i.item_id}" ${i.item_id === selected ? 'aria-current="true"' : ""}>${esc(i.name)}${i.available ? "" : ` <span class="muted">(hết hàng)</span>`}</button></li>`).join("")}</ol></aside>
      <div class="kg-box">${selected ? drawGraph(view, paths) : emptyState("Khách chưa có lịch sử nên chưa có đồ thị.")}
      <p class="small muted" style="max-width:75ch">Đường đi lấy từ ma trận Â mà mô hình lan truyền, ba bước: khách, sản phẩm đã chạm, nút trung gian, sản phẩm gợi ý. Độ dày thể hiện ảnh hưởng của đường đó, không phải phân rã điểm, vì điểm còn phụ thuộc biểu diễn đã học.</p></div></div>
      <div class="two wide" style="margin-top:var(--space-6)"><section><h2 style="margin-bottom:var(--space-2)">Từng sự kiện, đồ thị truy vết</h2><p class="small muted" style="margin-bottom:var(--space-3)">w = α·e^(−λΔt), λ = ${view.lambda}, Δt tính tới τ.</p>
      <div class="table-wrap scroll"><table><thead><tr><th>Thời gian ghi nhận</th><th>Sản phẩm</th><th>Hành vi</th><th class="r">α</th><th class="r">Δt (ngày)</th><th class="r">w</th></tr></thead><tbody>${view.events.map((e) => `<tr class="${e.new ? "best" : ""}"><td class="num">${dataTime(e.timestamp)}</td><td>${esc(e.item)}</td><td>${esc(e.behavior)}</td><td class="r">${e.alpha}</td><td class="r">${e.age_days.toLocaleString("vi-VN")}</td><td class="r">${e.w}</td></tr>`).join("")}</tbody></table></div></section>
      <section style="margin-top:0"><h2 style="margin-bottom:var(--space-2)">Cạnh tổng hợp, đồ thị huấn luyện</h2><p class="small muted" style="margin-bottom:var(--space-3)">W(u,i) là tổng w của các sự kiện cùng cặp khách, sản phẩm.</p>
      <div class="table-wrap scroll"><table><thead><tr><th>Sản phẩm</th><th class="r">W(u,i)</th></tr></thead><tbody>${view.edges.map((e) => `<tr><td>${esc(e.name)}</td><td class="r">${e.W}</td></tr>`).join("")}</tbody></table></div></section></div>`;
    $("#who").addEventListener("change", (e) => { visitor = Number(e.target.value); selected = null; graphTab(pane); });
    pane.querySelectorAll("[data-pick]").forEach((b) => b.addEventListener("click", () => { selected = Number(b.dataset.pick); graphTab(pane); }));
    $("#back").addEventListener("click", async () => { await api(`/owner/graph/${visitor}/reset`, { method: "POST" }); selected = null; graphTab(pane); });
    $("#next").addEventListener("click", async (e) => {
      e.target.disabled = true;
      try {
        const r = await api(`/owner/graph/${visitor}/next`, { method: "POST" });
        if (r.done) return graphTab(pane, "Khách này đã hết sự kiện thật sau mốc huấn luyện.");
        const had = !r.first_touch ? "Sản phẩm khách đã chạm trước đó." : r.rank_before ? `Lần đầu khách chạm sản phẩm này, và nó <b>đã có sẵn ở hạng ${r.rank_before}</b> trong Top-20.` : "Lần đầu khách chạm sản phẩm này, chưa có trong Top-20.";
        selected = null;
        graphTab(pane, `<b>${dataTime(r.timestamp)}:</b> khách ${esc(r.behavior)} ${esc(r.item.name)}, w = ${r.weight}. ${had} Top-20 có ${r.n_new} sản phẩm mới. Còn ${r.remaining} sự kiện.`);
      } catch (error) { toast(error.message); e.target.disabled = false; }
    });
  } catch (error) { pane.innerHTML = errorState(error); }
}

async function results(pane) {
  try {
    const { rows, serving } = await api("/owner/results");
    const top = Math.max(...rows.map((r) => r["recall@20"]));
    pane.innerHTML = `<h2 style="margin-bottom:var(--space-2)">Recall@20 trên tập kiểm thử, nhóm Active</h2>
      <p class="small muted" style="margin-bottom:var(--space-5)">Đọc trực tiếp từ các lần chạy chính thức, trung bình 3 seed, trùng số trong Chương 4.</p>
      <div class="bar-chart">${rows.map((r) => `<div class="bar-row ${r.model === "bt_dkgrec_l05" ? "best" : ""}"><span>${esc(r.label)}</span><div class="bar-track"><div class="bar-fill" style="transform:scaleX(${r["recall@20"] / top})"></div></div><span class="num">${r["recall@20"].toFixed(4)}</span></div>`).join("")}</div>
      <div class="table-wrap" style="margin-top:var(--space-6)"><table><thead><tr><th>Mô hình</th><th class="r">Recall@20</th><th class="r">NDCG@20</th><th class="r">HitRate@20</th><th class="r">Coverage@20</th></tr></thead><tbody>${rows.map((r) => `<tr class="${r.model === "bt_dkgrec_l05" ? "best" : ""}"><td>${esc(r.label)}</td><td class="r">${r["recall@20"].toFixed(4)}</td><td class="r">${r["ndcg@20"].toFixed(4)}</td><td class="r">${r["hit_rate@20"].toFixed(4)}</td><td class="r">${r["coverage@20"].toFixed(4)}</td></tr>`).join("")}</tbody></table></div>
      ${serving ? `<div class="note" style="margin-top:var(--space-5)">Bản trình diễn chạy lần huấn luyện riêng trên CPU để lưu được trọng số, không nằm trong bảng trên: Recall@20 ${serving["recall@20"].toFixed(4)}, NDCG@20 ${serving["ndcg@20"].toFixed(4)}.</div>` : ""}`;
  } catch (error) { pane.innerHTML = errorState(error); }
}

async function replay(pane) {
  const draw = (s) => {
    const running = s.status === "running";
    const rate = (a, b) => (b ? `${((a / b) * 100).toLocaleString("vi-VN", { maximumFractionDigits: 1 })}%` : "0%");
    pane.innerHTML = `<div class="note">Đẩy các sự kiện thật của giai đoạn kiểm thử theo đúng thứ tự thời gian qua đường ghi đồ thị rồi tính Top-K. Mỗi khi khách chạm một sản phẩm lần đầu, kiểm tra sản phẩm đó đã có sẵn trong Top-20 chưa. Xong sẽ tự xóa dữ liệu đã ghi. Số này khác cách đo ở Chương 4.</div>
      <div class="toolbar"><label class="small">Số sự kiện <select class="field" id="n"><option>100</option><option>300</option><option selected>1000</option></select></label>
      <button class="btn primary sm" id="go" ${running ? "disabled" : ""}>${ico("play")}${running ? "Đang phát lại" : "Bắt đầu phát lại"}</button></div>
      ${s.done > 0 ? `<div class="meter" role="progressbar" aria-valuemin="0" aria-valuemax="${s.total}" aria-valuenow="${s.done}"><i style="transform:scaleX(${s.done / (s.total || 1)})"></i></div>
      <div class="stats"><div class="stat"><b>${count(s.done)}/${count(s.total)}</b><small>sự kiện, ${s.visitors} khách</small></div>
      <div class="stat"><b>${s.event_time ? dataDate(s.event_time) : ""}</b><small>đang phát tới ngày này, thời điểm thật của sự kiện</small></div>
      <div class="stat"><b>${rate(s.hits, s.targets)}</b><small>có sẵn trong Top-20 của mô hình, ${s.hits}/${s.targets} lần chạm đầu</small></div>
      <div class="stat"><b>${rate(s.hits_popular, s.targets)}</b><small>có trong Top-20 bán chạy, ${s.hits_popular}/${s.targets} lần chạm đầu</small></div></div>
      ${s.status === "done" ? `<p class="small muted">Xong${s.cleaned ? ", đã dọn dữ liệu phát lại" : ""}.</p>` : ""}` : ""}
      ${s.status === "error" ? `<div class="state"><p>Dừng vì lỗi: ${esc(s.error)}</p></div>` : ""}`;
    $("#go").addEventListener("click", async () => {
      const state = await api(`/owner/replay?limit=${$("#n").value}`, { method: "POST" });
      draw(state);
      clearInterval(timer);
      timer = setInterval(async () => {
        const next = await api("/owner/replay");
        if (!pane.isConnected) return clearInterval(timer);
        draw(next);
        if (next.status !== "running" && next.cleaned) clearInterval(timer);
      }, 800);
    });
  };
  try { draw(await api("/owner/replay")); } catch (error) { pane.innerHTML = errorState(error); }
}

async function wholeGraph(pane) {
  try {
    const g = await api("/owner/whole-graph");
    const t = g.training;
    const pos = { c: [110, 120], i: [540, 120], k: [960, 50], p: [960, 190], e: [325, 270] };
    const node = (key, label, n, cls) => `<g class="node ${cls}"><rect x="${pos[key][0] - 95}" y="${pos[key][1] - 30}" width="190" height="60" rx="6"/><text x="${pos[key][0]}" y="${pos[key][1] - 5}" text-anchor="middle" class="t1">${esc(label)}</text><text x="${pos[key][0]}" y="${pos[key][1] + 15}" text-anchor="middle" class="t2">${esc(n)}</text></g>`;
    const curve = (x1, y1, x2, y2, cls) => `<path class="edge ${cls}" d="M${x1} ${y1} C${(x1 + x2) / 2} ${y1} ${(x1 + x2) / 2} ${y2} ${x2} ${y2}" stroke-width="3"/>`;
    const tag = (x, y, text) => `<text x="${x}" y="${y}" text-anchor="middle" class="kg-label">${esc(text)}</text>`;
    pane.innerHTML = `<h2 style="margin-bottom:var(--space-2)">Đồ thị huấn luyện toàn cửa hàng</h2>
    <p class="small muted" style="margin-bottom:var(--space-4);max-width:75ch">Sơ đồ loại nút và loại cạnh, kèm số lượng thật. ${count(t.n_nodes_total)} nút không thể vẽ từng nút, nên muốn xem chi tiết hãy dùng tab Đồ thị theo khách.</p>
    <div class="kg-box"><svg class="kg" viewBox="0 0 1080 320" role="img" aria-label="Lược đồ đồ thị tri thức động với số lượng nút và cạnh">
      ${curve(205, 120, 445, 120, "w")}${tag(325, 108, `W(u,i): ${count(t.edges.interacted_with)} cạnh`)}
      ${curve(635, 120, 865, 50, "k-category")}${tag(750, 66, `${count(t.edges.has_category)} cạnh`)}
      ${curve(635, 120, 865, 190, "k-property")}${tag(750, 186, `${count(t.edges.has_property)} cạnh`)}
      ${curve(110, 150, 230, 270, "k-customer")}${curve(420, 270, 540, 150, "k-customer")}${tag(325, 225, "Đồ thị truy vết: mỗi hành vi là một nút")}
      ${node("c", "Khách", count(t.n_visitor), "n-me")}${node("i", "Sản phẩm", count(t.n_item), "n-item")}${node("k", "Danh mục", `${count(t.n_category)}, cạnh cha con ${count(t.edges.parent_category)}`, "n-category")}${node("p", "Thuộc tính", count(t.n_property_value), "n-property")}
      ${node("e", "Sự kiện", `${count(g.store_nodes.Event || 0)} trong kho`, "n-customer")}
    </svg></div>
    <div class="two wide" style="margin-top:var(--space-5)"><section><h3 style="margin-bottom:var(--space-2)">Lớp động và lớp ngữ nghĩa</h3><p class="small" style="max-width:60ch">Danh mục và thuộc tính là lớp ngữ nghĩa. Lớp động là cạnh khách, sản phẩm có trọng số w = α·e^(−λΔt), λ = ${t.lambda_decay}, α xem ${t.alpha.view}, thêm giỏ ${t.alpha.addtocart}, mua ${t.alpha.transaction}. Mốc ảnh chụp ${dataTime(t.t_train)}.</p></section>
    <section style="margin-top:0"><h3 style="margin-bottom:var(--space-2)">Kho đồ thị đang chạy</h3><div class="table-wrap"><table><tbody>${Object.entries(g.store_nodes).map(([k, v]) => `<tr><td>Nút ${esc(k)}</td><td class="r">${count(v)}</td></tr>`).join("")}${Object.entries(g.store_edges).map(([k, v]) => `<tr><td>Cạnh ${esc(k)}</td><td class="r">${count(v)}</td></tr>`).join("")}</tbody></table></div></section></div>`;
  } catch (error) { pane.innerHTML = errorState(error); }
}

async function models(pane) {
  const draw = async () => {
    const { versions, pending } = await api("/owner/models");
    pane.innerHTML = `<div class="note">Quy trình theo mục 3.7.1: đồ thị cập nhật liên tục, mô hình huấn luyện lại theo chu kỳ trên ảnh chụp đồ thị, đạt thì áp dụng. Bước áp dụng và quay lại phiên bản cũ đã chạy được. Bước tự động huấn luyện mỗi tuần chưa có.</div>
      <div class="stats"><div class="stat"><b>${count(pending.new_events)}</b><small>hành vi mới chờ lần huấn luyện tới</small></div><div class="stat"><b>${count(pending.new_products)}</b><small>sản phẩm mới chờ vào đồ thị</small></div><div class="stat"><b>${count(versions.length)}</b><small>phiên bản mô hình đã lưu</small></div><div class="stat"><b>Chưa có</b><small>lịch huấn luyện tự động</small></div></div>
      <h2 style="margin-bottom:var(--space-3)">Các phiên bản mô hình</h2>
      <div class="table-wrap"><table><thead><tr><th>Phiên bản</th><th>Huấn luyện lúc</th><th class="r">NDCG@20 tập xác thực</th><th class="r">Recall@20 tập kiểm thử</th><th class="r">NDCG@20 tập kiểm thử</th><th></th></tr></thead><tbody>${versions.map((v) => `<tr class="${v.active ? "best" : ""}"><td>${esc(v.name)}</td><td class="num">${clock(v.trained)}</td><td class="r">${v.valid_ndcg20.toFixed(4)}</td><td class="r">${v.recall20 == null ? "" : v.recall20.toFixed(4)}</td><td class="r">${v.ndcg20 == null ? "" : v.ndcg20.toFixed(4)}</td><td class="r">${v.active ? "Đang dùng" : `<button class="btn sm primary" data-apply="${esc(v.name)}">Áp dụng</button>`}</td></tr>`).join("")}</tbody></table></div>
      <p class="small muted" style="margin-top:var(--space-3);max-width:75ch">Chọn mô hình theo tập xác thực, không theo tập kiểm thử.</p>`;
    pane.querySelectorAll("[data-apply]").forEach((b) => b.addEventListener("click", async () => {
      b.disabled = true;
      try { await api("/owner/models/apply", { method: "POST", body: { name: b.dataset.apply } }); toast("Đã áp dụng phiên bản mới."); draw(); } catch (error) { toast(error.message); b.disabled = false; }
    }));
  };
  try { await draw(); } catch (error) { pane.innerHTML = errorState(error); }
}

async function system(pane) {
  const base = location.origin;
  try {
    const o = await api("/api/v1/health");
    pane.innerHTML = `<div class="two"><section><h2 style="margin-bottom:var(--space-3)">Năm lớp kiến trúc, mục 3.7.1</h2>
    <div class="table-wrap"><table><thead><tr><th>Lớp</th><th>Trong bản này</th></tr></thead><tbody>
    <tr><td>Tiếp nhận sự kiện</td><td>Đã có: <code>POST /api/v1/events</code></td></tr>
    <tr><td>Cập nhật đồ thị</td><td>Đã có: ghi sự kiện và cạnh vào Neo4j</td></tr>
    <tr><td>Kho đồ thị tri thức động</td><td>${o.neo4j ? "Đang chạy" : "Mất kết nối"}: ${count(o.n_events)} sự kiện, ${count(o.n_visitors)} khách. Chưa có ảnh chụp đồ thị tự động</td></tr>
    <tr><td>Huấn luyện ngoại tuyến</td><td>Chạy tay. Chưa tự động theo chu kỳ</td></tr>
    <tr><td>Dịch vụ dự báo Top-K</td><td>Đã có: ${esc(o.model)}, phía khách tính lại theo hành vi mới, phía sản phẩm giữ từ lần huấn luyện</td></tr></tbody></table></div></section>
    <section style="margin-top:0"><h2 style="margin-bottom:var(--space-3)">Tích hợp vào cửa hàng</h2>
<pre>curl -X POST ${base}/api/v1/events \\
  -H "X-API-Key: KHOA_API_CUA_BAN" -H "Content-Type: application/json" \\
  -d '{"visitor_id": 962057, "item_id": 85791, "behavior": "addtocart"}'

curl ${base}/api/v1/recommendations/962057?k=10</pre>
    <div class="toolbar"><a class="btn sm" href="/docs" target="_blank" rel="noopener">Mở tài liệu API</a><button class="btn sm danger" id="reset">Đưa dữ liệu trình diễn về ban đầu</button></div></section></div>`;
    $("#reset").addEventListener("click", async () => {
      if (!confirm("Xóa mọi hành vi mới, đơn hàng và ưu đãi?")) return;
      const done = await api("/owner/reset", { method: "POST" });
      toast(`Đã xóa ${done.removed} sự kiện mô phỏng.`);
    });
  } catch (error) { pane.innerHTML = errorState(error); }
}

export async function ownerRoute(parts, route) {
  clearInterval(timer);
  const user = await whoAmI("provider");
  if (!user) return loginPage("provider", "Nhà cung cấp giải pháp", "Đăng nhập bằng tài khoản nhà cung cấp.", () => route());
  const tab = parts[1] || "";
  const pane = consoleShell("nha-cung-cap", ["BT-DKGRec", "Nhà cung cấp giải pháp", user.display], TABS, tab);
  bindLogout("provider", route);
  if (tab === "toan-do-thi") return wholeGraph(pane);
  if (tab === "mo-hinh") return models(pane);
  if (tab === "hieu-qua") return results(pane);
  if (tab === "phat-lai") return replay(pane);
  if (tab === "he-thong") return system(pane);
  return graphTab(pane);
}
export const stopOwner = () => clearInterval(timer);
