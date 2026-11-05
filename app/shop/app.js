// Chợ Hiên: cửa hàng và kênh người bán. Mọi danh sách cá nhân hóa đến từ mô hình qua API.
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const money = (v) => `${Number(v).toLocaleString("vi-VN")}đ`;
const count = (v) => Number(v).toLocaleString("vi-VN");
const ico = (name) => `<span class="ico" style="--i:url(/assets/icons/${name}.svg)" aria-hidden="true"></span>`;
const BEHAVIOR_DONE = { view: "Đã ghi nhận lượt xem", addtocart: "Đã thêm vào giỏ", transaction: "Đã đặt hàng" };

async function api(path, options = {}) {
  const response = await fetch(path, options.body
    ? { ...options, headers: { "content-type": "application/json" }, body: JSON.stringify(options.body) } : options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Máy chủ trả lỗi ${response.status}`);
  }
  return response.json();
}

const store = {
  get user() { try { return JSON.parse(localStorage.getItem("user")); } catch { return null; } },
  set user(v) { try { v ? localStorage.setItem("user", JSON.stringify(v)) : localStorage.removeItem("user"); } catch { /* private mode */ } },
};
let session = store.user;
let lastList = null;
let departments = [];
let cartCount = 0;
let pollTimer = null;

function toast(text) {
  const el = $("#toast");
  el.textContent = text;
  el.classList.add("on");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove("on"), 3200);
}

// ── Label ───────────────────────────────────────────────────────────
function tone(item) { return `--bg:var(--l${item.tone});--fg:var(--li${item.tone})`; }
function label(item, link = true, extra = "") {
  const tag = link ? "a" : "div";
  const href = link ? `href="#/sp/${item.item_id}"` : "";
  return `<${tag} class="label" ${href} style="${tone(item)}">${extra}${ico(item.icon)}<b>${esc(item.product_type)}</b><span>${esc(item.variant)}</span></${tag}>`;
}
function productCard(item, change = "", cls = "") {
  const out = !item.available;
  const flag = change === "mới" ? `<span class="tag new">Mới</span>` : change ? `<span class="tag">${esc(change)}</span>` : "";
  return `<article class="product ${cls} ${out ? "is-out" : ""} ${change === "mới" ? "is-new" : ""}">
    ${label(item, true, flag)}
    <a class="name" href="#/sp/${item.item_id}">${esc(item.name)}</a>
    <div class="product-foot"><span class="price">${money(item.price)}</span>
    ${out ? `<span class="small muted">Tạm hết</span>` : `<button class="btn sm" data-add="${item.item_id}" aria-label="Thêm ${esc(item.name)} vào giỏ">${ico("plus")}Giỏ</button>`}</div>
  </article>`;
}
function changeOf(prev, id, rank) {
  if (!prev) return "";
  const i = prev.indexOf(id);
  if (i < 0) return "mới";
  return i > rank ? `Lên ${i - rank} bậc` : "";
}
const skeletonGrid = (n = 5) => `<div class="grid" aria-busy="true">${`<div class="skeleton" style="aspect-ratio:4/5"></div>`.repeat(n)}</div>`;
const errorState = (error) => `<div class="state"><p>Chưa tải được: ${esc(error.message)}</p><button class="btn" onclick="location.reload()">Thử lại</button></div>`;
const emptyState = (text, action = "") => `<div class="state"><p>${esc(text)}</p>${action}</div>`;

// ── Chrome ──────────────────────────────────────────────────────────
function renderBar(active = "") {
  const seller = location.hash.startsWith("#/ban-hang");
  $("#bar").innerHTML = `<div class="bar-in">
    <a class="mark" href="#/">Chợ Hiên</a>
    ${seller ? `<span class="small muted">Kênh người bán</span>` : `<form class="search" role="search" id="search"><input name="q" type="search" placeholder="Tìm áo polo, nồi inox, sổ tay" aria-label="Tìm sản phẩm"><button aria-label="Tìm">${ico("search")}</button></form>`}
    <nav class="bar-links" aria-label="Tài khoản">
      ${seller ? `<a href="#/">${ico("shopping-bag")}Về cửa hàng</a>` : `<a href="#/ban-hang">${ico("store")}Kênh người bán</a>
      <a href="#/tai-khoan">${ico("user")}${esc(session ? session.name : "Đăng nhập")}</a>
      <a href="#/gio" aria-label="Giỏ hàng, ${cartCount} sản phẩm">${ico("shopping-bag")}Giỏ${cartCount ? `<span class="badge">${cartCount}</span>` : ""}</a>`}
    </nav></div>
    ${seller ? "" : `<nav class="depts" aria-label="Ngành hàng"><div class="depts-in">${departments.filter((d) => d.dept_idx >= 0).slice(0, 13).map((d) =>
      `<a href="#/nganh/${encodeURIComponent(d.department)}" ${active === d.department ? 'aria-current="true"' : ""}>${ico(d.icon)}${esc(d.department)}</a>`).join("")}</div></nav>`}`;
  const form = $("#search");
  if (form) form.addEventListener("submit", (e) => { e.preventDefault(); location.hash = `#/tim/${encodeURIComponent(form.q.value.trim())}`; });
}
function renderFoot() {
  $("#foot").innerHTML = `<div class="foot-in"><p>Chợ Hiên là cửa hàng mô phỏng của đề án BT-DKGRec-GCN. Hành vi, danh mục và gợi ý là thật từ RetailRocket và mô hình; tên sản phẩm, tên khách và nhãn hàng là minh họa vì dữ liệu gốc đã ẩn danh. Đơn hàng không thu tiền.</p></div>`;
}
function showBehind(result) {
  const el = $("#behind");
  if (!result || location.hash.startsWith("#/ban-hang")) { el.hidden = true; return; }
  el.hidden = false;
  const open = showBehind.open;
  const steps = [
    ["Ghi hành vi vào Neo4j", result.write_ms], ["Đọc lịch sử từ Neo4j", result.read_ms],
    [result.source === "personal" ? "Tính lại gợi ý bằng mô hình" : "Lấy danh sách bán chạy", result.model_ms],
  ];
  el.innerHTML = `<button class="head" aria-expanded="${!!open}">Phía sau hệ thống<span class="t">${Math.round(result.total_ms)} ms</span></button>
    ${open ? `<ol>${steps.map(([a, b]) => `<li><span>${a}</span><span class="t">${Math.round(b)} ms</span></li>`).join("")}
    <li><span>Trọng số sự kiện vừa ghi</span><span class="t">${result.weight}</span></li>
    <li><span>Số sự kiện của khách</span><span class="t">${count(result.n_events)}</span></li></ol>` : ""}`;
  $(".head", el).addEventListener("click", () => { showBehind.open = !open; showBehind(result); });
}

async function refreshCart() {
  if (!session) return;
  const cart = await api(`/shop/cart/${session.visitor_id}`);
  cartCount = cart.count;
}

// ── Pages: shop ─────────────────────────────────────────────────────
async function pageEntry() {
  const main = $("#main");
  main.innerHTML = `<div class="who"><div><h1>Chợ gần nhà,<br>gợi ý đúng người.</h1>
    <p class="muted" style="margin-top:var(--space-4)">Chọn một khách hàng để xem cửa hàng thay đổi theo chính hành vi của họ.</p></div>
    <div class="who-list" id="who"><div class="skeleton" style="height:280px"></div></div></div>`;
  try {
    const people = await api("/shop/personas");
    $("#who").innerHTML = people.map((p) => `<button data-who="${p.visitor_id}" data-name="${esc(p.name)}"><span><b>${esc(p.name)}</b><br><span class="small muted">Khách ${p.visitor_id}, ${count(p.n_events)} hành vi, ${count(p.n_bought)} lần mua</span></span>${ico("arrow-left")}</button>`).join("")
      + `<button data-guest="1"><span><b>Khách mới</b><br><span class="small muted">Chưa có lịch sử, bắt đầu từ danh sách bán chạy</span></span></button>`;
    $("#who").querySelectorAll(".ico").forEach((i) => { i.style.transform = "scaleX(-1)"; });
  } catch (error) { $("#who").innerHTML = errorState(error); }
}

async function pageHome() {
  const main = $("#main");
  main.innerHTML = `<div class="greet"><h1>Chào ${esc(session.name)}</h1></div><div id="offers"></div><div id="shelf">${skeletonGrid()}</div><section id="pop"></section>`;
  try {
    const [me, list] = await Promise.all([api(`/shop/me/${session.visitor_id}`), api(`/api/v1/recommendations/${session.visitor_id}?k=40`)]);
    $("#offers").innerHTML = me.offers.map((o) => `<div class="offer"><span><strong>Giảm ${o.discount}% riêng cho bạn</strong><br><span class="small">${esc(o.item.name)}, còn ${money(o.price_after)}</span></span><a class="btn primary sm" href="#/sp/${o.item.item_id}?uu-dai=${o.campaign_id}">Xem ưu đãi</a></div>`).join("");
    const items = list.items.filter((i) => i.available).slice(0, 9);
    const ids = items.map((i) => i.item_id);
    const prev = lastList && lastList.visitor === session.visitor_id ? lastList.ids : null;
    const personal = list.source === "personal";
    const lead = items[0];
    const why = personal && lead ? await api(`/api/v1/recommendations/${session.visitor_id}/why/${lead.item_id}`) : null;
    $("#shelf").innerHTML = lead ? `<div class="sec-head"><h2>${personal ? "Kệ hàng của bạn" : "Đang bán chạy"}</h2>
        <span class="small muted">${personal ? `Theo ${count(me.n_events)} hành vi của bạn` : "Xem hoặc mua một món, kệ này sẽ đổi theo bạn"}</span></div>
      <div class="shelf">${items.map((item, i) => productCard(item, changeOf(prev, item.item_id, i), i === 0 ? "lead" : "")).join("")}</div>
      ${why ? `` : ""}` : emptyState("Kệ hàng đang trống. Hãy tìm một sản phẩm để bắt đầu.");
    lastList = { visitor: session.visitor_id, ids: list.items.filter((i) => i.available).map((i) => i.item_id) };
    if (why && why.seen) { const line = document.createElement("p"); line.className = "why-line"; line.textContent = whyText(why, lead); $("#shelf .lead").append(line); }
    if (personal) {
      const pop = await api(`/shop/items?limit=5&department=${encodeURIComponent(lead.department)}`);
      $("#pop").innerHTML = `<div class="sec-head"><h2>Bán chạy trong ${esc(lead.department)}</h2><a class="small" href="#/nganh/${encodeURIComponent(lead.department)}">Xem cả ngành hàng</a></div><div class="grid">${pop.filter((i) => i.available).map((i) => productCard(i)).join("")}</div>`;
    }
  } catch (error) { $("#shelf").innerHTML = errorState(error); }
}

function whyText(why, item) {
  const verb = { view: "đã xem", addtocart: "đã thêm vào giỏ", transaction: "đã mua" }[why.behavior] || "đã xem";
  if (why.kind === "direct") return `Bạn ${verb} chính sản phẩm này.`;
  if (why.kind === "category") return `Cùng nhóm ${item.product_type} với ${why.seen.name} bạn ${verb}.`;
  if (why.kind === "department") return `Cùng ngành ${item.department} với ${why.seen.name} bạn ${verb}.`;
  return "Những khách có hành vi gần giống bạn cũng quan tâm sản phẩm này.";
}

async function pageProduct(id, query) {
  const main = $("#main");
  main.innerHTML = `<div class="pdp"><div class="skeleton" style="aspect-ratio:1/1"></div><div class="skeleton" style="height:320px"></div></div>`;
  try {
    const item = await api(`/shop/items/${id}`);
    const campaign = Number(query.get("uu-dai")) || null;
    const me = await api(`/shop/me/${session.visitor_id}`);
    const offer = me.offers.find((o) => o.item.item_id === item.item_id);
    const seen = await api(`/api/v1/recommendations/${session.visitor_id}/why/${id}`);
    main.innerHTML = `<div class="pdp">${label(item, false)}
      <div><p class="small muted"><a href="#/nganh/${encodeURIComponent(item.department)}">${esc(item.department)}</a></p>
      <h1 style="margin-top:var(--space-2)">${esc(item.name)}</h1>
      <p class="pdp-price num">${offer ? `${money(offer.price_after)} <s class="small muted" style="font-weight:400">${money(item.price)}</s>` : money(item.price)}</p>
      <div class="pdp-actions">${item.available
        ? `<button class="btn primary" data-add="${item.item_id}">${ico("shopping-bag")}Thêm vào giỏ</button><button class="btn" data-buy="${item.item_id}">Mua ngay</button>`
        : `<button class="btn" disabled>Tạm hết hàng</button>`}</div>
      ${seen.kind !== "collaborative" || me.n_events ? `<div class="why"><h3>Vì sao bạn thấy sản phẩm này</h3><p class="small" style="margin-top:var(--space-1)">${esc(whyText(seen, item))}</p></div>` : ""}
      <dl class="facts"><dt>Mã gốc RetailRocket</dt><dd class="num">${item.item_id}</dd><dt>Danh mục gốc</dt><dd class="num">${item.category_id < 0 ? "Chưa có" : item.category_id}</dd>
      <dt>Lượt trong tập huấn luyện</dt><dd class="num">${count(item.n_view)} xem, ${count(item.n_cart)} giỏ, ${count(item.n_txn)} mua</dd><dt>Tên và giá</dt><dd>Minh họa${item.price_is_real ? ", giá quy đổi từ thuộc tính gốc" : ""}</dd></dl></div></div>
      ${item.similar.length ? `<section><div class="sec-head"><h2>Sản phẩm gần giống</h2><span class="small muted">Theo biểu diễn học được từ đồ thị</span></div><div class="grid">${item.similar.filter((i) => i.available).slice(0, 5).map((i) => productCard(i)).join("")}</div></section>` : ""}`;
    const result = await api("/shop/visit", { method: "POST", body: { visitor_id: session.visitor_id, item_id: item.item_id, campaign_id: campaign } });
    showBehind(result);
  } catch (error) { main.innerHTML = errorState(error); }
}

async function pageList(title, params, active = "") {
  const main = $("#main");
  renderBar(active);
  main.innerHTML = `<div class="greet"><h1>${esc(title)}</h1></div><div id="list">${skeletonGrid(10)}</div>`;
  try {
    const items = await api(`/shop/items?${params}`);
    $("#list").innerHTML = items.length ? `<div class="grid">${items.map((i) => productCard(i)).join("")}</div>`
      : emptyState("Không tìm thấy sản phẩm phù hợp.", `<a class="btn" href="#/">Về kệ hàng của bạn</a>`);
  } catch (error) { $("#list").innerHTML = errorState(error); }
}

async function pageCart() {
  const main = $("#main");
  main.innerHTML = `<h1 style="margin-bottom:var(--space-5)">Giỏ hàng</h1><div id="cart"><div class="skeleton" style="height:240px"></div></div>`;
  try {
    const cart = await api(`/shop/cart/${session.visitor_id}`);
    $("#cart").innerHTML = cart.lines.length ? `<div class="split"><div class="table-wrap"><table><thead><tr><th>Sản phẩm</th><th class="r">Đơn giá</th><th class="r">SL</th><th></th></tr></thead><tbody>
      ${cart.lines.map((l) => `<tr><td><span class="mini" style="${tone(l.item)}">${ico(l.item.icon)}</span><a href="#/sp/${l.item.item_id}">${esc(l.item.name)}</a>${l.discount ? ` <span class="small" style="color:var(--mark)">Giảm ${l.discount}%</span>` : ""}</td><td class="r">${money(l.price)}</td><td class="r">${l.qty}</td><td class="r"><button class="btn sm" data-remove="${l.item.item_id}" aria-label="Bỏ ${esc(l.item.name)}">${ico("x")}</button></td></tr>`).join("")}
      </tbody></table></div><aside class="summary"><div class="total"><span>Tổng</span><span class="num">${money(cart.total)}</span></div>
      <p class="small muted">Thanh toán khi nhận hàng. Đơn này là mô phỏng, không thu tiền.</p><button class="btn primary" id="order">Đặt hàng</button></aside></div>`
      : emptyState("Giỏ hàng đang trống.", `<a class="btn primary" href="#/">Về kệ hàng của bạn</a>`);
    const order = $("#order");
    if (order) order.addEventListener("click", placeOrder);
  } catch (error) { $("#cart").innerHTML = errorState(error); }
}

async function placeOrder() {
  $("#order").disabled = true;
  try {
    const result = await api("/shop/checkout", { method: "POST", body: { visitor_id: session.visitor_id } });
    cartCount = 0;
    showBehind(result);
    toast(`Đã đặt đơn #${result.order_id}. Kệ hàng của bạn vừa được cập nhật.`);
    location.hash = "#/";
  } catch (error) { toast(error.message); $("#order").disabled = false; }
}

async function pageAccount() {
  const main = $("#main");
  main.innerHTML = `<div id="acc"><div class="skeleton" style="height:300px"></div></div>`;
  try {
    const me = await api(`/shop/me/${session.visitor_id}`);
    $("#acc").innerHTML = `<div class="greet"><h1>${esc(me.name)}</h1><button class="btn sm" id="out">Đổi khách hàng</button></div>
      <div class="two"><section><h2 style="margin-bottom:var(--space-3)">Hoạt động của bạn</h2>
      ${me.timeline.length ? `<div class="table-wrap"><table><thead><tr><th>Sản phẩm</th><th>Hành vi</th><th class="r">Trọng số</th></tr></thead><tbody>${me.timeline.map((t) => `<tr><td><a href="#/sp/${t.item.item_id}">${esc(t.item.name)}</a>${t.demo ? ` <span class="small" style="color:var(--accent-2)">vừa rồi</span>` : ""}</td><td>${esc(t.behavior_vn)}</td><td class="r">${t.weight}</td></tr>`).join("")}</tbody></table></div>` : emptyState("Chưa có hoạt động nào.")}
      </section><section style="margin-top:0"><h2 style="margin-bottom:var(--space-3)">Đơn hàng</h2>
      ${me.orders.length ? me.orders.map((o) => `<p style="margin-bottom:var(--space-3)"><b>Đơn #${o.id}</b>, ${money(o.total)}<br><span class="small muted">${o.lines.map((l) => esc(l.item.name)).join(", ")}</span></p>`).join("") : `<p class="small muted">Chưa có đơn nào.</p>`}
      <h2 style="margin:var(--space-6) 0 var(--space-3)">Dữ liệu của bạn</h2>
      <p class="small muted" style="margin-bottom:var(--space-3)">Xóa ${count(me.n_new)} hành vi mới, giỏ hàng và đơn hàng trong cửa hàng này. Gợi ý trở về như trước.</p>
      <button class="btn danger sm" id="forget" ${me.n_new || me.orders.length ? "" : "disabled"}>${ico("trash-2")}Xóa dữ liệu của tôi</button></section></div>`;
    $("#out").addEventListener("click", () => { session = null; store.user = null; lastList = null; cartCount = 0; showBehind(null); location.hash = "#/"; route(); });
    $("#forget").addEventListener("click", async () => {
      if (!confirm("Xóa toàn bộ hành vi mới và đơn hàng của bạn?")) return;
      const done = await api(`/shop/me/${session.visitor_id}`, { method: "DELETE" });
      lastList = null; cartCount = 0; showBehind(null); toast(`Đã xóa ${done.removed} hành vi.`); route();
    });
  } catch (error) { $("#acc").innerHTML = errorState(error); }
}

// ── Pages: seller ───────────────────────────────────────────────────
const SELLER_TABS = [["", "Tổng quan"], ["du-bao", "Dự báo khách hàng"], ["chien-dich", "Chiến dịch"], ["phat-lai", "Phát lại dữ liệu thật"], ["hieu-qua", "Hiệu quả mô hình"], ["tich-hop", "Tích hợp"]];

function sellerShell(tab) {
  $("#main").innerHTML = `<nav class="console-top" aria-label="Kênh người bán">${SELLER_TABS.map(([k, v]) => `<a href="#/ban-hang${k ? "/" + k : ""}" ${k === tab ? 'aria-current="page"' : ""}>${v}</a>`).join("")}</nav><div id="pane"><div class="skeleton" style="height:320px"></div></div>`;
  return $("#pane");
}

async function sellerOverview(pane) {
  const draw = async () => {
    const [o, feed, carts] = await Promise.all([api("/seller/overview"), api("/seller/events?limit=14"), api("/seller/abandoned")]);
    if (!pane.isConnected) return;
    const h = o.health;
    pane.innerHTML = `<div class="stats">
      <div class="stat"><b>${count(o.n_events)}</b><small>hành vi mới đã thu</small></div>
      <div class="stat"><b>${Math.round(o.p50)} ms</b><small>độ trễ trung vị, ghi đến có gợi ý (p95 ${Math.round(o.p95)} ms)</small></div>
      <div class="stat"><b>${count(o.n_orders)}</b><small>đơn hàng mô phỏng</small></div>
      <div class="stat"><b>${money(o.revenue)}</b><small>giá trị đơn mô phỏng</small></div></div>
      <div class="two"><section><h2 style="margin-bottom:var(--space-3)">Hành vi vừa vào hệ thống</h2>
      ${feed.length ? `<div class="table-wrap"><table><thead><tr><th>Khách</th><th>Hành vi</th><th>Sản phẩm</th><th class="r">Độ trễ</th></tr></thead><tbody>${feed.map((e) => `<tr><td>${esc(e.name)}</td><td>${esc(e.behavior_vn)}</td><td>${esc(e.item_name)}</td><td class="r">${Math.round(e.total_ms)} ms</td></tr>`).join("")}</tbody></table></div>`
        : emptyState("Chưa có hành vi nào. Mở cửa hàng và xem một sản phẩm, dòng đầu tiên sẽ hiện ở đây.", `<a class="btn" href="#/">Mở cửa hàng</a>`)}
      </section><section style="margin-top:0"><h2 style="margin-bottom:var(--space-3)">Hệ thống</h2>
      <dl class="facts" style="margin-bottom:var(--space-5)"><dt>Kho đồ thị Neo4j</dt><dd>${h.neo4j ? "Đang chạy" : "Mất kết nối"}, ${count(h.n_events)} sự kiện</dd><dt>Mô hình phục vụ</dt><dd>${esc(h.model)}</dd><dt>Danh mục</dt><dd>${count(h.n_items)} sản phẩm</dd><dt>Khách trong kho</dt><dd>${count(h.n_visitors)} khách nhóm Active</dd><dt>Huấn luyện lại</dt><dd>Thủ công theo chu kỳ, chưa tự động</dd></dl>
      <h2 style="margin-bottom:var(--space-3)">Khách bỏ giỏ gần đây</h2>
      ${carts.length ? `<div class="table-wrap"><table><tbody>${carts.slice(0, 6).map((c) => `<tr><td>${esc(c.name)} <span class="muted small">${c.visitor_id}</span></td><td><a href="#/ban-hang/du-bao/${c.item.item_id}">${esc(c.item.name)}</a></td></tr>`).join("")}</tbody></table></div>` : `<p class="small muted">Không có.</p>`}
      <button class="btn sm danger" id="reset" style="margin-top:var(--space-5)">Đưa dữ liệu demo về ban đầu</button></section></div>`;
    $("#reset").addEventListener("click", async () => {
      if (!confirm("Xóa mọi hành vi mới, đơn hàng và chiến dịch?")) return;
      const done = await api("/seller/reset", { method: "POST" });
      session && (lastList = null); cartCount = 0; toast(`Đã xóa ${done.removed} sự kiện mô phỏng.`); draw();
    });
  };
  try { await draw(); pollTimer = setInterval(() => draw().catch(() => {}), 4000); } catch (error) { pane.innerHTML = errorState(error); }
}

async function sellerForecast(pane, itemId) {
  pane.innerHTML = `<div class="note">Chọn một sản phẩm, hệ thống xếp hạng khách có khả năng quan tâm nhất bằng điểm của mô hình, rồi gửi ưu đãi cho nhóm đó. Điểm dùng để xếp hạng, không phải xác suất mua.</div>
    <form class="toolbar" id="find"><input class="field" name="q" style="flex:1;min-width:220px;max-width:420px" placeholder="Tên hoặc mã sản phẩm" aria-label="Tìm sản phẩm" value="${itemId || ""}"><button class="btn sm">Tìm</button></form>
    <div id="found"></div><div id="aud"></div>`;
  const form = $("#find");
  const showAudience = async (id) => {
    $("#aud").innerHTML = `<div class="skeleton" style="height:240px"></div>`;
    try {
      const [item, rows] = await Promise.all([api(`/shop/items/${id}`), api(`/api/v1/items/${id}/audience?k=12`)]);
      $("#found").innerHTML = "";
      $("#aud").innerHTML = `<h2 style="margin-bottom:var(--space-3)">Khách nên tiếp cận cho ${esc(item.name)}</h2>
      <div class="table-wrap"><table><thead><tr><th></th><th class="r">Hạng</th><th>Khách</th><th class="r">Điểm mô hình</th><th class="r">Hành vi cùng ngành</th><th class="r">Lần thêm giỏ hoặc mua</th></tr></thead><tbody>
      ${rows.map((r) => `<tr><td><input type="checkbox" data-v="${r.visitor_id}" ${r.rank <= 5 ? "checked" : ""} aria-label="Chọn ${esc(r.name)}"></td><td class="r">${r.rank}</td><td>${esc(r.name)} <span class="muted small">${r.visitor_id}</span></td><td class="r">${r.score}</td><td class="r">${count(r.same_department)}</td><td class="r">${count(r.intent_events)}</td></tr>`).join("")}</tbody></table></div>
      <div class="toolbar" style="margin-top:var(--space-4)"><label class="small">Mức giảm <select class="field" id="disc"><option>5</option><option selected>10</option><option>15</option><option>20</option></select> %</label><button class="btn primary sm" id="send">${ico("send")}Gửi ưu đãi</button></div>`;
      $("#send").addEventListener("click", async () => {
        const visitors = [...document.querySelectorAll("[data-v]:checked")].map((c) => Number(c.dataset.v));
        try {
          await api("/seller/campaigns", { method: "POST", body: { item_id: Number(id), visitors, discount: Number($("#disc").value) } });
          toast(`Đã gửi ưu đãi cho ${visitors.length} khách.`); location.hash = "#/ban-hang/chien-dich";
        } catch (error) { toast(error.message); }
      });
    } catch (error) { $("#aud").innerHTML = errorState(error); }
  };
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = form.q.value.trim();
    if (!q) return;
    $("#aud").innerHTML = "";
    $("#found").innerHTML = `<div class="skeleton" style="height:120px"></div>`;
    try {
      const items = await api(`/shop/items?limit=8&q=${encodeURIComponent(q)}`);
      if (items.length === 1) return showAudience(items[0].item_id);
      $("#found").innerHTML = items.length ? `<div class="table-wrap" style="margin-bottom:var(--space-5)"><table><tbody>${items.map((i) => `<tr><td><a href="#/ban-hang/du-bao/${i.item_id}">${esc(i.name)}</a></td><td class="muted num">${i.item_id}</td><td class="r">${count(i.n_txn)} lượt mua</td></tr>`).join("")}</tbody></table></div>` : emptyState("Không có sản phẩm phù hợp.");
    } catch (error) { $("#found").innerHTML = errorState(error); }
  });
  if (itemId) showAudience(itemId);
}

async function sellerCampaigns(pane) {
  try {
    const rows = await api("/seller/campaigns");
    pane.innerHTML = rows.length ? `<div class="note">Số dưới đây đếm từ thao tác thật trong cửa hàng demo: khách đã thấy khi vào trang chủ, đã mở sản phẩm, đã đặt hàng với ưu đãi.</div>
    <div class="table-wrap"><table><thead><tr><th>Sản phẩm</th><th class="r">Giảm</th><th class="r">Đã gửi</th><th class="r">Đã thấy</th><th class="r">Đã mở</th><th class="r">Đã mua</th></tr></thead><tbody>${rows.map((c) => `<tr><td>${esc(c.item.name)}</td><td class="r">${c.discount}%</td><td class="r">${c.n}</td><td class="r">${c.seen}</td><td class="r">${c.clicked}</td><td class="r">${c.bought}</td></tr>`).join("")}</tbody></table></div>`
      : emptyState("Chưa có chiến dịch nào. Chọn một sản phẩm để xem nên gửi ưu đãi cho ai.", `<a class="btn primary" href="#/ban-hang/du-bao">Dự báo khách hàng</a>`);
  } catch (error) { pane.innerHTML = errorState(error); }
}

async function sellerReplay(pane) {
  const draw = (s) => {
    const running = s.status === "running";
    const hasData = s.done > 0;
    const rate = (a, b) => (b ? `${((a / b) * 100).toLocaleString("vi-VN", { maximumFractionDigits: 1 })}%` : "0%");
    pane.innerHTML = `<div class="note">Đẩy lại các sự kiện thật của giai đoạn kiểm thử theo đúng thứ tự thời gian, qua đúng đường ghi Neo4j rồi tính Top-K. Mỗi khi khách chạm vào một sản phẩm lần đầu, hệ thống kiểm tra sản phẩm đó đã có sẵn trong Top-20 của khách chưa. Xong sẽ tự xóa dữ liệu đã ghi. Số này khác cách đo ở Chương 4.</div>
      <div class="toolbar"><label class="small">Số sự kiện <select class="field" id="n"><option>100</option><option selected>300</option><option>1000</option></select></label>
      <button class="btn primary sm" id="go" ${running ? "disabled" : ""}>${ico("play")}${running ? "Đang phát lại" : "Bắt đầu phát lại"}</button></div>
      ${hasData ? `<div class="meter" role="progressbar" aria-valuemin="0" aria-valuemax="${s.total}" aria-valuenow="${s.done}"><i style="transform:scaleX(${s.done / (s.total || 1)})"></i></div>
      <div class="stats"><div class="stat"><b>${count(s.done)}/${count(s.total)}</b><small>sự kiện, ${s.visitors} khách</small></div>
      <div class="stat"><b>${Math.round(s.p50)} ms</b><small>độ trễ trung vị (p95 ${Math.round(s.p95)} ms)</small></div>
      <div class="stat"><b>${rate(s.hits, s.targets)}</b><small>có sẵn trong Top-20 của mô hình, ${s.hits}/${s.targets} lần chạm đầu</small></div>
      <div class="stat"><b>${rate(s.hits_popular, s.targets)}</b><small>có trong Top-20 bán chạy, ${s.hits_popular}/${s.targets} lần chạm đầu</small></div></div>
      ${s.status === "done" ? `<p class="small muted">Xong trong ${s.seconds} giây${s.cleaned ? ", đã dọn dữ liệu phát lại" : ""}.</p>` : ""}` : ""}
      ${s.status === "error" ? `<div class="state"><p>Dừng vì lỗi: ${esc(s.error)}</p></div>` : ""}`;
    $("#go").addEventListener("click", async () => {
      const state = await api(`/seller/replay?limit=${$("#n").value}`, { method: "POST" });
      draw(state);
      clearInterval(pollTimer);
      pollTimer = setInterval(async () => {
        const next = await api("/seller/replay");
        if (!pane.isConnected) return;
        draw(next);
        if (next.status !== "running" && next.cleaned) clearInterval(pollTimer);
      }, 800);
    });
  };
  try { draw(await api("/seller/replay")); } catch (error) { pane.innerHTML = errorState(error); }
}

async function sellerResults(pane) {
  try {
    const { rows, serving } = await api("/seller/results");
    const top = Math.max(...rows.map((r) => r["recall@20"]));
    const bars = (key) => rows.map((r) => `<div class="bar-row ${r.model === "bt_dkgrec_l05" ? "best" : ""}"><span>${esc(r.label)}</span><div class="bar-track"><div class="bar-fill" style="transform:scaleX(${r[key] / top})"></div></div><span class="num">${r[key].toFixed(4)}</span></div>`).join("");
    pane.innerHTML = `<h2 style="margin-bottom:var(--space-2)">Recall@20 trên tập kiểm thử, nhóm Active</h2>
      <p class="small muted" style="margin-bottom:var(--space-5)">Đọc trực tiếp từ các lần chạy chính thức, trung bình 3 seed, trùng số trong Chương 4.</p>
      <div class="bar-chart">${bars("recall@20")}</div>
      <div class="table-wrap" style="margin-top:var(--space-6)"><table><thead><tr><th>Mô hình</th><th class="r">Recall@20</th><th class="r">NDCG@20</th><th class="r">HitRate@20</th><th class="r">Coverage@20</th></tr></thead><tbody>${rows.map((r) => `<tr class="${r.model === "bt_dkgrec_l05" ? "best" : ""}"><td>${esc(r.label)}</td><td class="r">${r["recall@20"].toFixed(4)}</td><td class="r">${r["ndcg@20"].toFixed(4)}</td><td class="r">${r["hit_rate@20"].toFixed(4)}</td><td class="r">${r["coverage@20"].toFixed(4)}</td></tr>`).join("")}</tbody></table></div>
      ${serving ? `<div class="note" style="margin-top:var(--space-5)">Cửa hàng này chạy bản huấn luyện riêng trên CPU để lưu được trọng số, không nằm trong bảng trên: Recall@20 ${serving["recall@20"].toFixed(4)}, NDCG@20 ${serving["ndcg@20"].toFixed(4)}.</div>` : ""}`;
  } catch (error) { pane.innerHTML = errorState(error); }
}

function sellerIntegrate(pane) {
  const base = location.origin;
  pane.innerHTML = `<div class="two"><section><h2 style="margin-bottom:var(--space-3)">Gắn vào cửa hàng có sẵn</h2>
    <p class="small" style="margin-bottom:var(--space-4);max-width:65ch">Website hoặc ứng dụng nào cũng dùng được qua hai lời gọi: gửi hành vi khi khách thao tác, lấy danh sách gợi ý khi cần hiển thị.</p>
<pre>curl -X POST ${base}/api/v1/events \\
  -H "X-API-Key: KHOA_API_CUA_BAN" -H "Content-Type: application/json" \\
  -d '{"visitor_id": 962057, "item_id": 85791, "behavior": "addtocart"}'

curl ${base}/api/v1/recommendations/962057?k=10</pre>
    <a class="btn sm" href="/docs" target="_blank" rel="noopener">Mở tài liệu API</a></section>
    <section style="margin-top:0"><h2 style="margin-bottom:var(--space-3)">Đã có và chưa có</h2>
    <dl class="facts"><dt>Đã có</dt><dd>Thu hành vi, cập nhật đồ thị, gợi ý tức thời, khách mới, giải thích gợi ý, dự báo theo sản phẩm, ưu đãi, xóa dữ liệu khách</dd>
    <dt>Chưa có</dt><dd>Tự động huấn luyện lại, xác thực người dùng thật, thanh toán, chạy với toàn bộ 60.559 khách trên Neo4j</dd></dl></section></div>`;
}

// ── Router ──────────────────────────────────────────────────────────
async function route() {
  clearInterval(pollTimer);
  const [path, queryText] = (location.hash.slice(1) || "/").split("?");
  const query = new URLSearchParams(queryText || "");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  window.scrollTo(0, 0);
  if (parts[0] === "ban-hang") {
    renderBar();
    showBehind(null);
    const tab = parts[1] || "";
    const pane = sellerShell(tab);
    if (tab === "du-bao") return sellerForecast(pane, parts[2]);
    if (tab === "chien-dich") return sellerCampaigns(pane);
    if (tab === "phat-lai") return sellerReplay(pane);
    if (tab === "hieu-qua") return sellerResults(pane);
    if (tab === "tich-hop") return sellerIntegrate(pane);
    return sellerOverview(pane);
  }
  if (!session) { renderBar(); return pageEntry(); }
  await refreshCart().catch(() => {});
  renderBar(parts[0] === "nganh" ? parts[1] : "");
  if (parts[0] === "sp") return pageProduct(parts[1], query);
  if (parts[0] === "nganh") return pageList(parts[1], `limit=40&department=${encodeURIComponent(parts[1])}`, parts[1]);
  if (parts[0] === "tim") return pageList(`Kết quả cho "${parts[1] || ""}"`, `limit=40&q=${encodeURIComponent(parts[1] || "")}`);
  if (parts[0] === "gio") return pageCart();
  if (parts[0] === "tai-khoan") return pageAccount();
  return pageHome();
}

document.addEventListener("click", async (event) => {
  const who = event.target.closest("[data-who]");
  if (who) { session = { visitor_id: Number(who.dataset.who), name: who.dataset.name }; store.user = session; lastList = null; return route(); }
  if (event.target.closest("[data-guest]")) { session = await api("/shop/guest", { method: "POST" }); store.user = session; lastList = null; return route(); }
  const add = event.target.closest("[data-add]");
  const buy = event.target.closest("[data-buy]");
  if (add || buy) {
    const button = add || buy;
    button.disabled = true;
    try {
      const result = await api("/shop/cart", { method: "POST", body: { visitor_id: session.visitor_id, item_id: Number(button.dataset.add || button.dataset.buy) } });
      cartCount = result.cart.count;
      showBehind(result);
      if (buy) { location.hash = "#/gio"; return; }
      toast(`${BEHAVIOR_DONE.addtocart}. Kệ hàng được tính lại trong ${Math.round(result.total_ms)} ms.`);
      renderBar();
      if ((location.hash.slice(1) || "/") === "/") await pageHome();
    } catch (error) { toast(error.message); } finally { button.disabled = false; }
    return;
  }
  const remove = event.target.closest("[data-remove]");
  if (remove) {
    const cart = await api(`/shop/cart/${session.visitor_id}/${remove.dataset.remove}`, { method: "DELETE" });
    cartCount = cart.count; renderBar(); pageCart();
  }
});

(async () => {
  renderFoot();
  departments = await api("/shop/departments").catch(() => []);
  window.addEventListener("hashchange", route);
  route();
})();
