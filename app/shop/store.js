// Cổng cửa hàng: doanh nghiệp xem danh mục, hành vi theo thời gian, khách nên chăm sóc.
import { $, api, esc, money, count, clock, dataTime, dataDate, ico, toast, errorState, emptyState, consoleShell, mini, skeleton, whoAmI, loginPage, bindLogout } from "./core.js";

const TABS = [["", "Tổng quan"], ["khach", "Khách ghé thăm"], ["danh-muc", "Danh mục và sản phẩm"], ["cham-soc", "Chăm sóc khách"], ["quan-tam", "Sản phẩm được quan tâm"]];
let timer = null;

async function overview(pane) {
  const draw = async () => {
    const [o, feed, carts] = await Promise.all([api("/seller/overview"), api("/seller/events?limit=12"), api("/seller/abandoned")]);
    if (!pane.isConnected) return;
    pane.innerHTML = `<div class="stats">
      <div class="stat"><b>${count(o.n_events)}</b><small>hành vi mới đã thu</small></div>
      <div class="stat"><b>${o.last_event ? new Date(o.last_event * 1000).toLocaleTimeString("vi-VN") : "Chưa có"}</b><small>${o.last_event ? `hành vi gần nhất, ngày ${new Date(o.last_event * 1000).toLocaleDateString("vi-VN")}` : "hành vi gần nhất"}</small></div>
      <div class="stat"><b>${count(o.n_orders)}</b><small>đơn hàng mô phỏng</small></div>
      <div class="stat"><b>${money(o.revenue)}</b><small>giá trị đơn mô phỏng</small></div></div>
      <div class="two"><section><h2 style="margin-bottom:var(--space-3)">Khách đang làm gì trong cửa hàng</h2>
      ${feed.length ? `<div class="table-wrap"><table><thead><tr><th>Khách</th><th>Hành vi</th><th>Sản phẩm</th><th class="r">Thời gian ghi nhận</th></tr></thead><tbody>${feed.map((e) => `<tr><td>${esc(e.name)}</td><td>${esc(e.behavior_vn)}</td><td>${esc(e.item_name)}</td><td class="r">${clock(e.created)}</td></tr>`).join("")}</tbody></table></div>`
        : emptyState("Chưa có hành vi mới. Mở cửa hàng và xem một sản phẩm, dòng đầu tiên sẽ hiện ở đây.", `<a class="btn" href="#/mua">Mở cửa hàng</a>`)}
      </section><section style="margin-top:0"><h2 style="margin-bottom:var(--space-3)">Khách bỏ giỏ</h2>
      ${carts.length ? `<div class="table-wrap"><table><tbody>${carts.slice(0, 8).map((c) => `<tr><td>${esc(c.name)} <span class="muted small">${c.visitor_id}</span></td><td><a href="#/cua-hang/cham-soc/${c.item.item_id}">${esc(c.item.name)}</a></td></tr>`).join("")}</tbody></table></div>` : `<p class="small muted">Không có.</p>`}</section></div>`;
  };
  try { await draw(); timer = setInterval(() => draw().catch(() => {}), 4000); } catch (error) { pane.innerHTML = errorState(error); }
}

function weekBars(weeks, key, label) {
  const top = Math.max(1, ...weeks.map((w) => w[key]));
  return `<figure class="weeks" aria-label="${esc(label)} theo tuần"><figcaption class="small muted">${esc(label)}</figcaption><div class="week-bars">${weeks.map((w) => `<i style="transform:scaleY(${Math.max(w[key] / top, 0.01)})" title="Tuần ${esc(w.week)}: ${count(w[key])}"></i>`).join("")}</div></figure>`;
}

async function customers(pane, visitor) {
  try {
    if (visitor) {
      const d = await api(`/seller/customers/${visitor}`);
      pane.innerHTML = `<p class="small"><a href="#/cua-hang/khach">Danh sách khách</a></p><div class="greet" style="margin-top:var(--space-3)"><h1>${esc(d.name)}</h1><span class="muted small">Mã khách ${d.visitor_id}, ${count(d.n_events)} hành vi</span></div>
      <div class="two wide"><section><h2 style="margin-bottom:var(--space-3)">Hành vi theo thời gian</h2><div class="table-wrap scroll"><table><thead><tr><th>Thời gian ghi nhận</th><th>Hành vi</th><th>Sản phẩm</th></tr></thead><tbody>${d.events.map((e) => `<tr class="${e.new ? "best" : ""}"><td class="num">${dataTime(e.timestamp)}</td><td>${esc(e.behavior)}</td><td>${esc(e.item)}</td></tr>`).join("")}</tbody></table></div></section>
      <section style="margin-top:0"><h2 style="margin-bottom:var(--space-3)">Nên giới thiệu cho khách này</h2><div class="table-wrap"><table><tbody>${d.suggestions.map((i) => `<tr><td>${mini(i)}<a href="#/cua-hang/cham-soc/${i.item_id}">${esc(i.name)}</a></td><td class="r">${money(i.price)}</td></tr>`).join("")}</tbody></table></div></section></div>`;
      return;
    }
    const rows = await api("/seller/customers");
    const draw = (q) => {
      const shown = rows.filter((c) => !q || c.name.toLowerCase().includes(q) || String(c.visitor_id).includes(q));
      $("#rows").innerHTML = shown.map((c) => `<tr><td><a href="#/cua-hang/khach/${c.visitor_id}">${esc(c.name)}</a> <span class="muted small">${c.visitor_id}</span></td><td class="num">${dataDate(c.last)}</td><td class="r">${count(c.n_events)}</td><td class="r">${count(c.n_items)}</td><td class="r">${count(c.n_cart)}</td><td class="r">${count(c.n_txn)}</td><td class="r">${money(c.spent)}</td></tr>`).join("");
    };
    pane.innerHTML = `<div class="note">${count(rows.length)} khách trong dữ liệu RetailRocket, lịch sử là hành vi thật. Tên khách và giá là minh họa.</div>
      <div class="toolbar"><input class="field" id="q" style="flex:1;min-width:220px;max-width:420px" placeholder="Tìm theo tên hoặc mã khách" aria-label="Tìm khách"></div>
      <div class="table-wrap"><table><thead><tr><th>Khách</th><th>Ghé gần nhất</th><th class="r">Hành vi</th><th class="r">Sản phẩm đã xem</th><th class="r">Thêm giỏ</th><th class="r">Lần mua</th><th class="r">Đã chi</th></tr></thead><tbody id="rows"></tbody></table></div>`;
    draw("");
    $("#q").addEventListener("input", (e) => draw(e.target.value.trim().toLowerCase()));
  } catch (error) { pane.innerHTML = errorState(error); }
}

function productForm(names, active, item) {
  return `<form class="toolbar" id="product" data-id="${item ? item.item_id : ""}"><h3 style="flex-basis:100%">${item ? `Sửa sản phẩm mã ${item.item_id}` : "Thêm sản phẩm"}</h3>
    <input class="field" name="name" required minlength="2" placeholder="Tên sản phẩm" aria-label="Tên sản phẩm" value="${item ? esc(item.name) : ""}" style="flex:2;min-width:200px">
    <select class="field" name="department" aria-label="Ngành hàng">${names.map((n) => `<option ${n === (item ? item.department : active) ? "selected" : ""}>${esc(n)}</option>`).join("")}</select>
    <input class="field" name="price" type="number" min="1000" step="1000" required placeholder="Giá" aria-label="Giá" value="${item ? item.price : ""}" style="width:130px">
    <label class="small"><input type="checkbox" name="available" ${!item || item.available ? "checked" : ""}> Còn hàng</label>
    <button class="btn primary sm">${item ? "Lưu" : "Thêm"}</button>${item ? `<button class="btn sm" type="button" id="cancel-edit">Hủy</button>` : ""}</form>`;
}

async function catalog(pane, department) {
  try {
    const rows = await api("/seller/departments");
    const active = department || rows[0].department;
    const names = rows.map((r) => r.department);
    pane.innerHTML = `<div class="two catalog-grid"><section><h2 style="margin-bottom:var(--space-3)">Ngành hàng</h2>
      <form class="toolbar" id="add-dept"><input class="field" name="name" required minlength="2" placeholder="Tên ngành hàng mới" aria-label="Tên ngành hàng mới" style="flex:1"><button class="btn sm">Thêm ngành</button></form>
      <div class="table-wrap"><table><thead><tr><th>Ngành</th><th class="r">Sản phẩm</th><th class="r">Lượt xem</th><th class="r">Giỏ</th><th class="r">Mua</th></tr></thead><tbody>
      ${rows.map((d) => `<tr class="${d.department === active ? "best" : ""}"><td><a href="#/cua-hang/danh-muc/${encodeURIComponent(d.department)}">${esc(d.department)}</a></td><td class="r">${count(d.n)}</td><td class="r">${count(d.n_view)}</td><td class="r">${count(d.n_cart)}</td><td class="r">${count(d.n_txn)}</td></tr>`).join("")}</tbody></table></div></section>
      <section style="margin-top:0" id="dept">${skeleton(360)}</section></div>`;
    const reload = (to = active) => { location.hash === `#/cua-hang/danh-muc/${encodeURIComponent(to)}` ? catalog(pane, to) : (location.hash = `#/cua-hang/danh-muc/${encodeURIComponent(to)}`); };
    const run = async (work, message, to) => { try { await work(); toast(message); reload(to); } catch (error) { toast(error.message); } };
    const addForm = $("#add-dept");
    addForm.addEventListener("submit", (e) => { e.preventDefault(); const name = addForm.name.value.trim(); run(() => api("/seller/department-names", { method: "POST", body: { name } }), `Đã thêm ngành ${name}.`, name); });
    const d = await api(`/seller/departments/${encodeURIComponent(active)}`);
    const draw = (editing) => {
      $("#dept").innerHTML = `<div class="sec-head"><h2>${esc(d.department)}</h2><span class="toolbar" style="margin:0"><button class="btn sm" id="rename">Đổi tên</button><button class="btn sm danger" id="drop">Xóa ngành</button></span></div>
      ${d.weeks.length ? `<div class="week-row">${weekBars(d.weeks, "view", "Lượt xem")}${weekBars(d.weeks, "cart", "Thêm vào giỏ")}${weekBars(d.weeks, "buy", "Mua")}</div><p class="small muted" style="margin:var(--space-2) 0 var(--space-5)">Mỗi cột là một tuần, từ ${esc(d.weeks[0].week)} đến ${esc(d.weeks[d.weeks.length - 1].week)}.</p>` : `<p class="small muted" style="margin-bottom:var(--space-4)">Chưa có hành vi.</p>`}
      ${productForm(names, active, editing)}
      <p class="small muted" style="margin-bottom:var(--space-3)">Sản phẩm mới có thể bán ngay nhưng chưa được gợi ý, vì chưa có trong đồ thị đã huấn luyện. Sản phẩm xóa sẽ ẩn khỏi cửa hàng và khỏi gợi ý.</p>
      ${d.items.length ? `<div class="table-wrap"><table><thead><tr><th>Sản phẩm</th><th title="Mã gốc RetailRocket">Mã</th><th class="r">Giá</th><th class="r">Mua</th><th>Còn</th><th></th></tr></thead><tbody>${d.items.map((i) => `<tr><td>${mini(i)}${esc(i.name)}${i.in_model ? "" : ` <span class="small" style="color:var(--mark)">Chờ huấn luyện</span>`}</td><td class="num">${i.in_model ? i.item_id : "Chưa có"}</td><td class="r">${money(i.price)}</td><td class="r">${count(i.n_txn)}</td><td>${i.available ? "Còn" : "Hết"}</td><td class="r nowrap"><button class="btn sm" data-edit="${i.item_id}">Sửa</button> <button class="btn sm danger" data-del="${i.item_id}" aria-label="Xóa ${esc(i.name)}">Xóa</button></td></tr>`).join("")}</tbody></table></div>` : emptyState("Ngành này chưa có sản phẩm.")}`;
      $("#rename").addEventListener("click", () => {
        const name = (prompt("Tên mới của ngành hàng", d.department) || "").trim();
        if (name && name !== d.department) run(() => api(`/seller/department-names/${encodeURIComponent(d.department)}`, { method: "PUT", body: { name } }), "Đã đổi tên ngành.", name);
      });
      $("#drop").addEventListener("click", () => { if (confirm(`Xóa ngành ${d.department}?`)) run(() => api(`/seller/department-names/${encodeURIComponent(d.department)}`, { method: "DELETE" }), "Đã xóa ngành.", names.find((n) => n !== d.department)); });
      const form = $("#product");
      form.addEventListener("submit", (e) => {
        e.preventDefault();
        const body = { name: form.name.value.trim(), department: form.department.value, price: Number(form.price.value), available: form.available.checked };
        const id = form.dataset.id;
        run(() => api(id ? `/seller/products/${id}` : "/seller/products", { method: id ? "PUT" : "POST", body }), id ? "Đã lưu sản phẩm." : "Đã thêm sản phẩm.", body.department);
      });
      const cancel = $("#cancel-edit");
      if (cancel) cancel.addEventListener("click", () => draw(null));
      $("#dept").querySelectorAll("[data-edit]").forEach((b) => b.addEventListener("click", () => { draw(d.items.find((i) => i.item_id === Number(b.dataset.edit))); $("#product").scrollIntoView({ block: "center" }); }));
      $("#dept").querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", () => { if (confirm("Xóa sản phẩm này khỏi cửa hàng?")) run(() => api(`/seller/products/${b.dataset.del}`, { method: "DELETE" }), "Đã xóa sản phẩm."); }));
    };
    draw(null);
  } catch (error) { pane.innerHTML = errorState(error); }
}

async function care(pane, itemId) {
  pane.innerHTML = `<div class="note">Chọn một sản phẩm để xem nên tiếp cận khách nào trước. Điểm dùng để xếp hạng, không phải xác suất mua.</div>
    <form class="toolbar" id="find"><input class="field" name="q" style="flex:1;min-width:220px;max-width:420px" placeholder="Tên hoặc mã sản phẩm" aria-label="Tìm sản phẩm" value="${itemId || ""}"><button class="btn sm">Tìm</button></form>
    <div id="found"></div><div id="aud"></div><section id="camps"></section>`;
  const form = $("#find");
  const showCampaigns = async () => {
    const rows = await api("/seller/campaigns");
    $("#camps").innerHTML = rows.length ? `<h2 style="margin-bottom:var(--space-3)">Ưu đãi đã gửi</h2><div class="table-wrap"><table><thead><tr><th>Sản phẩm</th><th class="r">Giảm</th><th class="r">Đã gửi</th><th class="r">Đã thấy</th><th class="r">Đã mở</th><th class="r">Đã mua</th></tr></thead><tbody>${rows.map((c) => `<tr><td>${esc(c.item.name)}</td><td class="r">${c.discount}%</td><td class="r">${c.n}</td><td class="r">${c.seen}</td><td class="r">${c.clicked}</td><td class="r">${c.bought}</td></tr>`).join("")}</tbody></table></div>` : "";
  };
  const showAudience = async (id) => {
    $("#aud").innerHTML = skeleton(240);
    try {
      const [item, rows] = await Promise.all([api(`/shop/items/${id}`), api(`/api/v1/items/${id}/audience?k=12`)]);
      $("#found").innerHTML = "";
      $("#aud").innerHTML = `<h2 style="margin-bottom:var(--space-3)">Khách nên tiếp cận cho ${esc(item.name)}</h2>
      <div class="table-wrap"><table><thead><tr><th></th><th class="r">Hạng</th><th>Khách</th><th class="r">Điểm</th><th class="r">Hành vi cùng ngành</th><th class="r">Lần thêm giỏ hoặc mua</th></tr></thead><tbody>
      ${rows.map((r) => `<tr><td><input type="checkbox" data-v="${r.visitor_id}" ${r.rank <= 5 ? "checked" : ""} aria-label="Chọn ${esc(r.name)}"></td><td class="r">${r.rank}</td><td>${esc(r.name)} <span class="muted small">${r.visitor_id}</span></td><td class="r">${r.score}</td><td class="r">${count(r.same_department)}</td><td class="r">${count(r.intent_events)}</td></tr>`).join("")}</tbody></table></div>
      <div class="toolbar" style="margin-top:var(--space-4)"><label class="small">Mức giảm <select class="field" id="disc"><option>5</option><option selected>10</option><option>15</option><option>20</option></select> %</label><button class="btn primary sm" id="send">${ico("send")}Gửi ưu đãi</button></div>`;
      $("#send").addEventListener("click", async () => {
        const visitors = [...document.querySelectorAll("[data-v]:checked")].map((c) => Number(c.dataset.v));
        try {
          await api("/seller/campaigns", { method: "POST", body: { item_id: Number(id), visitors, discount: Number($("#disc").value) } });
          toast(`Đã gửi ưu đãi cho ${visitors.length} khách.`); showCampaigns();
        } catch (error) { toast(error.message); }
      });
    } catch (error) { $("#aud").innerHTML = errorState(error); }
  };
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = form.q.value.trim();
    if (!q) return;
    $("#aud").innerHTML = "";
    $("#found").innerHTML = skeleton(120);
    try {
      const items = await api(`/shop/items?limit=8&q=${encodeURIComponent(q)}`);
      if (items.length === 1) { $("#found").innerHTML = ""; return showAudience(items[0].item_id); }
      $("#found").innerHTML = items.length ? `<div class="table-wrap" style="margin-bottom:var(--space-5)"><table><tbody>${items.map((i) => `<tr><td><a href="#/cua-hang/cham-soc/${i.item_id}">${esc(i.name)}</a></td><td class="muted num">${i.item_id}</td><td class="r">${count(i.n_txn)} lượt mua</td></tr>`).join("")}</tbody></table></div>` : emptyState("Không có sản phẩm phù hợp.");
    } catch (error) { $("#found").innerHTML = errorState(error); }
  });
  showCampaigns().catch(() => {});
  if (itemId) showAudience(itemId);
}

async function interest(pane) {
  try {
    const rows = await api("/seller/interest?limit=30");
    pane.innerHTML = `<div class="note">Đếm số khách đang có sản phẩm trong danh sách gợi ý, trên 234 khách của cửa hàng. Đây là tín hiệu để xem xét khi lên kế hoạch hàng, không phải dự báo số lượng. Khi đo lại, cách đếm này chưa tốt hơn danh sách bán chạy trong việc đoán nhu cầu thật.</div>
    <div class="table-wrap"><table><thead><tr><th>Sản phẩm</th><th>Ngành</th><th class="r">Số khách quan tâm</th><th class="r">Đã mua trước đây</th><th>Tình trạng</th></tr></thead><tbody>${rows.map((i) => `<tr><td>${mini(i)}${esc(i.name)}</td><td>${esc(i.department)}</td><td class="r">${count(i.n_customers)}</td><td class="r">${count(i.n_txn)}</td><td>${i.available ? "Còn hàng" : `<b style="color:var(--mark)">Đang hết, nên xem xét nhập</b>`}</td></tr>`).join("")}</tbody></table></div>`;
  } catch (error) { pane.innerHTML = errorState(error); }
}

export async function storeRoute(parts, route) {
  clearInterval(timer);
  const user = await whoAmI("store");
  if (!user) return loginPage("store", "Quản lý cửa hàng", "Đăng nhập bằng tài khoản cửa hàng.", () => route());
  const tab = parts[1] || "";
  const pane = consoleShell("cua-hang", ["Chợ Hiên", "Quản lý cửa hàng", user.display], TABS, tab);
  bindLogout("store", route);
  if (tab === "khach") return customers(pane, parts[2]);
  if (tab === "danh-muc") return catalog(pane, parts[2]);
  if (tab === "cham-soc") return care(pane, parts[2]);
  if (tab === "quan-tam") return interest(pane);
  return overview(pane);
}
export const stopStore = () => clearInterval(timer);
