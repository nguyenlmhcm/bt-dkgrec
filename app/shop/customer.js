// Cổng khách hàng: chỉ thấy cửa hàng và gợi ý của mình.
import { $, api, esc, money, dataDate, ico, toast, tone, errorState, emptyState, skeleton, skeletonGrid, mini, whoAmI, loginPage } from "./core.js";

let session = null;
let lastList = null;
let departments = [];
let cartCount = 0;

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
function whyText(why) {
  if (why.kind === "direct") return "Bạn đã từng quan tâm chính sản phẩm này.";
  if (why.kind === "customer") return "Những khách có hành vi gần giống bạn cũng quan tâm sản phẩm này.";
  if (why.kind === "category") return `Cùng nhóm hàng với ${why.seen} bạn từng quan tâm.`;
  if (why.kind === "property") return `Có đặc điểm giống ${why.seen} bạn từng quan tâm.`;
  return "Đang được nhiều khách quan tâm.";
}

function renderBar(active = "") {
  $("#bar").innerHTML = `<div class="bar-in">
    <a class="mark" href="#/mua">Chợ Hiên</a>
    <form class="search" role="search" id="search"><input name="q" type="search" placeholder="Tìm áo polo, nồi inox, sổ tay" aria-label="Tìm sản phẩm"><button aria-label="Tìm">${ico("search")}</button></form>
    <nav class="bar-links" aria-label="Tài khoản">
      <a href="#/">${ico("arrow-left")}Đổi cổng</a>
      <a href="#/tai-khoan">${ico("user")}${esc(session ? session.display : "Đăng nhập")}</a>
      <a href="#/gio" aria-label="Giỏ hàng, ${cartCount} sản phẩm">${ico("shopping-bag")}Giỏ${cartCount ? `<span class="badge">${cartCount}</span>` : ""}</a>
    </nav></div>
    <nav class="depts" aria-label="Ngành hàng"><div class="depts-in">${departments.filter((d) => d.dept_idx >= 0).slice(0, 13).map((d) =>
      `<a href="#/nganh/${encodeURIComponent(d.department)}" ${active === d.department ? 'aria-current="true"' : ""}>${ico(d.icon)}${esc(d.department)}</a>`).join("")}</div></nav>`;
  const form = $("#search");
  form.addEventListener("submit", (e) => { e.preventDefault(); location.hash = `#/tim/${encodeURIComponent(form.q.value.trim())}`; });
}

async function pageHome() {
  const main = $("#main");
  main.innerHTML = `<div class="greet"><h1>Chào ${esc(session.display)}</h1></div><div id="offers"></div><div id="shelf">${skeletonGrid()}</div><section id="also"></section>`;
  try {
    const [me, list] = await Promise.all([api(`/shop/me/${session.visitor_id}`), api(`/api/v1/recommendations/${session.visitor_id}?k=40`)]);
    $("#offers").innerHTML = me.offers.map((o) => `<div class="offer"><span><strong>Giảm ${o.discount}% riêng cho bạn</strong><br><span class="small">${esc(o.item.name)}, còn ${money(o.price_after)}</span></span><a class="btn primary sm" href="#/sp/${o.item.item_id}?uu-dai=${o.campaign_id}">Xem ưu đãi</a></div>`).join("");
    const inStock = list.items.filter((i) => i.available);
    const items = inStock.slice(0, 9);
    const prev = lastList && lastList.visitor === session.visitor_id ? lastList.ids : null;
    const personal = list.source === "personal";
    const lead = items[0];
    $("#shelf").innerHTML = lead ? `<div class="sec-head"><h2>${personal ? "Dành riêng cho bạn" : "Đang được mua nhiều"}</h2>
      <span class="small muted">${personal ? "Thay đổi theo những gì bạn xem và mua" : "Xem hoặc mua một món, kệ này sẽ đổi theo bạn"}</span></div>
      <div class="shelf">${items.map((item, i) => productCard(item, changeOf(prev, item.item_id, i), i === 0 ? "lead" : "")).join("")}</div>`
      : emptyState("Kệ hàng đang trống. Hãy tìm một sản phẩm để bắt đầu.");
    lastList = { visitor: session.visitor_id, ids: inStock.map((i) => i.item_id) };
    if (personal && lead) {
      const why = await api(`/api/v1/recommendations/${session.visitor_id}/why/${lead.item_id}`);
      const line = document.createElement("p");
      line.className = "why-line"; line.textContent = whyText(why);
      $("#shelf .lead").append(line);
      const also = await api(`/shop/also/${session.visitor_id}?skip=${items.map((i) => i.item_id).join(",")}`);
      if (also.length) $("#also").innerHTML = `<div class="sec-head"><h2>Khách giống bạn cũng quan tâm</h2></div><div class="grid">${also.map((i) => productCard(i)).join("")}</div>`;
    }
  } catch (error) { $("#shelf").innerHTML = errorState(error); }
}

async function pageProduct(id, query) {
  const main = $("#main");
  main.innerHTML = `<div class="pdp"><div class="skeleton" style="aspect-ratio:4/3"></div>${skeleton(320)}</div>`;
  try {
    const item = await api(`/shop/items/${id}`);
    const campaign = Number(query.get("uu-dai")) || null;
    const [me, why] = await Promise.all([api(`/shop/me/${session.visitor_id}`), api(`/api/v1/recommendations/${session.visitor_id}/why/${id}`)]);
    const offer = me.offers.find((o) => o.item.item_id === item.item_id);
    const together = item.bought_with.slice(0, 6);
    const near = item.similar.filter((i) => i.available).slice(0, 6);
    main.innerHTML = `<div class="pdp">${label(item, false)}
      <div><p class="small muted"><a href="#/nganh/${encodeURIComponent(item.department)}">${esc(item.department)}</a></p>
      <h1 style="margin-top:var(--space-2)">${esc(item.name)}</h1>
      <p class="pdp-price num">${offer ? `${money(offer.price_after)} <s class="small muted" style="font-weight:400">${money(item.price)}</s>` : money(item.price)}</p>
      <div class="pdp-actions">${item.available
        ? `<button class="btn primary" data-add="${item.item_id}">${ico("shopping-bag")}Thêm vào giỏ</button><button class="btn" data-buy="${item.item_id}">Mua ngay</button>`
        : `<button class="btn" disabled>Tạm hết hàng</button>`}</div>
      ${me.n_events ? `<div class="why"><h3>Vì sao bạn thấy sản phẩm này</h3><p class="small" style="margin-top:var(--space-1)">${esc(whyText(why))}</p></div>` : ""}</div></div>
      ${together.length ? `<section><div class="sec-head"><h2>Thường mua kèm</h2><span class="small muted">Từ những khách cũng thêm vào giỏ hoặc mua sản phẩm này</span></div><div class="grid">${together.map((i) => productCard(i)).join("")}</div></section>` : ""}
      ${near.length ? `<section><div class="sec-head"><h2>Sản phẩm gần giống</h2></div><div class="grid">${near.map((i) => productCard(i)).join("")}</div></section>` : ""}`;
    await api("/shop/visit", { method: "POST", body: { visitor_id: session.visitor_id, item_id: item.item_id, campaign_id: campaign } });
  } catch (error) { main.innerHTML = errorState(error); }
}

async function pageList(title, params, active = "") {
  const main = $("#main");
  renderBar(active);
  main.innerHTML = `<div class="greet"><h1>${esc(title)}</h1></div><div id="list">${skeletonGrid(12)}</div>`;
  try {
    const items = await api(`/shop/items?${params}`);
    $("#list").innerHTML = items.length ? `<div class="grid">${items.map((i) => productCard(i)).join("")}</div>`
      : emptyState("Không tìm thấy sản phẩm phù hợp.", `<a class="btn" href="#/mua">Về trang chủ</a>`);
  } catch (error) { $("#list").innerHTML = errorState(error); }
}

async function pageCart() {
  const main = $("#main");
  main.innerHTML = `<h1 style="margin-bottom:var(--space-5)">Giỏ hàng</h1><div id="cart">${skeleton(240)}</div>`;
  try {
    const cart = await api(`/shop/cart/${session.visitor_id}`);
    $("#cart").innerHTML = cart.lines.length ? `<div class="split"><div class="table-wrap"><table><thead><tr><th>Sản phẩm</th><th class="r">Đơn giá</th><th class="r">SL</th><th></th></tr></thead><tbody>
      ${cart.lines.map((l) => `<tr><td>${mini(l.item)}<a href="#/sp/${l.item.item_id}">${esc(l.item.name)}</a>${l.discount ? ` <span class="small" style="color:var(--mark)">Giảm ${l.discount}%</span>` : ""}</td><td class="r">${money(l.price)}</td><td class="r">${l.qty}</td><td class="r"><button class="btn sm" data-remove="${l.item.item_id}" aria-label="Bỏ ${esc(l.item.name)}">${ico("x")}</button></td></tr>`).join("")}
      </tbody></table></div><aside class="summary"><div class="total"><span>Tổng</span><span class="num">${money(cart.total)}</span></div>
      <p class="small muted">Thanh toán khi nhận hàng. Đơn này là mô phỏng, không thu tiền.</p><button class="btn primary" id="order">Đặt hàng</button></aside></div>`
      : emptyState("Giỏ hàng đang trống.", `<a class="btn primary" href="#/mua">Về trang chủ</a>`);
    const order = $("#order");
    if (order) order.addEventListener("click", placeOrder);
  } catch (error) { $("#cart").innerHTML = errorState(error); }
}

async function placeOrder() {
  $("#order").disabled = true;
  try {
    const result = await api("/shop/checkout", { method: "POST", body: { visitor_id: session.visitor_id } });
    cartCount = 0;
    toast(`Đã đặt đơn #${result.order_id}. Gợi ý của bạn vừa được cập nhật.`);
    location.hash = "#/mua";
  } catch (error) { toast(error.message); $("#order").disabled = false; }
}

async function pageAccount(route) {
  const main = $("#main");
  main.innerHTML = `<div id="acc">${skeleton(300)}</div>`;
  try {
    const me = await api(`/shop/me/${session.visitor_id}`);
    $("#acc").innerHTML = `<div class="greet"><h1>${esc(me.name)}</h1><button class="btn sm" id="out">Đăng xuất</button></div>
      <div class="two"><section><h2 style="margin-bottom:var(--space-3)">Sản phẩm bạn đã xem và mua</h2>
      ${me.timeline.length ? `<div class="table-wrap scroll"><table><thead><tr><th>Ngày</th><th>Sản phẩm</th><th>Bạn đã</th></tr></thead><tbody>${me.timeline.map((t) => `<tr><td class="num">${t.demo ? `<span style="color:var(--accent-2)">Vừa rồi</span>` : dataDate(t.timestamp)}</td><td><a href="#/sp/${t.item.item_id}">${esc(t.item.name)}</a></td><td>${esc(t.behavior_vn)}</td></tr>`).join("")}</tbody></table></div>` : emptyState("Bạn chưa xem sản phẩm nào.")}
      </section><section style="margin-top:0"><h2 style="margin-bottom:var(--space-3)">Đơn hàng</h2>
      ${me.orders.map((o) => `<p style="margin-bottom:var(--space-3)"><b>Đơn #${o.id}</b>, ${money(o.total)}<br><span class="small muted">${o.lines.filter((l) => l.item).map((l) => esc(l.item.name)).join(", ")}</span></p>`).join("")}
      ${me.past_purchases.length ? `<div class="table-wrap scroll" style="max-height:260px"><table><thead><tr><th>Đã mua ngày</th><th>Sản phẩm</th><th class="r">Giá</th></tr></thead><tbody>${me.past_purchases.map((p) => `<tr><td class="num">${dataDate(p.timestamp)}</td><td><a href="#/sp/${p.item.item_id}">${esc(p.item.name)}</a></td><td class="r">${money(p.item.price)}</td></tr>`).join("")}</tbody></table></div>` : ""}
      ${me.orders.length || me.past_purchases.length ? "" : `<p class="small muted">Chưa có đơn nào.</p>`}
      <h2 style="margin:var(--space-6) 0 var(--space-3)">Dữ liệu của bạn</h2>
      <p class="small muted" style="margin-bottom:var(--space-3)">Xóa những gì bạn vừa làm trong phiên này: hoạt động mới, giỏ hàng, đơn hàng mới.</p>
      <button class="btn danger sm" id="forget" ${me.n_new || me.orders.length ? "" : "disabled"}>${ico("trash-2")}Xóa dữ liệu của tôi</button></section></div>`;
    $("#out").addEventListener("click", async () => { await api("/auth/logout?role=customer", { method: "POST" }); session = null; lastList = null; cartCount = 0; route(); });
    $("#forget").addEventListener("click", async () => {
      if (!confirm("Xóa hoạt động và đơn hàng của bạn trong cửa hàng?")) return;
      await api(`/shop/me/${session.visitor_id}`, { method: "DELETE" });
      lastList = null; cartCount = 0; toast("Đã xóa dữ liệu của bạn."); route();
    });
  } catch (error) { $("#acc").innerHTML = errorState(error); }
}

export async function customerRoute(parts, query, route) {
  if (!departments.length) departments = await api("/shop/departments").catch(() => []);
  session = await whoAmI("customer");
  if (!session) return loginPage("customer", "Chợ gần nhà, gợi ý đúng người.", "Đăng nhập để xem gợi ý dành riêng cho bạn.", (user) => { session = user; lastList = null; route(); });
  const cart = await api(`/shop/cart/${session.visitor_id}`).catch(() => ({ count: 0 }));
  cartCount = cart.count;
  renderBar(parts[0] === "nganh" ? parts[1] : "");
  if (parts[0] === "sp") return pageProduct(parts[1], query);
  if (parts[0] === "nganh") return pageList(parts[1], `limit=40&department=${encodeURIComponent(parts[1])}`, parts[1]);
  if (parts[0] === "tim") return pageList(`Kết quả cho "${parts[1] || ""}"`, `limit=40&q=${encodeURIComponent(parts[1] || "")}`);
  if (parts[0] === "gio") return pageCart();
  if (parts[0] === "tai-khoan") return pageAccount(route);
  return pageHome();
}

export function customerClicks(route) {
  document.addEventListener("click", async (event) => {
    const add = event.target.closest("[data-add]");
    const buy = event.target.closest("[data-buy]");
    if (add || buy) {
      const button = add || buy;
      button.disabled = true;
      try {
        const result = await api("/shop/cart", { method: "POST", body: { visitor_id: session.visitor_id, item_id: Number(button.dataset.add || button.dataset.buy) } });
        cartCount = result.cart.count;
        if (buy) { location.hash = "#/gio"; return; }
        toast("Đã thêm vào giỏ. Gợi ý của bạn vừa được cập nhật.");
        renderBar();
        if (location.hash === "#/mua") await pageHome();
      } catch (error) { toast(error.message); } finally { button.disabled = false; }
      return;
    }
    const remove = event.target.closest("[data-remove]");
    if (remove) {
      const cart = await api(`/shop/cart/${session.visitor_id}/${remove.dataset.remove}`, { method: "DELETE" });
      cartCount = cart.count; renderBar(); pageCart();
    }
  });
}
