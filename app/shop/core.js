// Dùng chung cho ba cổng: khách hàng, cửa hàng, nhà cung cấp.
export const $ = (sel, root = document) => root.querySelector(sel);
export const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const money = (v) => `${Number(v).toLocaleString("vi-VN")}đ`;
export const count = (v) => Number(v).toLocaleString("vi-VN");
export const clock = (seconds) => new Date(seconds * 1000).toLocaleString("vi-VN");
export const dataTime = (ms) => new Date(ms).toLocaleString("vi-VN", { timeZone: "UTC" });
export const dataDate = (ms) => new Date(ms).toLocaleDateString("vi-VN", { timeZone: "UTC" });
export const ico = (name) => `<span class="ico" style="--i:url(/assets/icons/${name}.svg)" aria-hidden="true"></span>`;

export async function api(path, options = {}) {
  const response = await fetch(path, options.body
    ? { ...options, headers: { "content-type": "application/json" }, body: JSON.stringify(options.body) } : options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const failure = new Error(typeof error.detail === "string" ? error.detail : `Máy chủ trả lỗi ${response.status}`);
    failure.status = response.status;
    throw failure;
  }
  return response.json();
}

export function toast(text) {
  const el = $("#toast");
  el.textContent = text;
  el.classList.add("on");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove("on"), 3200);
}

export const tone = (item) => `--bg:var(--l${item.tone});--fg:var(--li${item.tone})`;
export const errorState = (error) => `<div class="state"><p>Chưa tải được: ${esc(error.message)}</p><button class="btn" onclick="location.reload()">Thử lại</button></div>`;
export const emptyState = (text, action = "") => `<div class="state"><p>${esc(text)}</p>${action}</div>`;
export const skeleton = (h = 280) => `<div class="skeleton" style="height:${h}px"></div>`;
export const skeletonGrid = (n = 6) => `<div class="grid" aria-busy="true">${`<div class="skeleton" style="aspect-ratio:1/1"></div>`.repeat(n)}</div>`;
export const mini = (item) => `<span class="mini" style="${tone(item)}">${ico(item.icon)}</span>`;

export function renderFoot() {
  $("#foot").innerHTML = `<div class="foot-in"><p>Bản trình diễn của đề án BT-DKGRec-GCN. Hành vi, danh mục và gợi ý là thật từ RetailRocket và mô hình; tên sản phẩm, tên khách và nhãn hàng là minh họa vì dữ liệu gốc đã ẩn danh. Đơn hàng không thu tiền.</p></div>`;
}

// Thanh trên cùng của cổng quản trị (cửa hàng, nhà cung cấp).
export function consoleShell(base, title, tabs, active) {
  $("#bar").innerHTML = `<div class="bar-in"><a class="mark" href="#/">${esc(title[0])}</a><span class="small muted">${esc(title[1])}</span>
    <nav class="bar-links" aria-label="Tài khoản"><span class="small muted">${esc(title[2] || "")}</span><a href="#/">${ico("arrow-left")}Đổi cổng</a><button class="linkish" id="logout">Đăng xuất</button></nav></div>`;
  $("#main").innerHTML = `<nav class="console-top" aria-label="${esc(title[1])}">${tabs.map(([k, v]) => `<a href="#/${base}${k ? "/" + k : ""}" ${k === active ? 'aria-current="page"' : ""}>${v}</a>`).join("")}</nav><div id="pane">${skeleton(320)}</div>`;
  return $("#pane");
}

// Đăng nhập: mỗi cổng giữ phiên riêng, nên có thể mở cả ba cổng cùng lúc.
export async function whoAmI(role) {
  try { return await api(`/auth/me?role=${role}`); } catch { return null; }
}
export function loginPage(role, heading, note, done) {
  $("#bar").innerHTML = `<div class="bar-in"><a class="mark" href="#/">BT-DKGRec</a><nav class="bar-links"><a href="#/">${ico("arrow-left")}Đổi cổng</a></nav></div>`;
  $("#main").innerHTML = `<div class="who"><div><h1>${esc(heading)}</h1><p class="muted" style="margin-top:var(--space-4)">${esc(note)}</p></div>
    <form class="login" id="login"><label>Tài khoản<input class="field" name="username" autocomplete="username" required></label>
    <label>Mật khẩu<input class="field" name="password" type="password" autocomplete="current-password" required></label>
    <p class="small" id="login-error" role="alert" style="color:var(--mark)"></p><button class="btn primary">Đăng nhập</button></form></div>`;
  const form = $("#login");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    $("#login-error").textContent = "";
    try {
      const user = await api("/auth/login", { method: "POST", body: { username: form.username.value, password: form.password.value, role } });
      done(user);
    } catch (error) { $("#login-error").textContent = error.message; }
  });
}
export function bindLogout(role, after) {
  const button = $("#logout");
  if (button) button.addEventListener("click", async () => { await api(`/auth/logout?role=${role}`, { method: "POST" }); after(); });
}
