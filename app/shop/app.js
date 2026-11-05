// Bộ định tuyến ba cổng: khách hàng, cửa hàng, nhà cung cấp.
import { $, renderFoot, ico } from "./core.js";
import { customerRoute, customerClicks } from "./customer.js";
import { storeRoute, stopStore } from "./store.js";
import { ownerRoute, stopOwner } from "./owner.js";

function landing() {
  $("#bar").innerHTML = `<div class="bar-in"><a class="mark" href="#/">BT-DKGRec</a><span class="small muted">Giải pháp gợi ý dựa trên đồ thị tri thức động</span></div>`;
  $("#main").innerHTML = `<div class="doors"><a href="#/mua"><span>${ico("shopping-bag")}</span><b>Khách hàng</b><p>Người mua ghé cửa hàng và nhận gợi ý riêng.</p></a>
    <a href="#/cua-hang"><span>${ico("store")}</span><b>Cửa hàng</b><p>Doanh nghiệp xem danh mục, hành vi khách và nên chăm sóc ai.</p></a>
    <a href="#/nha-cung-cap"><span>${ico("activity")}</span><b>Nhà cung cấp</b><p>Chúng ta vận hành đồ thị tri thức động và mô hình.</p></a></div>`;
}

async function route() {
  stopStore(); stopOwner();
  const [path, queryText] = (location.hash.slice(1) || "/").split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  window.scrollTo(0, 0);
  if (!parts.length) return landing();
  if (parts[0] === "cua-hang") return storeRoute(parts, route);
  if (parts[0] === "nha-cung-cap") return ownerRoute(parts, route);
  return customerRoute(parts, new URLSearchParams(queryText || ""), route);
}

renderFoot();
customerClicks(route);
window.addEventListener("hashchange", route);
route();
