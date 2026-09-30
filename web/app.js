"use strict";
/* UI Asset Extractor – frontend (vanilla JS, no build step) */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(method, url, body) {
  const opt = { method, headers: {} };
  if (body instanceof FormData) opt.body = body;
  else if (body !== undefined) {
    opt.headers["content-type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(url, opt);
  let data = null;
  try { data = await r.json(); } catch { /* empty */ }
  if (!r.ok) throw new Error((data && data.detail) || `HTTP ${r.status}`);
  return data;
}

let toastT;
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast show" + (err ? " err" : "");
  clearTimeout(toastT);
  toastT = setTimeout(() => (t.className = "toast"), err ? 6000 : 3000);
}

const TYPE_COLORS = {
  group: "#8b93a3", icon: "#5b8cff", button: "#3ecf8e", panel: "#b07cff", label_plate: "#f5a524",
  badge: "#f25f5c", frame: "#00c2d1", bar: "#e6c300", currency: "#ffb86b", avatar: "#ff79c6",
  decoration: "#9ad36a", background: "#666", text: "#aaa", other: "#cfcfcf",
};
const TYPES = Object.keys(TYPE_COLORS);
const JOB_NAMES = { detect: "Detect", analyze: "Phân tích AI", extract: "Trích xuất", qa: "QA",
  regen: "Vẽ lại GPT", export: "Export", auto: "Tự động" };

/* ---------------- header / keys ---------------- */
let SETTINGS = null;
async function loadSettings() {
  SETTINGS = await api("GET", "/api/settings");
  const k = SETTINGS.keys;
  const g = SETTINGS.gpu;
  $("#keyStatus").innerHTML =
    (SETTINGS.settings.ai_provider === "cowork"
      ? `<span class="pill ok" title="Claude chạy qua app Cowork, không cần API key">Claude: Cowork</span>`
      : `<span class="pill ${k.anthropic ? "ok" : "bad"}">Claude ${k.anthropic ? "✓" : "✗"}</span>`) +
    `<span class="pill ${k.openai ? "ok" : "bad"}">OpenAI ${k.openai ? "✓" : "✗"}</span>` +
    `<span class="pill ${g.cuda_available && SETTINGS.settings.gpu_mode !== "off" ? "ok" : ""}">` +
    `${g.cuda_available ? "GPU " + SETTINGS.settings.gpu_mode : "CPU"}</span>`;
  return SETTINGS;
}

const claudeReady = () => SETTINGS.settings.ai_provider === "cowork" || !!SETTINGS.keys.anthropic;

/* ---------------- jobs ---------------- */
let currentJob = null;
let onJobDone = null;
async function startJob(pid, kind, params = {}) {
  if (currentJob) return toast("Đang có tác vụ chạy, đợi xong nhé.", true);
  try {
    const j = await api("POST", `/api/projects/${pid}/jobs`, { kind, params });
    trackJob(j);
  } catch (e) { toast(e.message, true); }
}
function trackJob(j) {
  currentJob = j;
  $("#jobbar").classList.remove("hidden");
  $("#jobCancel").onclick = () => api("POST", `/api/jobs/${j.id}/cancel`);
  poll();
}
async function poll() {
  if (!currentJob) return;
  let j;
  try { j = await api("GET", `/api/jobs/${currentJob.id}`); } catch { j = currentJob; }
  $("#jobKind").textContent = JOB_NAMES[j.kind] || j.kind;
  $("#jobMsg").textContent = j.message || "";
  $("#jobProg").style.width = Math.round((j.progress || 0) * 100) + "%";
  if (["done", "error", "cancelled"].includes(j.status)) {
    currentJob = null;
    setTimeout(() => $("#jobbar").classList.add("hidden"), j.status === "done" ? 800 : 0);
    if (j.status === "error") toast(j.error || "Lỗi", true);
    else if (j.status === "done") toast(j.message || "Xong");
    const warn = (j.log || []).filter((l) => !l.startsWith("=="));
    if (warn.length && j.status === "done") console.warn(warn);
    if (onJobDone) onJobDone(j);
    return;
  }
  setTimeout(poll, 700);
}

/* ---------------- router ---------------- */
async function route() {
  const h = location.hash || "#/";
  $$(".top nav a").forEach((a) => a.classList.toggle("on",
    (h.startsWith("#/settings") && a.dataset.nav === "settings") ||
    (!h.startsWith("#/settings") && a.dataset.nav === "home")));
  onJobDone = null;
  document.onkeydown = null;
  if (h.startsWith("#/settings")) return renderSettings();
  const m = h.match(/^#\/p\/(\w+)/);
  if (m) return renderProject(m[1]);
  return renderHome();
}
window.addEventListener("hashchange", route);

/* ---------------- home ---------------- */
async function renderHome() {
  const v = $("#view");
  v.innerHTML = `<div class="page">
    <h2>Projects</h2>
    <div class="drop" id="drop">Kéo thả ảnh UI (PNG/JPG) vào đây hoặc <b>bấm để chọn</b>
      <input type="file" id="file" accept="image/*" multiple hidden></div>
    <div class="grid" id="plist"></div></div>`;
  const drop = $("#drop"), file = $("#file");
  drop.onclick = () => file.click();
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); upload(e.dataTransfer.files); };
  file.onchange = () => upload(file.files);
  const list = await api("GET", "/api/projects");
  $("#plist").innerHTML = list.length ? list.map((p) => `
    <div class="pcard" data-id="${p.id}">
      <div class="th" style="background-image:url('/files/${p.id}/thumb.png')"></div>
      <div class="meta"><b>${esc(p.name)}</b>
        <span class="muted small">${p.width}×${p.height} · ${p.assets} asset · ${esc(p.stage)}</span>
        <button class="btn sm ghost danger" data-del="${p.id}" style="float:right">Xoá</button>
      </div></div>`).join("") : `<p class="muted">Chưa có project nào.</p>`;
  $$(".pcard").forEach((c) => (c.onclick = (e) => {
    if (e.target.dataset.del) return;
    location.hash = `#/p/${c.dataset.id}`;
  }));
  $$("[data-del]").forEach((b) => (b.onclick = async () => {
    if (!confirm("Xoá project này?")) return;
    await api("DELETE", `/api/projects/${b.dataset.del}`);
    renderHome();
  }));
}
async function upload(files) {
  let last;
  for (const f of files) {
    const fd = new FormData();
    fd.append("file", f);
    fd.append("name", f.name.replace(/\.[^.]+$/, ""));
    try { last = await api("POST", "/api/projects", fd); } catch (e) { toast(e.message, true); }
  }
  if (last) location.hash = files.length === 1 ? `#/p/${last.id}` : "#/";
  if (files.length > 1) renderHome();
}

/* ---------------- settings ---------------- */
async function renderSettings() {
  const S = await loadSettings();
  const s = S.settings;
  const opt = (arr, cur) => arr.map((o) => {
    const [val, lab] = Array.isArray(o) ? o : [o, o];
    return `<option value="${esc(val)}" ${String(val) === String(cur) ? "selected" : ""}>${esc(lab)}</option>`;
  }).join("");
  const keyCard = (name, label) => `
    <div class="card">
      <h3>${label} API key</h3>
      <p class="small muted">${S.keys[name] ? `Đã lưu: <code>${esc(S.keys[name])}</code>` : "Chưa có key."}
        Key được lưu trong Windows Credential Manager, không lưu ra file.</p>
      <div class="keyrow">
        <a class="btn" href="${S.key_links[name]}" target="_blank" rel="noopener">1. Lấy key ↗</a>
        <input type="password" id="k_${name}" placeholder="2. Dán key vào đây">
        <button class="btn primary" data-save="${name}">3. Kiểm tra & lưu</button>
        ${S.keys[name] ? `<button class="btn" data-test="${name}">Test</button>
          <button class="btn danger" data-rm="${name}">Xoá</button>` : ""}
      </div>
      <div class="small muted" id="kinfo_${name}"></div>
    </div>`;
  $("#view").innerHTML = `<div class="page">
    <h2>Settings</h2>
    ${keyCard("anthropic", "Claude (Anthropic)")}
    ${keyCard("openai", "OpenAI (ChatGPT)")}
    <div class="card" id="cfg">
      <h3>Claude</h3>
      <label class="row"><span>Cách dùng Claude</span>
        <select data-k="ai_provider">${opt([["cowork", "Claude app (Cowork) – không cần API key"], ["api", "Claude API key"]], s.ai_provider)}</select></label>
      <p class="small muted">Chế độ Cowork: khi phân tích/QA, tool tạo 1 task trong <code>data/cowork_tasks</code> và chờ.
        Mở Claude (Cowork) có kết nối thư mục tool rồi nhắn <b>“Làm task UIAssetExtractor”</b> – Claude đọc ảnh, ghi kết quả, tool tự chạy tiếp.</p>
      <h3>Model</h3>
      <label class="row"><span>Claude model</span>
        <input list="cm" data-k="claude_model" value="${esc(s.claude_model)}"><datalist id="cm"></datalist></label>
      <label class="row"><span>OpenAI image model</span>
        <input list="om" data-k="openai_image_model" value="${esc(s.openai_image_model)}"><datalist id="om"></datalist></label>
      <label class="row"><span>Chất lượng ảnh GPT</span>
        <select data-k="image_quality">${opt(["low", "medium", "high"], s.image_quality)}</select></label>
      <h3>GPU / xử lý local</h3>
      <label class="row"><span>Chế độ GPU</span>
        <select data-k="gpu_mode">${opt([["off", "Tắt (chỉ CPU)"], ["eco", "Tiết kiệm (≤2GB VRAM)"], ["full", "Đầy đủ"]], s.gpu_mode)}</select></label>
      <p class="small muted">GPU: ${S.gpu.cuda_available ? "phát hiện CUDA ✓" : "không có CUDA – chạy CPU"}
        · ONNX Runtime ${esc(S.gpu.onnxruntime || "?")}</p>
      <label class="row"><span>Cách tách nền mặc định</span>
        <select data-k="matte_method">${opt([["grabcut", "GrabCut (nhanh, không cần GPU)"], ["ai", "AI (rembg)"], ["none", "Không tách"]], s.matte_method)}</select></label>
      <label class="row"><span>Model tách nền AI</span>
        <select data-k="ai_matte_model">${opt([["isnet-general-use", "ISNet (nhẹ ~180MB)"], ["birefnet-general-lite", "BiRefNet lite (~220MB)"], ["birefnet-general", "BiRefNet (~900MB, nặng)"], ["u2net", "U2Net"]], s.ai_matte_model)}</select></label>
      <h3>Chất lượng & chi phí</h3>
      <label class="row"><span>Ngưỡng QA (0-10)</span><input type="number" min="0" max="10" data-k="qa_threshold" value="${s.qa_threshold}"></label>
      <label class="row"><span>Tự QA sau khi vẽ lại</span><select data-k="auto_qa_after_regen">${opt([["true", "Có"], ["false", "Không"]], s.auto_qa_after_regen)}</select></label>
      <label class="row"><span>Số lần tự vẽ lại nếu QA thấp</span><input type="number" min="0" max="3" data-k="max_regen_retries" value="${s.max_regen_retries}"></label>
      <label class="row"><span>Giới hạn ảnh GPT / project</span><input type="number" min="0" data-k="max_images_per_project" value="${s.max_images_per_project}"></label>
      <label class="row"><span>Số request Claude song song</span><input type="number" min="1" max="8" data-k="claude_concurrency" value="${s.claude_concurrency}"></label>
      <h3>Export</h3>
      <label class="row"><span>Scale mặc định</span><select data-k="export_scale">${opt([1, 2, 3, 4], s.export_scale)}</select></label>
      <label class="row"><span>Kích thước atlas tối đa</span><select data-k="atlas_max_size">${opt([1024, 2048, 4096], s.atlas_max_size)}</select></label>
      <label class="row"><span>Padding (px)</span><input type="number" min="0" max="16" data-k="atlas_padding" value="${s.atlas_padding}"></label>
      <div class="actions"><button class="btn primary" id="saveCfg">Lưu cài đặt</button></div>
    </div></div>`;

  $$("[data-save]").forEach((b) => (b.onclick = async () => {
    const n = b.dataset.save, key = $(`#k_${n}`).value.trim();
    if (!key) return toast("Dán key trước đã", true);
    b.disabled = true; b.textContent = "Đang kiểm tra…";
    try {
      const r = await api("POST", `/api/keys/${n}`, { key });
      toast(`Đã lưu key (${r.stored === "keyring" ? "Credential Manager" : "file"})`);
      if (r.note) alert(r.note);
      renderSettings();
    } catch (e) { toast(e.message, true); b.disabled = false; b.textContent = "3. Kiểm tra & lưu"; }
  }));
  $$("[data-test]").forEach((b) => (b.onclick = async () => {
    try {
      const r = await api("POST", `/api/keys/${b.dataset.test}/test`);
      $(`#kinfo_${b.dataset.test}`).textContent = `OK · ${r.models.length} model` + (r.note ? ` · ${r.note}` : "");
      fillModels(b.dataset.test, r.models);
    } catch (e) { toast(e.message, true); }
  }));
  $$("[data-rm]").forEach((b) => (b.onclick = async () => {
    if (!confirm("Xoá key này?")) return;
    await api("DELETE", `/api/keys/${b.dataset.rm}`);
    renderSettings();
  }));
  $("#saveCfg").onclick = async () => {
    const patch = {};
    $$("#cfg [data-k]").forEach((el) => {
      let v = el.value;
      if (el.type === "number") v = Number(v);
      if (v === "true") v = true; else if (v === "false") v = false;
      if (["export_scale", "atlas_max_size"].includes(el.dataset.k)) v = Number(v);
      patch[el.dataset.k] = v;
    });
    await api("PUT", "/api/settings", patch);
    await loadSettings();
    toast("Đã lưu cài đặt");
  };
  // populate model lists silently
  for (const n of ["anthropic", "openai"]) {
    if (!S.keys[n]) continue;
    api("POST", `/api/keys/${n}/test`).then((r) => fillModels(n, r.models)).catch(() => {});
  }
}
function fillModels(name, models) {
  const dl = $(name === "anthropic" ? "#cm" : "#om");
  if (dl) dl.innerHTML = models.map((m) => `<option value="${esc(m)}">`).join("");
}

/* ---------------- project workspace ---------------- */
const W = { proj: null, sel: null, checked: new Set(), zoom: 1, mode: "select", tab: "list",
  view: "canvas", filter: "all", q: "" };

function vurl(a, i) {
  const idx = i ?? a.active;
  if (idx == null || !a.versions[idx]) return null;
  const v = a.versions[idx];
  return `/files/${W.proj.id}/${v.file}?t=${Math.round(v.created)}`;
}
function scoreCls(q) {
  if (!q) return "";
  const th = SETTINGS?.settings.qa_threshold ?? 7;
  return q.score >= th ? "ok" : q.score >= th - 2 ? "mid" : "bad";
}

async function renderProject(pid) {
  if (!SETTINGS) await loadSettings();
  if (!W.proj || W.proj.id !== pid) {
    Object.assign(W, { sel: null, checked: new Set(), zoom: 0, mode: "select", tab: "list", view: "canvas" });
  }
  try { W.proj = await api("GET", `/api/projects/${pid}`); } catch (e) {
    $("#view").innerHTML = `<div class="page"><p>${esc(e.message)}</p></div>`;
    return;
  }
  $("#view").innerHTML = `
  <div class="ws">
    <div class="ws-left">
      <div class="toolbar">
        <button class="btn primary" data-job="auto" title="Detect → Phân tích AI → Trích xuất → QA">⚡ Tự động</button>
        <span class="sep"></span>
        <button class="btn" data-job="detect" title="Tìm vùng UI bằng OpenCV + OCR (miễn phí)">1 Detect</button>
        <button class="btn" data-job="analyze" title="Claude đặt tên & tách từng asset">2 Phân tích AI</button>
        <button class="btn" data-job="extract" title="Tách nền, xoá chữ, cắt (local)">3 Trích xuất</button>
        <button class="btn" data-job="qa" title="Claude chấm điểm từng asset">4 QA</button>
        <button class="btn" id="btnRegen" title="Vẽ lại asset đã tick bằng GPT">5 Vẽ lại GPT</button>
        <button class="btn" id="btnExport">6 Export</button>
        <span class="sep"></span>
        <button class="btn" id="btnDraw" title="Vẽ thêm box (phím B)">＋ Box</button>
        <button class="btn sm" id="zOut">−</button><button class="btn sm" id="zFit">Fit</button><button class="btn sm" id="zIn">＋</button>
        <button class="btn sm" id="btnView">${W.view === "canvas" ? "Lưới kết quả" : "Khung ảnh"}</button>
        <span class="usage small muted" id="usage"></span>
      </div>
      <div class="viewport" id="vp"></div>
    </div>
    <div class="ws-right">
      <div class="tabs">
        <button data-tab="list">Danh sách</button><button data-tab="detail">Chi tiết</button><button data-tab="export">Export</button>
      </div>
      <div class="pane" id="pane"></div>
    </div>
  </div>`;

  $$("[data-job]").forEach((b) => (b.onclick = () => runStep(b.dataset.job)));
  $("#btnRegen").onclick = () => regenSelected();
  $("#btnExport").onclick = () => { W.tab = "export"; renderPane(); };
  $("#btnDraw").onclick = () => setMode(W.mode === "draw" ? "select" : "draw");
  $("#zIn").onclick = () => setZoom(W.zoom * 1.25);
  $("#zOut").onclick = () => setZoom(W.zoom / 1.25);
  $("#zFit").onclick = () => fitZoom();
  $("#btnView").onclick = () => { W.view = W.view === "canvas" ? "grid" : "canvas"; renderProject(pid); };
  $$(".tabs button").forEach((b) => (b.onclick = () => { W.tab = b.dataset.tab; renderPane(); }));

  onJobDone = async (j) => { if (j.project === pid) await refresh(); };
  document.onkeydown = keys;
  renderUsage();
  renderLeft();
  renderPane();
  // resume a running job for this project
  if (!currentJob) {
    const js = await api("GET", `/api/jobs?project=${pid}`);
    const run = js.find((j) => j.status === "running" || j.status === "queued");
    if (run) trackJob(run);
  }
}

async function refresh() {
  W.proj = await api("GET", `/api/projects/${W.proj.id}`);
  if (W.sel && !W.proj.assets.find((a) => a.id === W.sel)) W.sel = null;
  W.checked = new Set([...W.checked].filter((id) => W.proj.assets.some((a) => a.id === id)));
  renderUsage();
  renderLeft();
  renderPane();
}

function renderUsage() {
  const u = W.proj.usage;
  $("#usage").textContent = `Claude: ${u.claude_calls} lần · ${fmtK(u.claude_in)} in / ${fmtK(u.claude_out)} out · GPT ảnh: ${u.images}`;
}
const fmtK = (n) => (n > 999 ? (n / 1000).toFixed(1) + "k" : n);

function runStep(kind) {
  const p = W.proj;
  const analyzed = p.assets.some((a) => a.type !== "group");
  if ((kind === "detect" || kind === "auto") && analyzed &&
      !confirm("Chạy lại từ đầu sẽ xoá danh sách asset hiện tại. Tiếp tục?")) return;
  if ((kind === "analyze" || kind === "auto" || kind === "qa") && !claudeReady())
    return toast("Cần Claude API key (Settings).", true);
  if (kind === "analyze" && analyzed &&
      !confirm("Phân tích lại sẽ thay thế asset hiện tại bằng kết quả mới từ các vùng Detect. Tiếp tục?")) return;
  if (kind === "analyze" && !p.assets.length && !(p.groups || []).length) return toast("Chạy Detect trước.", true);
  let params = {};
  if (kind === "extract" && W.checked.size) params.asset_ids = [...W.checked];
  if (kind === "qa" && W.checked.size) params.asset_ids = [...W.checked];
  startJob(p.id, kind, params);
}

function regenSelected(ids) {
  ids = ids || [...W.checked];
  if (!ids.length && W.sel) ids = [W.sel];
  if (!ids.length) return toast("Tick chọn asset cần vẽ lại (hoặc chọn 1 asset).", true);
  if (!SETTINGS.keys.openai) return toast("Cần OpenAI API key (Settings).", true);
  const left = SETTINGS.settings.max_images_per_project - W.proj.usage.images;
  if (!confirm(`Vẽ lại ${ids.length} asset bằng GPT (~${ids.length}–${ids.length * (1 + SETTINGS.settings.max_regen_retries)} ảnh). ` +
    `Còn ${left} ảnh trong giới hạn project. Tiếp tục?`)) return;
  startJob(W.proj.id, "regen", { asset_ids: ids });
}

function visibleAssets() {
  let a = W.proj.assets;
  const th = SETTINGS.settings.qa_threshold;
  if (W.filter === "keep") a = a.filter((x) => x.keep);
  if (W.filter === "low") a = a.filter((x) => x.qa && x.qa.score < th);
  if (W.filter === "dup") a = a.filter((x) => x.dup_of);
  if (W.filter === "err") a = a.filter((x) => x.status === "error");
  if (W.filter === "noqa") a = a.filter((x) => !x.qa && x.active != null);
  if (W.q) a = a.filter((x) => x.name.toLowerCase().includes(W.q.toLowerCase()));
  return a;
}

/* ---------- left: canvas or grid ---------- */
function renderLeft() {
  const vp = $("#vp");
  if (W.view === "grid") return renderGrid(vp);
  const p = W.proj;
  vp.classList.toggle("draw", W.mode === "draw");
  vp.innerHTML = `<div class="stage" id="stage" style="width:${p.width}px;height:${p.height}px">
    <img src="/files/${p.id}/source.png" width="${p.width}" height="${p.height}" draggable="false"></div>`;
  const st = $("#stage");
  const sorted = [...p.assets].filter((a) => a.type !== "background").sort((a, b) => b.bbox[2] * b.bbox[3] - a.bbox[2] * a.bbox[3]);
  for (const a of sorted) {
    const [x, y, w, h] = a.bbox;
    const c = TYPE_COLORS[a.type] || "#fff";
    const el = document.createElement("div");
    el.className = "box" + (a.id === W.sel ? " sel" : "") + (a.keep ? "" : " off");
    el.dataset.id = a.id;
    Object.assign(el.style, { left: x + "px", top: y + "px", width: w + "px", height: h + "px", borderColor: c });
    el.innerHTML = `<span class="tag" style="background:${c}">${esc(a.name)}</span><span class="h"></span>`;
    el.onmousedown = (e) => boxDown(e, a, el);
    st.appendChild(el);
  }
  vp.onmousedown = (e) => {
    if (W.mode === "draw") return drawDown(e);
    if (e.target === vp || e.target.tagName === "IMG") { W.sel = null; renderLeft(); renderPane(); }
  };
  vp.onwheel = (e) => { if (e.ctrlKey) { e.preventDefault(); setZoom(W.zoom * (e.deltaY < 0 ? 1.1 : 0.9)); } };
  if (!W.zoom) fitZoom(); else applyZoom();
}
function applyZoom() {
  const st = $("#stage");
  if (!st) return;
  st.style.transform = `scale(${W.zoom})`;
  st.style.setProperty("--inv", String(1 / W.zoom));
  st.style.marginBottom = `${W.proj.height * (W.zoom - 1) + 20}px`;
  st.style.marginRight = `${W.proj.width * (W.zoom - 1) + 20}px`;
}
function setZoom(z) { W.zoom = Math.min(8, Math.max(0.1, z)); applyZoom(); }
function fitZoom() {
  const vp = $("#vp");
  W.zoom = Math.min((vp.clientWidth - 40) / W.proj.width, (vp.clientHeight - 40) / W.proj.height, 4);
  applyZoom();
}
function toImg(e) {
  const r = $("#stage").getBoundingClientRect();
  return [(e.clientX - r.left) / W.zoom, (e.clientY - r.top) / W.zoom];
}
function boxDown(e, a, el) {
  if (W.mode === "draw") return;
  e.stopPropagation();
  e.preventDefault();
  if (W.sel !== a.id) {
    W.sel = a.id;
    $$(".box.sel").forEach((b) => b.classList.remove("sel"));
    el.classList.add("sel");
    if (W.tab !== "export") W.tab = "detail";
    renderPane();
  }
  const resize = e.target.classList.contains("h");
  const [sx, sy] = toImg(e);
  const b0 = [...a.bbox];
  let moved = false;
  const mm = (ev) => {
    const [cx, cy] = toImg(ev);
    const dx = Math.round(cx - sx), dy = Math.round(cy - sy);
    if (Math.abs(dx) + Math.abs(dy) > 1) moved = true;
    const nb = resize ? [b0[0], b0[1], Math.max(4, b0[2] + dx), Math.max(4, b0[3] + dy)]
      : [b0[0] + dx, b0[1] + dy, b0[2], b0[3]];
    a.bbox = nb;
    Object.assign(el.style, { left: nb[0] + "px", top: nb[1] + "px", width: nb[2] + "px", height: nb[3] + "px" });
  };
  const mu = async () => {
    window.removeEventListener("mousemove", mm);
    window.removeEventListener("mouseup", mu);
    if (moved) { await patchAsset(a.id, { bbox: a.bbox }); renderPane(); }
  };
  window.addEventListener("mousemove", mm);
  window.addEventListener("mouseup", mu);
}
function drawDown(e) {
  e.preventDefault();
  const [sx, sy] = toImg(e);
  const d = document.createElement("div");
  d.className = "drawing";
  $("#stage").appendChild(d);
  let b = [sx, sy, 0, 0];
  const mm = (ev) => {
    const [cx, cy] = toImg(ev);
    b = [Math.min(sx, cx), Math.min(sy, cy), Math.abs(cx - sx), Math.abs(cy - sy)];
    Object.assign(d.style, { left: b[0] + "px", top: b[1] + "px", width: b[2] + "px", height: b[3] + "px" });
  };
  const mu = async () => {
    window.removeEventListener("mousemove", mm);
    window.removeEventListener("mouseup", mu);
    d.remove();
    if (b[2] < 4 || b[3] < 4) return;
    const name = prompt("Tên asset:", "asset");
    if (name === null) return;
    const a = await api("POST", `/api/projects/${W.proj.id}/assets`,
      { bbox: b.map(Math.round), name: name || "asset", type: "other", z: 1 });
    W.sel = a.id;
    W.tab = "detail";
    setMode("select");
    await refresh();
  };
  window.addEventListener("mousemove", mm);
  window.addEventListener("mouseup", mu);
}
function setMode(m) {
  W.mode = m;
  $("#btnDraw").classList.toggle("primary", m === "draw");
  $("#vp").classList.toggle("draw", m === "draw");
}
function keys(e) {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (e.key === "Escape") setMode("select");
  if (e.key.toLowerCase() === "b") setMode(W.mode === "draw" ? "select" : "draw");
  if (e.key === "Delete" && W.sel) delAsset(W.sel);
}

function renderGrid(vp) {
  const items = visibleAssets().filter((a) => a.active != null);
  vp.classList.remove("draw");
  vp.onmousedown = null;
  vp.innerHTML = items.length ? `<div class="rgrid">${items.map((a) => `
    <div class="c ${a.id === W.sel ? "sel" : ""}" data-id="${a.id}">
      <div class="im"><img src="${vurl(a)}" loading="lazy"></div>
      <div class="cap"><span title="${esc(a.name)}">${a.keep ? "" : "⊘ "}${esc(a.name)}</span>
        ${a.qa ? `<span class="score ${scoreCls(a.qa)}">${a.qa.score}</span>` : ""}
        ${a.dup_of ? `<span class="score">dup</span>` : ""}</div>
    </div>`).join("")}</div>` : `<p class="muted" style="padding:20px">Chưa có asset đã trích xuất.</p>`;
  $$(".rgrid .c").forEach((c) => (c.onclick = () => { W.sel = c.dataset.id; W.tab = "detail"; renderGrid(vp); renderPane(); }));
}

/* ---------- right pane ---------- */
function renderPane() {
  $$(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === W.tab));
  const pane = $("#pane");
  if (!pane) return;
  if (W.tab === "detail") return renderDetail(pane);
  if (W.tab === "export") return renderExport(pane);
  return renderList(pane);
}

function renderList(pane) {
  const items = visibleAssets();
  const p = W.proj;
  pane.innerHTML = `
    ${!p.assets.length ? `<p class="muted">Bắt đầu: bấm <b>⚡ Tự động</b> (cần Claude key) hoặc <b>1 Detect</b> (miễn phí).</p>` : ""}
    <div class="filters">
      <select id="flt">${[["all", "Tất cả"], ["keep", "Được export"], ["low", "QA thấp"], ["noqa", "Chưa QA"], ["dup", "Trùng lặp"], ["err", "Lỗi"]]
        .map(([v, l]) => `<option value="${v}" ${W.filter === v ? "selected" : ""}>${l}</option>`).join("")}</select>
      <input id="q" placeholder="Tìm tên…" value="${esc(W.q)}">
    </div>
    <div class="bulk">
      <button class="btn sm" id="chkAll">${W.checked.size ? "Bỏ chọn" : "Chọn tất cả"}</button>
      <span class="small muted" style="align-self:center">${W.checked.size} đã chọn</span>
      ${W.checked.size ? `
        <button class="btn sm" data-bulk="keep1">Keep</button>
        <button class="btn sm" data-bulk="keep0">Bỏ keep</button>
        <button class="btn sm" data-bulk="extract">Trích xuất lại</button>
        <button class="btn sm" data-bulk="qa">QA</button>
        <button class="btn sm" data-bulk="regen">Vẽ lại GPT</button>
        <button class="btn sm danger" data-bulk="del">Xoá</button>` : ""}
    </div>
    <div class="alist">${items.map((a) => {
      const u = vurl(a);
      return `<div class="it ${a.id === W.sel ? "sel" : ""}" data-id="${a.id}">
        <input type="checkbox" ${W.checked.has(a.id) ? "checked" : ""} data-chk="${a.id}">
        ${u ? `<img src="${u}" loading="lazy">` : `<div style="width:40px;height:40px;border-radius:4px;background:${TYPE_COLORS[a.type]}33"></div>`}
        <div class="nm"><span style="color:${TYPE_COLORS[a.type]}">●</span> ${a.keep ? "" : "⊘ "}${esc(a.name)}
          <div class="small muted">${a.type}${a.dup_of ? " · trùng" : ""}${a.status === "error" ? " · lỗi" : ""}${a.status === "regenerated" ? " · GPT" : ""}</div></div>
        ${a.qa ? `<span class="score ${scoreCls(a.qa)}">${a.qa.score}</span>` : "<span></span>"}
      </div>`;
    }).join("")}</div>`;
  $("#flt").onchange = (e) => { W.filter = e.target.value; renderPane(); if (W.view === "grid") renderLeft(); };
  $("#q").oninput = (e) => { W.q = e.target.value; renderListOnly(); };
  $("#chkAll").onclick = () => {
    W.checked = W.checked.size ? new Set() : new Set(visibleAssets().map((a) => a.id));
    renderPane();
  };
  $$("[data-chk]").forEach((c) => (c.onclick = (e) => {
    e.stopPropagation();
    c.checked ? W.checked.add(c.dataset.chk) : W.checked.delete(c.dataset.chk);
    renderPane();
  }));
  $$(".alist .it").forEach((it) => (it.onclick = (e) => {
    if (e.target.dataset.chk) return;
    W.sel = it.dataset.id;
    W.tab = "detail";
    renderLeft();
    renderPane();
  }));
  $$("[data-bulk]").forEach((b) => (b.onclick = () => bulk(b.dataset.bulk)));
}
let qT;
function renderListOnly() { clearTimeout(qT); qT = setTimeout(() => { renderPane(); $("#q").focus(); $("#q").setSelectionRange(99, 99); }, 250); }

async function bulk(act) {
  const ids = [...W.checked];
  const pid = W.proj.id;
  if (act === "keep1" || act === "keep0")
    await api("POST", `/api/projects/${pid}/assets/bulk`, { ids, patch: { keep: act === "keep1" } });
  else if (act === "del") {
    if (!confirm(`Xoá ${ids.length} asset?`)) return;
    await api("POST", `/api/projects/${pid}/assets/bulk`, { ids, delete: true });
    W.checked = new Set();
  } else if (act === "extract") return startJob(pid, "extract", { asset_ids: ids });
  else if (act === "qa") return startJob(pid, "qa", { asset_ids: ids });
  else if (act === "regen") return regenSelected(ids);
  await refresh();
}

async function patchAsset(aid, patch) {
  try {
    const a = await api("PATCH", `/api/projects/${W.proj.id}/assets/${aid}`, patch);
    const i = W.proj.assets.findIndex((x) => x.id === aid);
    if (i >= 0) W.proj.assets[i] = a;
    return a;
  } catch (e) { toast(e.message, true); }
}
async function delAsset(aid) {
  if (!confirm("Xoá asset này?")) return;
  await api("DELETE", `/api/projects/${W.proj.id}/assets/${aid}`);
  W.sel = null;
  await refresh();
}

function renderDetail(pane) {
  const a = W.proj.assets.find((x) => x.id === W.sel);
  if (!a) { pane.innerHTML = `<p class="muted">Chọn 1 asset trên ảnh hoặc trong danh sách.<br><br>
    Mẹo: kéo box để di chuyển, kéo góc dưới-phải để đổi kích thước, <kbd>B</kbd> để vẽ box mới, <kbd>Del</kbd> để xoá.</p>`; return; }
  const u = vurl(a);
  const dup = a.dup_of && W.proj.assets.find((x) => x.id === a.dup_of);
  pane.innerHTML = `<div class="detail">
    <div class="preview">${u ? `<img src="${u}">` : `<span class="muted small">Chưa trích xuất</span>`}</div>
    ${a.versions.length ? `<h3>Phiên bản</h3><div class="versions">${a.versions.map((v, i) => `
      <div class="v ${i === a.active ? "on" : ""}" data-ver="${i}"><img src="${vurl(a, i)}">
      <small>${v.kind === "regen" ? "GPT" : "Cắt"} · ${v.w}×${v.h}</small></div>`).join("")}</div>` : ""}
    ${a.qa ? `<h3>QA: <span class="score ${scoreCls(a.qa)}">${a.qa.score}/10</span> ${esc(a.qa.recommend || "")}</h3>
      ${a.qa.issues?.length ? `<ul class="issues">${a.qa.issues.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>` : ""}` : ""}
    ${a.error ? `<p class="small" style="color:var(--bad)">${esc(a.error)}</p>` : ""}
    ${dup ? `<p class="small muted">Trùng với <a href="#" data-goto="${dup.id}">${esc(dup.name)}</a> – sẽ bỏ qua khi export.
      <a href="#" id="notdup">Không phải trùng</a></p>` : ""}
    ${a.occluded > 0.01 ? `<p class="small" style="color:var(--warn)">Bị che ${Math.round(a.occluded * 100)}% – nên Vẽ lại GPT.</p>` : ""}
    <div class="actions">
      <button class="btn" id="dExtract">Trích xuất lại</button>
      <button class="btn" id="dQa">QA</button>
      <button class="btn primary" id="dRegen">Vẽ lại GPT</button>
      <button class="btn danger" id="dDel">Xoá</button>
    </div>
    <h3>Thuộc tính</h3>
    <label class="f"><span>Tên (tên file khi export)</span><input data-f="name" value="${esc(a.name)}"></label>
    <div class="two">
      <label class="f"><span>Loại</span><select data-f="type">${TYPES.map((t) => `<option ${t === a.type ? "selected" : ""}>${t}</option>`).join("")}</select></label>
      <label class="f"><span>Thứ tự lớp (z)</span><input type="number" data-f="z" value="${a.z}"></label>
    </div>
    <label class="f"><span>Tách nền</span><select data-f="method">
      ${[["", "Theo Settings"], ["grabcut", "GrabCut"], ["ai", "AI (rembg)"], ["none", "Không tách"]]
        .map(([v, l]) => `<option value="${v}" ${(a.method || "") === v ? "selected" : ""}>${l}</option>`).join("")}</select></label>
    <div class="chks">
      <label class="chk"><input type="checkbox" data-f="keep" ${a.keep ? "checked" : ""}> Export</label>
      <label class="chk"><input type="checkbox" data-f="remove_text" ${a.remove_text ? "checked" : ""}> Xoá chữ</label>
      <label class="chk"><input type="checkbox" data-f="nine_slice" ${a.nine_slice ? "checked" : ""}> 9-slice</label>
    </div>
    ${a.text ? `<p class="small muted">Chữ phát hiện: “${esc(a.text)}” (${a.text_boxes.length} vùng OCR)</p>` : ""}
    ${a.borders ? `<p class="small muted">9-slice border: L${a.borders.l} T${a.borders.t} R${a.borders.r} B${a.borders.b}</p>` : ""}
    <label class="f"><span>Mô tả (dùng khi vẽ lại bằng GPT)</span><textarea data-f="description">${esc(a.description)}</textarea></label>
    <label class="f"><span>Gợi ý thêm cho lần vẽ lại tới</span><input id="hint" placeholder="vd: giữ viền nâu đậm hơn" value="${esc(a.qa?.hint || "")}"></label>
    <p class="small muted">bbox: ${a.bbox.join(", ")} · id ${a.id}</p>
  </div>`;
  $$("[data-f]").forEach((el) => (el.onchange = async () => {
    const f = el.dataset.f;
    let v = el.type === "checkbox" ? el.checked : el.value;
    if (f === "z") v = Number(v);
    if (f === "method") v = v || null;
    if (f === "name") v = v.trim().replace(/[^\w\-]+/g, "_") || "asset";
    await patchAsset(a.id, { [f]: v });
    if (["keep", "name", "type"].includes(f)) renderLeft();
  }));
  $$("[data-ver]").forEach((el) => (el.onclick = async () => {
    await patchAsset(a.id, { active: Number(el.dataset.ver) });
    renderPane();
    if (W.view === "grid") renderLeft();
  }));
  $$("[data-goto]").forEach((el) => (el.onclick = (e) => { e.preventDefault(); W.sel = el.dataset.goto; renderLeft(); renderPane(); }));
  if ($("#notdup")) $("#notdup").onclick = async (e) => { e.preventDefault(); await patchAsset(a.id, { dup_of: null }); renderPane(); };
  $("#dExtract").onclick = () => startJob(W.proj.id, "extract", { asset_ids: [a.id] });
  $("#dQa").onclick = () => claudeReady() ? startJob(W.proj.id, "qa", { asset_ids: [a.id] }) : toast("Cần Claude key", true);
  $("#dRegen").onclick = () => {
    if (!SETTINGS.keys.openai) return toast("Cần OpenAI API key (Settings).", true);
    startJob(W.proj.id, "regen", { asset_ids: [a.id], hint: $("#hint").value });
  };
  $("#dDel").onclick = () => delAsset(a.id);
}

function renderExport(pane) {
  const p = W.proj, s = SETTINGS.settings;
  const n = p.assets.filter((a) => a.keep && a.active != null && !a.dup_of).length;
  pane.innerHTML = `
    <p>${n} asset sẽ được export (Keep + đã trích xuất${s.skip_duplicates ? ", bỏ trùng" : ""}).</p>
    <label class="row"><span>Scale</span><select id="exScale">${[1, 2, 3, 4].map((v) => `<option ${v == s.export_scale ? "selected" : ""}>${v}</option>`).join("")}</select></label>
    <label class="row"><span>Atlas tối đa</span><select id="exMax">${[1024, 2048, 4096].map((v) => `<option ${v == s.atlas_max_size ? "selected" : ""}>${v}</option>`).join("")}</select></label>
    <label class="chk"><input type="checkbox" id="exDup" ${s.skip_duplicates ? "checked" : ""}> Bỏ asset trùng lặp</label>
    <p class="small muted">Scale tính theo kích thước trên ảnh gốc. Asset vẽ lại bằng GPT có độ phân giải cao nên scale lớn vẫn nét; asset chỉ cắt sẽ được phóng to.</p>
    <div class="actions"><button class="btn primary" id="doExport">Export sprite sheet + PNG</button></div>
    <h3>Lịch sử export</h3>
    <ul class="exports">${(p.exports || []).map((e) => `<li>
      <b>${e.stamp}</b> · ${e.count} sprite · ${e.atlases.length} atlas · ${e.scale}x<br>
      <a class="btn sm" href="/files/${p.id}/${e.zip}" download>Tải .zip</a>
      <button class="btn sm" data-open="${e.folder}">Mở thư mục</button>
      ${e.atlases.map((f) => `<a class="btn sm ghost" target="_blank" href="/files/${p.id}/${e.folder}/${f}">${f}</a>`).join("")}
    </li>`).join("") || `<li class="muted">Chưa có</li>`}</ul>`;
  $("#doExport").onclick = () => startJob(p.id, "export", {
    scale: Number($("#exScale").value), max_size: Number($("#exMax").value), skip_duplicates: $("#exDup").checked });
  $$("[data-open]").forEach((b) => (b.onclick = () =>
    api("POST", `/api/projects/${p.id}/open-folder`, { path: b.dataset.open }).catch((e) => toast(e.message, true))));
}

/* ---------------- boot ---------------- */
loadSettings().then(route).catch((e) => { $("#view").innerHTML = `<div class="page">Không kết nối được backend: ${esc(e.message)}</div>`; });
