/* Copyright 2025 edge_vlm_ros contributors */
/* VLM Experiment Workbench — vanilla JS, no build step, no framework. */

"use strict";

/* ── safe DOM helpers ────────────────────────────────────────────────────── */

/**
 * Create an element with optional class and textContent.
 * All user/artifact-derived text MUST flow through this function.
 * Never use innerHTML with untrusted content.
 */
function _el(tag, className, text) {
  var el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined && text !== null) el.textContent = String(text);
  return el;
}

function _badge(label, kind) {
  return _el("span", "badge " + (kind || ""), label);
}

function _append(parent) {
  var children = Array.prototype.slice.call(arguments, 1);
  children.forEach(function(c) { if (c) parent.appendChild(c); });
  return parent;
}

function _text(str) {
  return document.createTextNode(String(str || ""));
}

function _show(el) { el.style.display = ""; }
function _hide(el) { el.style.display = "none"; }

function _empty(el) { while (el.firstChild) el.removeChild(el.firstChild); }

function _setClass(el, cls, on) {
  if (on) el.classList.add(cls);
  else el.classList.remove(cls);
}

/* ── navigation ──────────────────────────────────────────────────────────── */

var _currentView = "dashboard";
/** Bag currently selected for the ROS experiment panel. */
var _selectedBag = null;

function navigate(viewId) {
  _currentView = viewId;
  document.querySelectorAll("nav button").forEach(function(btn) {
    _setClass(btn, "active", btn.dataset.view === viewId);
  });
  document.querySelectorAll(".view").forEach(function(v) {
    _setClass(v, "active", v.id === "view-" + viewId);
  });
  if (viewId === "dashboard") { _loadDashboard(); }
  else if (viewId === "models") { _loadModels(); }
  else if (viewId === "datasets") { _loadDatasets(); _loadExtractionBags(); }
  else if (viewId === "runs") { _loadRuns(); }
  else if (viewId === "diagnostics") { _loadDiagnostics(); }
  else if (viewId === "frame-explorer") { _loadFrameExplorer(); _loadExtractionBags(); }
  else if (viewId === "profiles") { _loadProfiles(); }
  else if (viewId === "compare") { _loadCompare(); }
  else if (viewId === "experiment") { _loadExpDatasets(); _loadExpProfiles(); }
}

document.addEventListener("DOMContentLoaded", function() {
  document.querySelectorAll("nav button").forEach(function(btn) {
    btn.addEventListener("click", function() { navigate(btn.dataset.view); });
  });
  navigate("dashboard");
});

/* ── API helpers ─────────────────────────────────────────────────────────── */

async function _apiGet(path) {
  var resp = await fetch(path);
  if (!resp.ok) { throw new Error("HTTP " + resp.status + " on " + path); }
  return resp.json();
}

async function _apiPost(path, data) {
  var resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  return { status: resp.status, body: await resp.json() };
}

function _fmtDate(iso) {
  if (!iso) return "";
  try { return new Date(iso).toLocaleString(); }
  catch (e) { return String(iso); }
}

function _fmtLatency(ms) {
  if (ms == null) return "—";
  return Number(ms).toFixed(1) + " ms";
}

function _fmtBytes(n) {
  if (!n) return "0 B";
  var units = ["B", "KB", "MB", "GB"];
  var i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return n.toFixed(i > 0 ? 1 : 0) + " " + units[i];
}

/* ── loading / empty helpers ─────────────────────────────────────────────── */

function _loadingRow(msg) {
  var row = _el("div", "loading-row");
  _append(row, _el("span", "spinner"), _text(msg || "Loading…"));
  return row;
}

function _emptyState(msg, icon) {
  var d = _el("div", "empty-state");
  if (icon) _append(d, _el("div", "empty-state-icon", icon));
  _append(d, _el("p", null, msg));
  return d;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Dashboard view
═══════════════════════════════════════════════════════════════════════════ */

async function _loadDashboard() {
  var serviceEl = document.getElementById("dash-service");
  var gpuEl = document.getElementById("dash-gpu");
  var activeEl = document.getElementById("dash-active");
  var recentEl = document.getElementById("dash-recent");
  if (!serviceEl) return;

  _empty(serviceEl);
  _empty(gpuEl);
  _empty(activeEl);
  _empty(recentEl);
  serviceEl.appendChild(_loadingRow());

  try {
    var data = await _apiGet("/api/status");
    _renderServiceCard(serviceEl, data);
    _renderGpuCard(gpuEl, data);
    _renderActiveRun(activeEl, data);
  } catch (e) {
    _empty(serviceEl);
    serviceEl.appendChild(_el("div", "alert error", "Failed to load status: " + e.message));
  }

  try {
    var runs = await _apiGet("/api/runs");
    _renderRecentRuns(recentEl, (runs.runs || []).slice(0, 5));
  } catch (e) {
    recentEl.appendChild(_el("div", "muted", "Could not load recent runs."));
  }
}

function _renderServiceCard(el, data) {
  _empty(el);
  var srv = data.server || {};
  var reachable = srv.reachable;
  _append(el, _badge(reachable ? "Reachable" : "Unreachable", reachable ? "ok" : "fail"));
  if (data.server_pid) {
    el.appendChild(_el("div", "card-meta", "PID: " + data.server_pid));
  }
  var env = data.env || {};
  if (env.EDGE_VLM_LLM_ENGINE_DIR) {
    el.appendChild(_el("div", "card-meta", "LLM: " + env.EDGE_VLM_LLM_ENGINE_DIR));
  }
  if (env.EDGE_VLM_MULTIMODAL_ENGINE_DIR) {
    el.appendChild(_el("div", "card-meta", "MM: " + env.EDGE_VLM_MULTIMODAL_ENGINE_DIR));
  }
  if (srv.error && !reachable) {
    el.appendChild(_el("div", "card-meta muted", srv.error));
  }
}

function _renderGpuCard(el, data) {
  _empty(el);
  var gpu = data.gpu || {};
  if (gpu.available && gpu.gpus && gpu.gpus.length > 0) {
    gpu.gpus.forEach(function(g) {
      var row = _el("div", "card-meta");
      _append(row, _el("span", "gpu-name", g.name));
      var memText = (g.memory_used_mib === "[N/A]" || g.memory_total_mib === "[N/A]")
        ? " | Unified memory"
        : " | VRAM: " + g.memory_used_mib + "/" + g.memory_total_mib + " MiB";
      row.appendChild(_text(" | Util: " + g.utilization_pct + "%" + memText));
      el.appendChild(row);
    });
  } else {
    _append(el, _el("span", "muted", gpu.error || "No GPU detected"));
  }
}

function _renderActiveRun(el, data) {
  _empty(el);
  var activeId = data.active_ros_run_id;
  if (activeId) {
    var row = _el("div");
    _append(row, _badge("Running", "running"), _text(" "));
    var link = _el("a", null, activeId);
    link.addEventListener("click", function() { navigate("runs"); });
    row.appendChild(link);
    var stopBtn = _el("button", "small danger", "Stop");
    stopBtn.addEventListener("click", function() { stopRosRun(activeId); });
    _append(row, _text(" "), stopBtn);
    el.appendChild(row);
  } else {
    _append(el, _el("span", "muted", "None"));
  }
}

function _renderRecentRuns(el, runs) {
  _empty(el);
  if (!runs || runs.length === 0) {
    el.appendChild(_emptyState("No runs yet.", "🧪"));
    return;
  }
  var tbl = _el("table", "runs-table");
  var thead = tbl.createTHead();
  var hrow = thead.insertRow();
  ["Kind", "Status", "Started", "Latency / Result"].forEach(function(h) {
    hrow.appendChild(_el("th", null, h));
  });
  var tbody = tbl.createTBody();
  runs.forEach(function(run) {
    var tr = tbody.insertRow();
    tr.insertCell().appendChild(_el("span", "run-kind", run.kind || ""));
    var statusBadge = _runStatusBadge(run);
    tr.insertCell().appendChild(statusBadge);
    tr.insertCell().appendChild(_text(_fmtDate(run.created_at)));
    var cell = tr.insertCell();
    if (run.mean_latency_ms != null) {
      cell.appendChild(_text("avg " + _fmtLatency(run.mean_latency_ms)));
    } else if (run.inference_seconds != null) {
      cell.appendChild(_text(_fmtLatency(run.inference_seconds * 1000)));
    }
    tr.style.cursor = "pointer";
    tr.addEventListener("click", function() {
      navigate("runs");
      setTimeout(function() { _showRunDetail(run.run_id); }, 50);
    });
  });
  el.appendChild(tbl);
}

function _runStatusBadge(run) {
  var s = run.status || "";
  var kind = s === "completed" ? "ok" : s === "failed" ? "fail" : s === "running" ? "running" : "pending";
  return _badge(s, kind);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Models view
═══════════════════════════════════════════════════════════════════════════ */

async function _loadModels() {
  var el = document.getElementById("models-list");
  if (!el) return;
  _empty(el);
  el.appendChild(_loadingRow());

  try {
    var data = await _apiGet("/api/models");
    _renderModelList(el, data.models || []);
  } catch (e) {
    _empty(el);
    el.appendChild(_el("div", "alert error", "Failed to load models: " + e.message));
  }
}

function _renderModelList(el, models) {
  _empty(el);
  if (models.length === 0) {
    el.appendChild(_emptyState(
      "No model profiles discovered.\n" +
      "Set EDGE_VLM_MODEL_NAME, EDGE_VLM_LLM_ENGINE_DIR, and EDGE_VLM_MULTIMODAL_ENGINE_DIR " +
      "in the environment and restart the workbench.",
      "🤖"
    ));
    return;
  }
  var grid = _el("div", "catalog-grid");
  models.forEach(function(m) {
    var tile = _el("div", "catalog-tile");
    var nameRow = _el("div", "tile-name");
    if (m.is_active) _append(nameRow, _badge("active", "ok"), _text(" "));
    nameRow.appendChild(_text(m.model_name || m.model_id));
    tile.appendChild(nameRow);

    var modalities = (m.modalities || []).join(", ");
    if (modalities) tile.appendChild(_el("div", "tile-meta", "Modalities: " + modalities));

    if (m.llm_engine_dir) {
      var llmRow = _el("div", "tile-meta");
      _append(llmRow, _badge(m.llm_engine_exists ? "✓" : "✗", m.llm_engine_exists ? "ok" : "fail"),
        _text(" LLM: " + m.llm_engine_dir));
      tile.appendChild(llmRow);
    }
    if (m.multimodal_engine_dir) {
      var mmRow = _el("div", "tile-meta");
      _append(mmRow, _badge(m.multimodal_engine_exists ? "✓" : "✗", m.multimodal_engine_exists ? "ok" : "fail"),
        _text(" MM: " + m.multimodal_engine_dir));
      tile.appendChild(mmRow);
    }
    if (m.plugin_path) {
      var plugRow = _el("div", "tile-meta");
      _append(plugRow, _badge(m.plugin_exists ? "✓" : "✗", m.plugin_exists ? "ok" : "fail"),
        _text(" Plugin: " + m.plugin_path));
      tile.appendChild(plugRow);
    }
    if (m.notes) tile.appendChild(_el("div", "tile-meta muted", m.notes));

    // Raw diagnostic details
    var det = _el("details", "raw-details");
    var sum = _el("summary", null, "Raw profile JSON");
    det.appendChild(sum);
    var pre = _el("pre", null, JSON.stringify(m, null, 2));
    det.appendChild(pre);
    tile.appendChild(det);

    grid.appendChild(tile);
  });
  el.appendChild(grid);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Datasets view
═══════════════════════════════════════════════════════════════════════════ */

async function _loadDatasets() {
  var el = document.getElementById("datasets-content");
  if (!el) return;
  _empty(el);
  el.appendChild(_loadingRow());

  try {
    var data = await _apiGet("/api/datasets");
    _renderDatasets(el, data);
  } catch (e) {
    _empty(el);
    el.appendChild(_el("div", "alert error", "Failed to load datasets: " + e.message));
  }
}

function _renderDatasets(el, data) {
  _empty(el);

  // Rosbags
  var bagsPanel = _el("div", "panel");
  bagsPanel.appendChild(_el("div", "panel-title", "Rosbags"));
  var bags = data.rosbags || [];
  if (bags.length === 0) {
    bagsPanel.appendChild(_emptyState("No rosbags configured.", "🎬"));
  } else {
    var bagGrid = _el("div", "catalog-grid");
    bags.forEach(function(bag) {
      bagGrid.appendChild(_renderBagTile(bag));
    });
    bagsPanel.appendChild(bagGrid);
  }
  el.appendChild(bagsPanel);

  // Image datasets
  var imgPanel = _el("div", "panel");
  imgPanel.appendChild(_el("div", "panel-title", "Image Datasets"));
  var imgs = data.image_datasets || [];
  if (imgs.length === 0) {
    imgPanel.appendChild(_emptyState(
      "No image datasets found. Set --image-dataset-dir to a directory of image subdirectories.",
      "🖼"
    ));
  } else {
    var imgGrid = _el("div", "catalog-grid");
    imgs.forEach(function(ds) {
      var tile = _el("div", "catalog-tile");
      tile.appendChild(_el("div", "tile-name", ds.name));
      tile.appendChild(_el("div", "tile-meta", ds.image_count + " images (" + (ds.extensions || []).join(", ") + ")"));
      tile.appendChild(_el("div", "tile-meta muted", ds.local_path));
      imgGrid.appendChild(tile);
    });
    imgPanel.appendChild(imgGrid);
  }
  el.appendChild(imgPanel);

  // Video datasets
  var vidPanel = _el("div", "panel");
  vidPanel.appendChild(_el("div", "panel-title", "Video Files"));
  var vids = data.video_datasets || [];
  if (vids.length === 0) {
    vidPanel.appendChild(_emptyState(
      "No video files found. Set --video-dataset-dir to a directory of video files.",
      "🎞"
    ));
  } else {
    var vidGrid = _el("div", "catalog-grid");
    vids.forEach(function(v) {
      var tile = _el("div", "catalog-tile");
      tile.appendChild(_el("div", "tile-name", v.name));
      tile.appendChild(_el("div", "tile-meta", _fmtBytes(v.size_bytes)));
      tile.appendChild(_el("div", "tile-meta muted", v.local_path));
      vidGrid.appendChild(tile);
    });
    vidPanel.appendChild(vidGrid);
  }
  el.appendChild(vidPanel);
}

function _renderBagTile(bag) {
  var tile = _el("div", "catalog-tile");
  var nameRow = _el("div", "tile-name");
  var badge = bag.installed ? _badge("installed", "ok") : _badge("not installed", "pending");
  _append(nameRow, badge, _text(" " + bag.name));
  tile.appendChild(nameRow);
  tile.appendChild(_el("div", "tile-meta", bag.description));
  if (bag.installed && bag.local_path) {
    tile.appendChild(_el("div", "tile-meta muted", bag.local_path));
    if (bag.size_bytes) tile.appendChild(_el("div", "tile-meta", _fmtBytes(bag.size_bytes)));
    if (bag.duration_seconds != null) {
      tile.appendChild(_el("div", "tile-meta", "Duration: " + bag.duration_seconds.toFixed(1) + "s"));
    }
    var topicNames = [];
    if (bag.topic_names && bag.topic_names.length > 0) {
      topicNames = bag.topic_names;
    } else if (bag.topics && bag.topics.length > 0) {
      if (typeof bag.topics[0] === "string") topicNames = bag.topics;
      else topicNames = bag.topics.map(function(t) { return t.name; });
    }
    if (topicNames.length > 0) {
      tile.appendChild(_el("div", "tile-meta", "Topics: " + topicNames.join(", ")));
    }
  }
  if (bag.content_types) {
    tile.appendChild(_el("div", "tile-meta muted", "Msg types: " + bag.content_types));
  }
  if (bag.source) {
    tile.appendChild(_el("div", "tile-meta muted", "Source: " + bag.source));
  }

  var actions = _el("div", "tile-actions");
  if (bag.downloadable && !bag.installed) {
    var dlBtn = _el("button", "small", "Download");
    dlBtn.addEventListener("click", function() { _downloadBag(bag.key, dlBtn); });
    actions.appendChild(dlBtn);
  }
  if (bag.installed) {
    if (bag.raw_image_compatible) {
      var useBtn = _el("button", "small secondary", "Use in Experiment");
      useBtn.addEventListener("click", function() { selectBagForExperiment(bag); });
      actions.appendChild(useBtn);
    } else {
      var noteText = bag.compatibility_note || "Incompatible bag type";
      var disabledBtn = _el("button", "small secondary", noteText);
      disabledBtn.disabled = true;
      disabledBtn.title = noteText;
      actions.appendChild(disabledBtn);
    }
  }
  tile.appendChild(actions);
  return tile;
}

async function _downloadBag(key, btn) {
  btn.disabled = true;
  btn.textContent = "Starting…";
  try {
    var result = await _apiPost("/api/datasets/download", { bag_key: key });
    if (result.status === 202) {
      btn.textContent = "Downloading…";
      var runId = result.body.run_id;
      _pollDownload(runId, btn);
    } else {
      btn.textContent = "Error: " + (result.body.error || result.status);
      btn.disabled = false;
    }
  } catch (e) {
    btn.textContent = "Error";
    btn.disabled = false;
  }
}

async function _pollDownload(runId, btn) {
  var maxPolls = 300;
  var count = 0;
  while (count++ < maxPolls) {
    await new Promise(function(r) { setTimeout(r, 2000); });
    try {
      var data = await _apiGet("/api/runs/" + runId);
      if (data.status === "completed") {
        btn.textContent = "Downloaded ✓";
        _loadDatasets();
        return;
      } else if (data.status === "failed" || data.status === "stopped") {
        btn.textContent = "Failed";
        btn.disabled = false;
        return;
      }
    } catch (e) { /* keep polling */ }
  }
  btn.textContent = "Timeout";
  btn.disabled = false;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Experiment view
═══════════════════════════════════════════════════════════════════════════ */

/**
 * Select an installed compatible bag for the ROS experiment panel.
 * Stores the bag, populates the selected-bag status widget, suggests the first
 * raw image topic, and navigates to the Experiment view.
 */
function selectBagForExperiment(bag) {
  _selectedBag = bag;
  var statusEl = document.getElementById("ros-selected-bag");
  if (statusEl) {
    _empty(statusEl);
    var infoSpan = _el("span", "selected-bag-info",
      "Selected bag: " + bag.name + " (" + bag.local_path + ")");
    var clrBtn = _el("button", "small secondary", "Clear");
    clrBtn.addEventListener("click", function() {
      _selectedBag = null;
      statusEl.style.display = "none";
    });
    statusEl.appendChild(infoSpan);
    statusEl.appendChild(clrBtn);
    statusEl.style.display = "";
  }
  // Suggest the first raw image topic if the ROS topic field is at its default.
  var topicEl = document.getElementById("ros-topic");
  if (topicEl && bag.image_topics && bag.image_topics.length > 0) {
    topicEl.value = bag.image_topics[0];
  }
  navigate("experiment");
}

// Standalone inference
async function submitInfer() {
  var btn = document.getElementById("infer-submit-btn");
  var resultEl = document.getElementById("infer-result-card");
  var outEl = document.getElementById("infer-out");
  if (btn) btn.disabled = true;
  _empty(resultEl);
  resultEl.style.display = "none";

  var form = document.getElementById("infer-form");
  if (!form) return;
  var fd = new FormData(form);

  try {
    var resp = await fetch("/api/infer", { method: "POST", body: fd });
    var data = await resp.json();
    if (outEl) outEl.textContent = JSON.stringify(data, null, 2);
    _renderInferResult(resultEl, data);
    resultEl.style.display = "";
    var det = document.getElementById("infer-raw-details");
    if (det) det.style.display = "";
  } catch (e) {
    if (resultEl) {
      resultEl.textContent = "Error: " + e.message;
      resultEl.style.display = "";
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

function _renderInferResult(el, data) {
  _empty(el);
  var header = _el("div", "result-header");
  _append(header,
    _badge(data.success ? "Success" : "Failed", data.success ? "ok" : "fail"),
    _el("span", "result-latency", _fmtLatency(data.inference_seconds != null ? data.inference_seconds * 1000 : null))
  );
  el.appendChild(header);
  if (data.success && data.text) {
    el.appendChild(_el("div", "result-text", data.text));
  }
  if (data.error) {
    el.appendChild(_el("div", "result-error", "Error: " + data.error));
  }
  if (data.prompt) {
    el.appendChild(_el("div", "result-meta", "Prompt: " + data.prompt));
  }
}

// ROS experiment
async function startRos() {
  var outEl = document.getElementById("ros-start-out");
  var logsEl = document.getElementById("ros-logs-area");
  var rawEl = document.getElementById("ros-out");

  var params = {
    image_topic: document.getElementById("ros-topic").value.trim(),
    max_generate_length: parseInt(document.getElementById("ros-max-gen").value, 10),
    instruction_delivery_mode: document.getElementById("ros-delivery").value,
    observation_history_max_entries: parseInt(document.getElementById("ros-hist-entries").value, 10),
    observation_history_max_chars: parseInt(document.getElementById("ros-hist-chars").value, 10),
    playback_duration: parseInt(document.getElementById("ros-playback").value, 10),
    result_timeout: parseInt(document.getElementById("ros-timeout").value, 10),
    success_results_required: parseInt(document.getElementById("ros-required").value, 10)
  };
  var prompt = document.getElementById("ros-prompt").value.trim();
  if (prompt) params.prompt = prompt;
  // Include the selected bag path when a compatible bag has been chosen.
  if (_selectedBag && _selectedBag.local_path) {
    params.rosbag_path = _selectedBag.local_path;
  }

  if (outEl) { outEl.textContent = "Starting…"; _show(outEl); }
  try {
    var result = await _apiPost("/api/ros/start", { params: params });
    if (rawEl) rawEl.textContent = JSON.stringify(result.body, null, 2);
    var data = result.body;
    if (result.status >= 400) {
      if (outEl) outEl.textContent = "Error: " + (data.error || result.status);
    } else {
      if (outEl) { outEl.textContent = "Started: " + data.run_id; }
      if (logsEl) _show(logsEl);
      showActiveRun(data.run_id);
      _pollRosLogs(data.run_id);
    }
    var det = document.getElementById("ros-raw-details");
    if (det) _show(det);
  } catch (e) {
    if (outEl) outEl.textContent = "Error: " + e.message;
  }
}

async function stopRosRun(runId) {
  try {
    await _apiPost("/api/ros/stop", { run_id: runId });
    _hideActiveRun();
    _loadDashboard();
  } catch (e) { /* ignore */ }
}

function showActiveRun(runId) {
  var bannerEl = document.getElementById("ros-active");
  var idEl = document.getElementById("ros-active-id");
  if (bannerEl && idEl) {
    idEl.textContent = runId;
    _show(bannerEl);
  }
}

function _hideActiveRun() {
  var bannerEl = document.getElementById("ros-active");
  if (bannerEl) _hide(bannerEl);
}

async function _pollRosLogs(runId) {
  var logsEl = document.getElementById("ros-logs");
  var maxPolls = 300;
  var count = 0;
  while (count++ < maxPolls) {
    await new Promise(function(r) { setTimeout(r, 1500); });
    try {
      var data = await _apiGet("/api/runs/" + runId + "/logs");
      if (logsEl) {
        logsEl.textContent = (data.log_lines || []).join("\n");
        logsEl.scrollTop = logsEl.scrollHeight;
      }
      if (data.terminal) {
        _hideActiveRun();
        navigate("runs");
        setTimeout(function() { _showRunDetail(runId); }, 50);
        return;
      }
    } catch (e) { /* keep polling */ }
  }
}

// Frame-sequence experiment
var _activeExperimentRunId = null;

async function _loadExpDatasets() {
  var sel = document.getElementById("exp-dataset-id");
  if (!sel) return;
  try {
    var data = await _apiGet("/api/frame-datasets");
    var datasets = data.datasets || [];
    _empty(sel);
    var blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "— select a frame dataset —";
    sel.appendChild(blank);
    datasets.forEach(function(ds) {
      var opt = document.createElement("option");
      opt.value = ds.dataset_id;
      opt.textContent = (ds.bag_key || ds.dataset_id) + " (" + (ds.frame_count || "?") + " frames)";
      sel.appendChild(opt);
    });
  } catch (e) { /* leave empty */ }
}

async function _loadExpProfiles() {
  var sel = document.getElementById("exp-profile-name");
  if (!sel) return;
  try {
    var data = await _apiGet("/api/profiles");
    var profiles = data.profiles || [];
    _empty(sel);
    var blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "— none (use fields below) —";
    sel.appendChild(blank);
    profiles.forEach(function(p) {
      var opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.name + (p.version ? " v" + p.version : "");
      sel.appendChild(opt);
    });
  } catch (e) { /* leave empty */ }
}

async function submitExperiment() {
  var btn = document.getElementById("exp-submit-btn");
  var cancelBtn = document.getElementById("exp-cancel-btn");
  var resultEl = document.getElementById("exp-result");
  if (btn) btn.disabled = true;
  if (cancelBtn) cancelBtn.style.display = "";
  _empty(resultEl);

  var datasetId = ((document.getElementById("exp-dataset-id") || {}).value || "").trim();
  if (!datasetId) {
    _append(resultEl, _el("div", "alert error", "Please select a frame dataset first."));
    if (btn) btn.disabled = false;
    if (cancelBtn) cancelBtn.style.display = "none";
    return;
  }

  var indicesRaw = ((document.getElementById("exp-frame-indices") || {}).value || "").trim();
  var frameIndices = null;
  if (indicesRaw) {
    frameIndices = indicesRaw.split(",")
      .map(function(s) { return parseInt(s.trim(), 10); })
      .filter(function(n) { return !isNaN(n); });
  }

  var profileName = ((document.getElementById("exp-profile-name") || {}).value || "").trim() || null;

  var params = {
    frame_dataset_id: datasetId,
    strategy: (document.getElementById("exp-strategy") || {}).value || "single_frame",
    observation_history_max_entries: parseInt(
      (document.getElementById("exp-history-entries") || {}).value || "0", 10),
    observation_history_max_chars: parseInt(
      (document.getElementById("exp-history-chars") || {}).value || "4000", 10),
    max_generate_length: parseInt(
      (document.getElementById("exp-max-gen") || {}).value || "96", 10),
    temperature: parseFloat(
      (document.getElementById("exp-temperature") || {}).value || "0.2"),
    top_p: parseFloat((document.getElementById("exp-top-p") || {}).value || "0.9"),
    top_k: parseInt((document.getElementById("exp-top-k") || {}).value || "20", 10),
    timeout_seconds: parseInt(
      (document.getElementById("exp-timeout") || {}).value || "120", 10),
    notes: ((document.getElementById("exp-notes") || {}).value || "").trim()
  };
  if (frameIndices !== null) params.frame_indices = frameIndices;
  if (profileName) {
    params.profile_name = profileName;
  } else {
    params.task_prompt = ((document.getElementById("exp-prompt") || {}).value || "").trim();
    params.system_instruction = ((document.getElementById("exp-system") || {}).value || "").trim();
  }

  try {
    var result = await _apiPost("/api/experiment/run", params);
    if (result.status >= 400) {
      _append(resultEl,
        _el("div", "alert error", "Error: " + (result.body.error || result.status)));
      if (cancelBtn) cancelBtn.style.display = "none";
    } else {
      var runId = result.body.run_id;
      _activeExperimentRunId = runId;
      _append(resultEl,
        _el("div", "alert success",
          "Experiment started (run_id: " + runId + ").\n" +
          "Results will appear in the Runs view when complete."));
      _pollExperimentResult(runId, resultEl);
    }
  } catch (e) {
    _append(resultEl, _el("div", "alert error", "Error: " + e.message));
    if (cancelBtn) cancelBtn.style.display = "none";
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function cancelExperiment() {
  var runId = _activeExperimentRunId;
  if (!runId) return;
  try {
    await _apiPost("/api/experiment/" + runId + "/cancel", {});
    var resultEl = document.getElementById("exp-result");
    if (resultEl) {
      _append(resultEl, _el("div", "alert", "Cancellation requested — stopping after current frame."));
    }
  } catch (e) { /* ignore */ }
  var cancelBtn = document.getElementById("exp-cancel-btn");
  if (cancelBtn) cancelBtn.style.display = "none";
}

async function _pollExperimentResult(runId, statusEl) {
  var maxPolls = 300;
  var count = 0;
  while (count++ < maxPolls) {
    await new Promise(function(r) { setTimeout(r, 2000); });
    try {
      var data = await _apiGet("/api/runs/" + runId);
      var status = data.status;
      // Update progress if available.
      var prog = data.progress_frames;
      var total = data.image_count;
      if (prog != null && total != null && status === "running") {
        var progEl = document.getElementById("exp-progress-" + runId);
        if (!progEl) {
          progEl = _el("div", "run-progress", "");
          progEl.id = "exp-progress-" + runId;
          if (statusEl) statusEl.appendChild(progEl);
        }
        progEl.textContent = "Progress: " + prog + " / " + total + " frames";
      }
      if (status === "completed" || status === "failed" || status === "stopped") {
        _activeExperimentRunId = null;
        var cancelBtn = document.getElementById("exp-cancel-btn");
        if (cancelBtn) cancelBtn.style.display = "none";
        _empty(statusEl);
        if (status === "completed") {
          _append(statusEl, _el("div", "alert success",
            "Experiment complete! Showing results…"));
          navigate("runs");
          setTimeout(function() { _showRunDetail(runId); }, 50);
        } else if (status === "stopped") {
          _append(statusEl, _el("div", "alert",
            "Experiment stopped (cancelled). Partial results available in Runs view."));
          navigate("runs");
          setTimeout(function() { _showRunDetail(runId); }, 50);
        } else {
          _append(statusEl, _el("div", "alert error",
            "Experiment failed: " + (data.error || "unknown error")));
        }
        return;
      }
    } catch (e) { /* keep polling */ }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Runs view
═══════════════════════════════════════════════════════════════════════════ */

var _selectedRunIds = new Set();

async function _loadRuns() {
  var listEl = document.getElementById("runs-list");
  if (!listEl) return;
  _empty(listEl);
  listEl.appendChild(_loadingRow());

  try {
    var data = await _apiGet("/api/runs");
    _renderRunsTable(listEl, data.runs || []);
  } catch (e) {
    _empty(listEl);
    listEl.appendChild(_el("div", "alert error", "Failed to load runs: " + e.message));
  }
}

function _renderRunsTable(el, runs) {
  _empty(el);
  if (runs.length === 0) {
    el.appendChild(_emptyState("No runs yet.\nStart an experiment or inference to see results here.", "🧪"));
    return;
  }

  var compareBar = _el("div");
  var compareBtn = _el("button", "secondary small", "Compare Selected");
  compareBtn.addEventListener("click", _compareSelected);
  _append(compareBar, _text("Select runs to compare: "), compareBtn);
  el.appendChild(compareBar);

  var tbl = _el("table", "runs-table");
  var thead = tbl.createTHead();
  var hrow = thead.insertRow();
  ["", "Kind", "Status", "Strategy", "Started", "Frames", "Avg Latency"].forEach(function(h) {
    hrow.appendChild(_el("th", null, h));
  });
  var tbody = tbl.createTBody();

  runs.forEach(function(run) {
    var tr = tbody.insertRow();
    // Checkbox
    var chkCell = tr.insertCell();
    var chk = document.createElement("input");
    chk.type = "checkbox";
    chk.setAttribute("aria-label", "Select run " + run.run_id);
    chk.checked = _selectedRunIds.has(run.run_id);
    chk.addEventListener("change", function() {
      if (chk.checked) _selectedRunIds.add(run.run_id);
      else _selectedRunIds.delete(run.run_id);
    });
    chkCell.appendChild(chk);

    tr.insertCell().appendChild(_el("span", "run-kind", run.kind || ""));
    tr.insertCell().appendChild(_runStatusBadge(run));
    tr.insertCell().appendChild(_text(run.strategy || ""));
    tr.insertCell().appendChild(_text(_fmtDate(run.created_at)));

    var framesCell = tr.insertCell();
    if (run.successful_frames != null) {
      framesCell.appendChild(_text(run.successful_frames + "/" + (run.image_count || (run.successful_frames + (run.failed_frames || 0)))));
    } else if (run.result_frames) {
      framesCell.appendChild(_text(run.result_frames.length));
    }

    tr.insertCell().appendChild(_text(_fmtLatency(run.mean_latency_ms)));

    var viewLink = _el("a", null, "View →");
    viewLink.addEventListener("click", function(e) {
      e.stopPropagation();
      _showRunDetail(run.run_id);
    });
    tr.insertCell().appendChild(viewLink);
  });

  el.appendChild(tbl);
}

async function _showRunDetail(runId) {
  var detailEl = document.getElementById("run-detail");
  if (!detailEl) return;
  _show(detailEl);
  _empty(detailEl);
  detailEl.appendChild(_loadingRow("Loading run " + runId + "…"));

  try {
    var data = await _apiGet("/api/runs/" + runId);
    _renderRunDetail(detailEl, data);
  } catch (e) {
    _empty(detailEl);
    detailEl.appendChild(_el("div", "alert error", "Failed to load run: " + e.message));
  }
}

function _renderRunDetail(el, run) {
  _empty(el);

  // Header
  var header = _el("div", "run-detail-header");
  _append(header,
    _el("span", "run-kind", run.kind || ""),
    _runStatusBadge(run),
    _text(" " + run.run_id)
  );
  if (run.strategy) _append(header, _badge(run.strategy, "info"));
  if (run.mean_latency_ms != null) {
    _append(header, _el("span", "result-latency", "avg " + _fmtLatency(run.mean_latency_ms)));
  }
  el.appendChild(header);

  // Summary row
  var meta = _el("div", "result-meta");
  var parts = [];
  if (run.created_at) parts.push("Started: " + _fmtDate(run.created_at));
  if (run.completed_at) parts.push("Completed: " + _fmtDate(run.completed_at));
  if (run.successful_frames != null) {
    parts.push("Frames: " + run.successful_frames + " ok / " + (run.failed_frames || 0) + " failed");
  }
  if (run.repetition_flags) parts.push("Repetition flags: " + run.repetition_flags);
  if (run.task_prompt) parts.push("Prompt: " + run.task_prompt);
  meta.appendChild(_text(parts.join("  |  ")));
  el.appendChild(meta);

  // For standalone inference runs, show the full response
  if (run.kind === "standalone") {
    var rc = _el("div", "result-card");
    var rh = _el("div", "result-header");
    _append(rh, _badge(run.success ? "Success" : "Failed", run.success ? "ok" : "fail"),
      _el("span", "result-latency", _fmtLatency(run.inference_seconds != null ? run.inference_seconds * 1000 : null)));
    rc.appendChild(rh);
    if (run.text) rc.appendChild(_el("div", "result-text", run.text));
    if (run.error) rc.appendChild(_el("div", "result-error", run.error));
    if (run.prompt) rc.appendChild(_el("div", "result-meta", "Prompt: " + run.prompt));
    el.appendChild(rc);
  }

  // Frame results
  var frames = run.result_frames || [];
  if (frames.length > 0) {
    var framesTitle = _el("div", "panel-title", "Frame Results (" + frames.length + ")");
    el.appendChild(framesTitle);
    var frameList = _el("div", "frame-list");
    frames.forEach(function(fr) {
      frameList.appendChild(_renderFrameCard(fr));
    });
    el.appendChild(frameList);
  }

  // Benchmark summary (ROS runs)
  if (run.benchmark_summary) {
    var bs = run.benchmark_summary;
    var bsTitle = "Benchmark Summary";
    if (bs.source) { bsTitle += " — " + bs.source; }
    el.appendChild(_el("div", "panel-title", bsTitle));
    var bsEl = _el("div", "result-meta");
    bsEl.appendChild(_text(
      "Frames: " + bs.frame_count +
      " | Success: " + bs.successful_frames +
      " | Failed: " + bs.failed_frames +
      " | Dropped: " + (bs.dropped_frames || 0) +
      " | Mean: " + _fmtLatency(bs.mean_inference_ms) +
      " | Min: " + _fmtLatency(bs.min_inference_ms) +
      " | Max: " + _fmtLatency(bs.max_inference_ms)
    ));
    el.appendChild(bsEl);
    if (bs.count_note) {
      var noteEl = _el("div", "result-meta");
      noteEl.appendChild(_text("\u26a0\ufe0f " + bs.count_note));
      el.appendChild(noteEl);
    }
  }

  el.appendChild(_el("hr", "separator"));

  // Configuration
  var cfgDetails = _el("details", "raw-details");
  cfgDetails.appendChild(_el("summary", null, "Configuration & reproducibility metadata"));
  var cfgFields = {
    run_id: run.run_id,
    kind: run.kind,
    strategy: run.strategy,
    max_generate_length: run.max_generate_length,
    temperature: run.temperature,
    top_p: run.top_p,
    top_k: run.top_k,
    observation_history_max_entries: run.observation_history_max_entries,
    timeout_seconds: run.timeout_seconds,
    notes: run.notes,
    params: run.params,
    external_worker: run.external_worker
  };
  cfgDetails.appendChild(_el("pre", null, JSON.stringify(cfgFields, null, 2)));
  el.appendChild(cfgDetails);

  // Raw JSON
  var rawDetails = _el("details", "raw-details");
  rawDetails.appendChild(_el("summary", null, "Raw manifest JSON"));
  rawDetails.appendChild(_el("pre", null, JSON.stringify(run, null, 2)));
  el.appendChild(rawDetails);
}

function _renderFrameCard(fr) {
  var card = _el("div", "frame-card");
  var hdr = _el("div", "frame-header");
  var frameLabel = fr.frame_index != null
    ? "Frame " + fr.frame_index
    : (fr.frame_seq != null ? "Frame " + fr.frame_seq : "Frame");
  _append(hdr,
    _el("span", null, frameLabel),
    _badge(fr.success ? "ok" : "fail", fr.success ? "ok" : "fail"),
    _el("span", "result-latency", _fmtLatency(fr.latency_ms))
  );
  if (fr.repetition_flag) _append(hdr, _badge("repetition", "rep"));
  if (fr.history_entries_used != null && fr.history_entries_used > 0) {
    _append(hdr, _el("span", "result-meta", "hist:" + fr.history_entries_used));
  }
  card.appendChild(hdr);
  if (fr.text || fr.response) {
    card.appendChild(_el("div", "result-text", fr.text || fr.response));
  }
  if (fr.error) card.appendChild(_el("div", "result-error", fr.error));
  if (fr.image_path || fr.source_timestamp_ns != null) {
    var meta = [];
    if (fr.image_path) meta.push(fr.image_path);
    if (fr.source_timestamp_ns != null) {
      meta.push("ts: " + fr.source_timestamp_ns);
    }
    card.appendChild(_el("div", "result-meta", meta.join("  |  ")));
  }
  return card;
}

// Comparison
async function _compareSelected() {
  var detailEl = document.getElementById("run-detail");
  if (!detailEl) return;
  if (_selectedRunIds.size < 2) {
    alert("Select at least 2 runs to compare.");
    return;
  }
  _show(detailEl);
  _empty(detailEl);
  detailEl.appendChild(_loadingRow("Loading comparison…"));

  try {
    var runs = await Promise.all(
      Array.from(_selectedRunIds).map(function(id) {
        return _apiGet("/api/runs/" + id);
      })
    );
    _renderComparison(detailEl, runs);
  } catch (e) {
    _empty(detailEl);
    detailEl.appendChild(_el("div", "alert error", "Comparison failed: " + e.message));
  }
}

function _renderComparison(el, runs) {
  _empty(el);
  el.appendChild(_el("div", "panel-title", "Run Comparison (" + runs.length + " runs)"));

  var tbl = _el("table", "compare-table");
  var thead = tbl.createTHead();
  var hrow = thead.insertRow();
  var cols = ["Metric"].concat(runs.map(function(r) {
    return (r.kind || "") + "\n" + r.run_id.substring(0, 8);
  }));
  cols.forEach(function(h) { hrow.appendChild(_el("th", null, h)); });

  var tbody = tbl.createTBody();

  function _row(label, fn) {
    var tr = tbody.insertRow();
    tr.insertCell().appendChild(_text(label));
    runs.forEach(function(r) {
      var v = fn(r);
      var cell = tr.insertCell();
      if (v !== null && v !== undefined) cell.appendChild(_text(v));
    });
  }

  _row("Strategy", function(r) { return r.strategy || r.kind; });
  _row("Status", function(r) { return r.status; });
  _row("Successful frames", function(r) { return r.successful_frames; });
  _row("Failed frames", function(r) { return r.failed_frames; });
  _row("Repetition flags", function(r) { return r.repetition_flags; });
  _row("Mean latency (ms)", function(r) { return r.mean_latency_ms != null ? r.mean_latency_ms.toFixed(1) : "—"; });
  _row("Min latency (ms)", function(r) { return r.min_latency_ms != null ? r.min_latency_ms.toFixed(1) : "—"; });
  _row("Max latency (ms)", function(r) { return r.max_latency_ms != null ? r.max_latency_ms.toFixed(1) : "—"; });
  _row("History depth", function(r) { return r.observation_history_max_entries; });
  _row("Max tokens", function(r) { return r.max_generate_length; });
  _row("Temperature", function(r) { return r.temperature; });
  _row("Task prompt", function(r) { return r.task_prompt || (r.params && r.params.prompt) || ""; });
  _row("Notes", function(r) { return r.notes || ""; });
  _row("Started", function(r) { return _fmtDate(r.created_at); });
  _row("Completed", function(r) { return _fmtDate(r.completed_at); });

  el.appendChild(tbl);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Diagnostics view
═══════════════════════════════════════════════════════════════════════════ */

async function _loadDiagnostics() {
  var statusEl = document.getElementById("diag-status-pre");
  if (!statusEl) return;
  statusEl.textContent = "Loading…";
  try {
    var data = await _apiGet("/api/status");
    statusEl.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    statusEl.textContent = "Error: " + e.message;
  }
}

async function loadDiagStatus() { await _loadDiagnostics(); }

async function loadDiagRuns() {
  var el = document.getElementById("diag-runs-pre");
  if (!el) return;
  el.textContent = "Loading…";
  try {
    var data = await _apiGet("/api/runs");
    el.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    el.textContent = "Error: " + e.message;
  }
}

async function loadDiagModels() {
  var el = document.getElementById("diag-models-pre");
  if (!el) return;
  el.textContent = "Loading…";
  try {
    var data = await _apiGet("/api/models");
    el.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    el.textContent = "Error: " + e.message;
  }
}

async function loadDiagDatasets() {
  var el = document.getElementById("diag-datasets-pre");
  if (!el) return;
  el.textContent = "Loading…";
  try {
    var data = await _apiGet("/api/datasets");
    el.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    el.textContent = "Error: " + e.message;
  }
}

/* ── Frame Extraction ────────────────────────────────────────────────────── */

async function _startExtraction(bagKey, imageTopic, options) {
  var body = Object.assign({ bag_key: bagKey, image_topic: imageTopic }, options || {});
  return _apiPost("/api/extract", body);
}

async function _cancelExtraction(runId) {
  return _apiPost("/api/extract/" + runId + "/cancel", {});
}

/* ── Extraction panel ────────────────────────────────────────────────────── */

/** Active extraction run ID for the panel (null when idle). */
var _extractionRunId = null;
var _extractionBagsByKey = {};

/**
 * Populate the bag-key <select> in the extraction panel with installed rosbags
 * from the catalog.
 */
async function _loadExtractionBags() {
  var sel = document.getElementById("extract-bag-key");
  if (!sel) return;
  try {
    var data = await _apiGet("/api/datasets");
    var bags = (data.rosbags || []).filter(function(b) { return b.installed; });
    _extractionBagsByKey = {};
    _empty(sel);
    var blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "— select an installed rosbag —";
    sel.appendChild(blank);
    bags.forEach(function(bag) {
      _extractionBagsByKey[bag.key] = bag;
      var opt = document.createElement("option");
      opt.value = bag.key;
      var label = bag.display_name || bag.name || bag.key;
      if (bag.storage_identifier) label += " [" + bag.storage_identifier + "]";
      if (bag.duration_seconds) label += " (" + bag.duration_seconds.toFixed(1) + "s)";
      opt.textContent = label;
      sel.appendChild(opt);
    });
    _onExtractBagChange();
  } catch (e) { /* leave empty on error */ }
}

/**
 * Read the extraction panel form and start a frame extraction run.
 * Wires up progress polling and auto-opens the dataset on completion.
 */
async function _submitExtractPanel() {
  var startBtn = document.getElementById("extract-start-btn");
  var cancelBtn = document.getElementById("extract-cancel-btn");
  var statusEl = document.getElementById("extract-status");

  var bagKey = (document.getElementById("extract-bag-key") || {}).value || "";
  var topicSelect = document.getElementById("extract-image-topic-select");
  var topicAdvanced = document.getElementById("extract-topic-advanced");
  var topicInput = document.getElementById("extract-image-topic");
  var topic = "";
  if (topicAdvanced && topicAdvanced.checked) {
    topic = (topicInput || {}).value || "";
  } else {
    topic = (topicSelect || {}).value || "";
  }
  var startOffset = parseFloat((document.getElementById("extract-start-offset") || {}).value || "0") || 0;
  var durationVal = (document.getElementById("extract-duration") || {}).value || "";
  var endOffsetVal = (document.getElementById("extract-end-offset") || {}).value || "";
  var sampleInterval = (document.getElementById("extract-sample-interval") || {}).value || "";
  var targetCount = (document.getElementById("extract-target-count") || {}).value || "";
  var maxFrames = parseInt((document.getElementById("extract-max-frames") || {}).value || "100", 10) || 100;

  if (!bagKey || !topic) {
    if (statusEl) statusEl.textContent = "Select a bag and choose a supported image topic.";
    return;
  }

  var options = { start_offset: startOffset, max_frames: maxFrames };
  if (durationVal) options.duration = parseFloat(durationVal);
  if (endOffsetVal) options.end_offset = parseFloat(endOffsetVal);
  if (sampleInterval) options.sample_interval = parseFloat(sampleInterval);
  if (targetCount) options.target_sample_count = parseInt(targetCount, 10);

  if (startBtn) startBtn.disabled = true;
  if (cancelBtn) { cancelBtn.disabled = false; _show(cancelBtn); }
  if (statusEl) statusEl.textContent = "Starting extraction…";

  try {
    var result = await _startExtraction(bagKey, topic, options);
    if (result.status !== 202) {
      if (statusEl) statusEl.textContent = "Error: " + (result.body.error || result.status);
      if (startBtn) startBtn.disabled = false;
      if (cancelBtn) _hide(cancelBtn);
      return;
    }
    _extractionRunId = result.body.run_id;
    if (statusEl) statusEl.textContent = "Extracting… run " + _extractionRunId.slice(0, 8);
    _pollExtractionRun(_extractionRunId, statusEl, startBtn, cancelBtn);
  } catch (e) {
    if (statusEl) statusEl.textContent = "Error: " + e.message;
    if (startBtn) startBtn.disabled = false;
    if (cancelBtn) _hide(cancelBtn);
  }
}

/**
 * Poll an extraction run until it reaches a terminal state, then refresh the
 * frame-dataset list and auto-open the new dataset.
 */
async function _pollExtractionRun(runId, statusEl, startBtn, cancelBtn) {
  var maxPolls = 600;
  var count = 0;
  while (count++ < maxPolls) {
    await new Promise(function(r) { setTimeout(r, 1500); });
    if (_extractionRunId !== runId) return;  // superseded
    try {
      var data = await _apiGet("/api/runs/" + runId);
      var status = data.status || "unknown";
      if (statusEl) statusEl.textContent = "Status: " + status;
      if (status === "completed") {
        _extractionRunId = null;
        if (startBtn) startBtn.disabled = false;
        if (cancelBtn) _hide(cancelBtn);
        if (statusEl) statusEl.textContent = "Extraction complete. Loading dataset…";
        // Refresh frame-dataset list and open the new dataset.
        await _loadFrameExplorer();
        var datasetId = data.dataset_id;
        if (datasetId) {
          await _openFrameDataset(datasetId);
          var explorerPanel = document.getElementById("frame-explorer-viewer");
          if (explorerPanel) _show(explorerPanel);
        }
        if (statusEl) statusEl.textContent = "Done. Dataset: " + (datasetId || "?");
        return;
      } else if (status === "failed" || status === "stopped" || status === "ros_unavailable") {
        _extractionRunId = null;
        if (startBtn) startBtn.disabled = false;
        if (cancelBtn) _hide(cancelBtn);
        if (statusEl) statusEl.textContent = "Extraction " + status;
        return;
      }
    } catch (e) { /* keep polling */ }
  }
  if (statusEl) statusEl.textContent = "Polling timed out.";
  if (startBtn) startBtn.disabled = false;
  if (cancelBtn) _hide(cancelBtn);
}

/**
 * Cancel the currently active extraction run from the panel.
 */
async function _cancelExtractionRun() {
  var cancelBtn = document.getElementById("extract-cancel-btn");
  var statusEl = document.getElementById("extract-status");
  var runId = _extractionRunId;
  if (!runId) return;
  if (cancelBtn) cancelBtn.disabled = true;
  try {
    await _cancelExtraction(runId);
    if (statusEl) statusEl.textContent = "Cancelling…";
  } catch (e) {
    if (statusEl) statusEl.textContent = "Cancel error: " + e.message;
  }
}

/**
 * Auto-populate the image topic field when the bag selection changes.
 */
function _onExtractBagChange() {
  var sel = document.getElementById("extract-bag-key");
  var topicSel = document.getElementById("extract-image-topic-select");
  var topicHelp = document.getElementById("extract-topic-help");
  var topicInput = document.getElementById("extract-image-topic");
  var topicAdvanced = document.getElementById("extract-topic-advanced");
  var startBtn = document.getElementById("extract-start-btn");
  if (!sel || !topicSel) return;

  _empty(topicSel);
  var bag = _extractionBagsByKey[sel.value];
  if (!bag) {
    topicSel.appendChild(new Option("— select a bag first —", ""));
    if (topicHelp) topicHelp.textContent = "";
    if (startBtn) startBtn.disabled = false;
    return;
  }

  var topicRows = Array.isArray(bag.topic_details) ? bag.topic_details : [];
  var selectable = topicRows.filter(function(t) { return t && t.selectable; });
  var fallbackSelectable = [];
  if (selectable.length === 0 && bag.image_topics && bag.image_topics.length > 0) {
    fallbackSelectable = bag.image_topics.map(function(name) {
      return { name: name, type: (bag.topic_types || {})[name] || "sensor_msgs/msg/Image", selectable: true, modality: "image" };
    });
  }
  if (selectable.length === 0) selectable = fallbackSelectable;

  if (selectable.length === 0) {
    topicSel.appendChild(new Option("— no raw Image topics discovered —", ""));
    if (topicHelp) {
      var compressed = topicRows.some(function(t) { return t && t.type === "sensor_msgs/msg/CompressedImage"; });
      topicHelp.textContent = compressed
        ? "No directly compatible raw image topic. CompressedImage topics require decoding."
        : "No compatible raw image topic in metadata. Use Advanced manual override only for legacy/incomplete metadata.";
    }
    if (startBtn) startBtn.disabled = !(topicAdvanced && topicAdvanced.checked);
    if (topicInput) topicInput.value = "";
    return;
  }

  if (selectable.length > 1) {
    topicSel.appendChild(new Option("— select an image topic —", ""));
  }
  selectable.forEach(function(t) {
    var countTxt = (typeof t.message_count === "number") ? (" · " + t.message_count + " msgs") : "";
    var modality = t.modality ? (" · " + t.modality) : "";
    var label = t.name + " (" + (t.type || "unknown") + countTxt + modality + ")";
    var opt = new Option(label, t.name);
    if (selectable.length === 1) opt.selected = true;
    topicSel.appendChild(opt);
  });
  if (selectable.length === 1 && topicInput) {
    topicInput.value = selectable[0].name;
  }
  if (topicHelp) {
    topicHelp.textContent = selectable.length === 1
      ? "Auto-selected the only directly compatible raw image topic."
      : "Multiple compatible topics found. Please choose one.";
  }
  if (startBtn) startBtn.disabled = false;
}

function _onExtractTopicSelectionChange() {
  var topicSel = document.getElementById("extract-image-topic-select");
  var topicInput = document.getElementById("extract-image-topic");
  var topicAdvanced = document.getElementById("extract-topic-advanced");
  var startBtn = document.getElementById("extract-start-btn");
  if (!topicSel || !topicInput) return;
  if (topicSel.value) topicInput.value = topicSel.value;
  if (startBtn && !(topicAdvanced && topicAdvanced.checked)) {
    startBtn.disabled = !topicSel.value;
  }
}

function _onExtractTopicAdvancedToggle() {
  var advanced = document.getElementById("extract-topic-advanced");
  var input = document.getElementById("extract-image-topic");
  var sel = document.getElementById("extract-image-topic-select");
  var startBtn = document.getElementById("extract-start-btn");
  if (!advanced || !input || !sel) return;
  if (advanced.checked) {
    _show(input);
    if (!input.value && sel.value) input.value = sel.value;
    if (startBtn) startBtn.disabled = false;
  } else {
    _hide(input);
    _onExtractTopicSelectionChange();
  }
}

/* ── Frame Explorer view ─────────────────────────────────────────────────── */

var _frameExplorerDatasetId = null;
var _frameExplorerFrames = [];
var _frameExplorerSelected = 0;

async function _loadFrameExplorer() {
  var listEl = document.getElementById("frame-dataset-list");
  if (!listEl) return;
  _empty(listEl);
  listEl.appendChild(_el("p", "muted", "Loading datasets…"));
  try {
    var data = await _apiGet("/api/frame-datasets");
    _empty(listEl);
    if (!data.datasets || data.datasets.length === 0) {
      listEl.appendChild(_el("p", "muted", "No frame datasets found. Extract frames from a rosbag to get started."));
      return;
    }
    data.datasets.forEach(function(ds) {
      var btn = _el("button", "dataset-btn");
      btn.appendChild(_el("span", "ds-id", ds.dataset_id.slice(0, 8) + "…"));
      btn.appendChild(_el("span", "ds-meta", " bag: " + (ds.bag_key || "?") + " · " + (ds.frame_count || "?") + " frames"));
      btn.addEventListener("click", function() { _openFrameDataset(ds.dataset_id); });
      listEl.appendChild(btn);
    });
  } catch (e) {
    _empty(listEl);
    listEl.appendChild(_el("p", "error-msg", "Failed to load datasets: " + e.message));
  }
}

async function _openFrameDataset(datasetId) {
  _frameExplorerDatasetId = datasetId;
  var previewEl = document.getElementById("frame-preview-area");
  var stripEl = document.getElementById("frame-thumbnail-strip");
  var metaEl = document.getElementById("frame-metadata");
  if (!previewEl || !stripEl) return;
  _empty(stripEl);
  _empty(previewEl);
  if (metaEl) metaEl.textContent = "Loading…";
  try {
    var manifest = await _apiGet("/api/frame-datasets/" + datasetId);
    _frameExplorerFrames = manifest.frames || [];
    _frameExplorerSelected = 0;
    var viewerEl = document.getElementById("frame-explorer-viewer");
    if (viewerEl) _show(viewerEl);
    _renderFrameStrip(datasetId, manifest.frames);
    if (manifest.frames && manifest.frames.length > 0) {
      _showFrame(datasetId, 0, manifest.frames[0]);
    }
  } catch (e) {
    previewEl.appendChild(_el("p", "error-msg", "Failed to load dataset: " + e.message));
  }
}

function _renderFrameStrip(datasetId, frames) {
  var stripEl = document.getElementById("frame-thumbnail-strip");
  if (!stripEl) return;
  _empty(stripEl);
  (frames || []).forEach(function(frame, idx) {
    var thumb = document.createElement("img");
    thumb.className = "frame-thumb" + (idx === 0 ? " selected" : "");
    thumb.src = "/api/frame-datasets/" + datasetId + "/frames/" + idx;
    thumb.alt = "Frame " + idx;
    thumb.title = "Frame " + idx + " @ " + (frame.timestamp_sec || "?") + "s";
    thumb.addEventListener("click", function() { _showFrame(datasetId, idx, frame); });
    stripEl.appendChild(thumb);
  });
}

function _showFrame(datasetId, idx, frame) {
  _frameExplorerSelected = idx;
  var previewEl = document.getElementById("frame-preview-area");
  var metaEl = document.getElementById("frame-metadata");
  if (!previewEl) return;
  _empty(previewEl);
  var img = document.createElement("img");
  img.className = "frame-preview-img";
  img.src = "/api/frame-datasets/" + datasetId + "/frames/" + idx;
  img.alt = "Frame " + idx;
  previewEl.appendChild(img);
  if (metaEl) {
    _empty(metaEl);
    metaEl.appendChild(_el("span", "meta-item", "Frame: " + idx));
    metaEl.appendChild(_el("span", "meta-item", " · Timestamp: " + (frame.timestamp_sec || "?") + "s"));
    if (frame.source_seq) metaEl.appendChild(_el("span", "meta-item", " · Seq: " + frame.source_seq));
  }
  // Update thumbnail selection.
  document.querySelectorAll(".frame-thumb").forEach(function(t, i) {
    _setClass(t, "selected", i === idx);
  });
}

function _framePrev() {
  if (!_frameExplorerDatasetId || _frameExplorerFrames.length === 0) return;
  var next = Math.max(0, _frameExplorerSelected - 1);
  _showFrame(_frameExplorerDatasetId, next, _frameExplorerFrames[next]);
}

function _frameNext() {
  if (!_frameExplorerDatasetId || _frameExplorerFrames.length === 0) return;
  var next = Math.min(_frameExplorerFrames.length - 1, _frameExplorerSelected + 1);
  _showFrame(_frameExplorerDatasetId, next, _frameExplorerFrames[next]);
}

/* ── Profiles view ───────────────────────────────────────────────────────── */

async function _loadProfiles() {
  var el = document.getElementById("profiles-list");
  if (!el) return;
  _empty(el);
  el.appendChild(_el("p", "muted", "Loading profiles…"));
  try {
    var data = await _apiGet("/api/profiles");
    _empty(el);
    if (!data.profiles || data.profiles.length === 0) {
      el.appendChild(_el("p", "muted", "No task profiles found."));
      return;
    }
    data.profiles.forEach(function(p) {
      var card = _el("div", "profile-card");
      card.appendChild(_el("h3", null, p.name + " v" + p.version));
      card.appendChild(_el("p", "profile-hash", "Hash: " + p.prompt_hash.slice(0, 16) + "…"));
      var pre = _el("pre", "profile-prompt");
      pre.textContent = p.task_prompt;
      card.appendChild(pre);
      el.appendChild(card);
    });
  } catch (e) {
    _empty(el);
    el.appendChild(_el("p", "error-msg", "Failed to load profiles: " + e.message));
  }
}

/* ── Compare view ────────────────────────────────────────────────────────── */

async function _loadCompare() {
  var formEl = document.getElementById("compare-form");
  var resultsEl = document.getElementById("compare-results");
  if (!formEl || !resultsEl) return;
  _empty(resultsEl);
  resultsEl.appendChild(_el("p", "muted", "Enter two or more run IDs above and click Compare."));
}

async function _runCompare() {
  var input1 = document.getElementById("compare-run-id-1");
  var input2 = document.getElementById("compare-run-id-2");
  var resultsEl = document.getElementById("compare-results");
  if (!input1 || !input2 || !resultsEl) return;
  var ids = [input1.value.trim(), input2.value.trim()].filter(Boolean);
  if (ids.length < 2) {
    _empty(resultsEl);
    resultsEl.appendChild(_el("p", "error-msg", "Please enter at least two run IDs."));
    return;
  }
  _empty(resultsEl);
  resultsEl.appendChild(_el("p", "muted", "Comparing…"));
  try {
    var data = await _apiGet("/api/compare?run_ids=" + ids.join(","));
    _empty(resultsEl);
    _renderCompareResults(data, resultsEl, ids);
  } catch (e) {
    _empty(resultsEl);
    resultsEl.appendChild(_el("p", "error-msg", "Compare failed: " + e.message));
  }
}

function _renderCompareResults(data, container, runIds) {
  // Summary headers.
  var summaryDiv = _el("div", "compare-summary");
  runIds.forEach(function(rid) {
    var s = data.summaries && data.summaries[rid];
    if (!s) return;
    var card = _el("div", "compare-summary-card");
    card.appendChild(_el("h4", null, rid.slice(0, 8) + "…"));
    card.appendChild(_el("p", null, "Model: " + (s.model || "?")));
    card.appendChild(_el("p", null, "Strategy: " + (s.strategy || s.kind || "?")));
    card.appendChild(_el("p", null, "Status: " + (s.status || "?")));
    summaryDiv.appendChild(card);
  });
  container.appendChild(summaryDiv);

  // Aligned frames table.
  var frames = data.aligned_frames || [];
  if (frames.length === 0) {
    container.appendChild(_el("p", "muted", "No aligned frames found."));
    return;
  }
  var table = document.createElement("table");
  table.className = "compare-table";
  var thead = document.createElement("thead");
  var hRow = document.createElement("tr");
  hRow.appendChild(_el("th", null, "Frame"));
  runIds.forEach(function(rid) {
    hRow.appendChild(_el("th", null, rid.slice(0, 8) + "…"));
  });
  thead.appendChild(hRow);
  table.appendChild(thead);
  var tbody = document.createElement("tbody");
  frames.forEach(function(row) {
    var tr = document.createElement("tr");
    tr.appendChild(_el("td", null, String(row.frame_key)));
    runIds.forEach(function(rid) {
      var cell = _el("td", null);
      var fr = row[rid];
      if (fr) {
        var txt = _el("pre", "compare-cell-text");
        txt.textContent = fr.text || "(no text)";
        var lat = _el("span", "compare-latency", fr.latency_ms ? fr.latency_ms + "ms" : "");
        _append(cell, txt, lat);
      } else {
        cell.appendChild(_el("span", "muted", "—"));
      }
      tr.appendChild(cell);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  container.appendChild(table);
}

/* ── Review annotations ──────────────────────────────────────────────────── */

async function _submitReview(runId, frameIndex, label, note) {
  return _apiPost("/api/runs/" + runId + "/reviews", {
    frame_index: frameIndex,
    label: label,
    note: note || "",
  });
}

async function _loadReviewsForRun(runId, containerEl) {
  if (!containerEl) return;
  try {
    var data = await _apiGet("/api/runs/" + runId + "/reviews");
    _empty(containerEl);
    if (!data.reviews || data.reviews.length === 0) {
      containerEl.appendChild(_el("p", "muted", "No reviews yet."));
      return;
    }
    data.reviews.forEach(function(r) {
      var row = _el("div", "review-row");
      row.appendChild(_el("span", "review-frame", "Frame " + r.frame_index));
      row.appendChild(_badge(r.label, "review-label-" + r.label.replace(/_/g, "-")));
      if (r.note) row.appendChild(_el("span", "review-note", r.note));
      containerEl.appendChild(row);
    });
  } catch (e) {
    containerEl.appendChild(_el("p", "error-msg", "Failed to load reviews: " + e.message));
  }
}

function _renderReviewUI(runId, frameIndex, containerEl) {
  if (!containerEl) return;
  _empty(containerEl);
  var labels = ["acceptable", "unsupported_hallucinated", "missed_important_detail", "ambiguous"];
  var noteInput = document.createElement("input");
  noteInput.type = "text";
  noteInput.className = "review-note-input";
  noteInput.placeholder = "Optional note…";
  containerEl.appendChild(_el("span", "review-label-prompt", "Annotate frame " + frameIndex + ": "));
  labels.forEach(function(lbl) {
    var btn = _el("button", "review-btn review-btn-" + lbl.replace(/_/g, "-"), lbl.replace(/_/g, " "));
    btn.addEventListener("click", async function() {
      try {
        await _submitReview(runId, frameIndex, lbl, noteInput.value);
        btn.classList.add("review-saved");
        btn.textContent = "✓ " + lbl.replace(/_/g, " ");
      } catch (e) {
        containerEl.appendChild(_el("span", "error-msg", "Save failed: " + e.message));
      }
    });
    containerEl.appendChild(btn);
  });
  containerEl.appendChild(noteInput);
}
