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
  else if (viewId === "datasets") { _loadDatasets(); }
  else if (viewId === "runs") { _loadRuns(); }
  else if (viewId === "diagnostics") { _loadDiagnostics(); }
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
    if (bag.topics && bag.topics.length > 0) {
      tile.appendChild(_el("div", "tile-meta", "Topics: " + bag.topics.join(", ")));
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
    var useBtn = _el("button", "small secondary", "Use in Experiment");
    useBtn.addEventListener("click", function() {
      navigate("experiment");
      setTimeout(function() {
        var bagPathEl = document.getElementById("exp-bag-path");
        if (bagPathEl) bagPathEl.value = bag.local_path;
      }, 50);
    });
    actions.appendChild(useBtn);
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
async function submitExperiment() {
  var btn = document.getElementById("exp-submit-btn");
  var resultEl = document.getElementById("exp-result");
  if (btn) btn.disabled = true;
  _empty(resultEl);

  var pathsRaw = (document.getElementById("exp-image-paths") || {}).value || "";
  var imagePaths = pathsRaw.split("\n")
    .map(function(s) { return s.trim(); })
    .filter(function(s) { return s.length > 0; });

  var params = {
    strategy: (document.getElementById("exp-strategy") || {}).value || "single_frame",
    image_paths: imagePaths,
    task_prompt: ((document.getElementById("exp-prompt") || {}).value || "").trim(),
    system_instruction: ((document.getElementById("exp-system") || {}).value || "").trim(),
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

  try {
    var result = await _apiPost("/api/experiment/run", params);
    if (result.status >= 400) {
      _append(resultEl,
        _el("div", "alert error", "Error: " + (result.body.error || result.status)));
    } else {
      var runId = result.body.run_id;
      _append(resultEl,
        _el("div", "alert success",
          "Experiment started (run_id: " + runId + ").\n" +
          "Results will appear in the Runs view when complete."));
      _pollExperimentResult(runId, resultEl);
    }
  } catch (e) {
    _append(resultEl, _el("div", "alert error", "Error: " + e.message));
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function _pollExperimentResult(runId, statusEl) {
  var maxPolls = 300;
  var count = 0;
  while (count++ < maxPolls) {
    await new Promise(function(r) { setTimeout(r, 2000); });
    try {
      var data = await _apiGet("/api/runs/" + runId);
      if (data.status === "completed" || data.status === "failed") {
        _empty(statusEl);
        if (data.status === "completed") {
          _append(statusEl, _el("div", "alert success",
            "Experiment complete! Showing results…"));
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
    el.appendChild(_el("div", "panel-title", "Benchmark Summary"));
    el.appendChild(bsEl);
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
