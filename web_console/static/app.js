/* Copyright 2025 edge_vlm_ros contributors */
/* Vanilla JS for the local web experiment console. No build step required. */

"use strict";

/* ── safe DOM helpers ────────────────────────────────────────────────────── */

/**
 * Create a DOM element, set optional class, and set textContent safely.
 * Use for all artifact-derived or user-supplied values to prevent XSS.
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

/* ── status ─────────────────────────────────────────────────────────────── */

async function refreshStatus() {
  var serviceEl = document.getElementById("service-details");
  var gpuEl = document.getElementById("gpu-details");
  var activeEl = document.getElementById("active-run-status");
  var badge = document.getElementById("server-badge");
  var rawOut = document.getElementById("status-out");

  try {
    var resp = await fetch("/api/status");
    var data = await resp.json();
    rawOut.textContent = JSON.stringify(data, null, 2);

    // Service card
    var srvInfo = data.server || {};
    var reachable = srvInfo.reachable;
    badge.textContent = reachable ? "server reachable" : "server unreachable";
    badge.className = "badge " + (reachable ? "ok" : "fail");

    serviceEl.textContent = "";
    serviceEl.appendChild(_badge(reachable ? "Reachable" : "Unreachable", reachable ? "ok" : "fail"));
    if (data.server_pid) {
      var pidLine = _el("div", "card-meta", "PID: " + data.server_pid);
      serviceEl.appendChild(pidLine);
    }
    var envInfo = data.env || {};
    if (envInfo.EDGE_VLM_LLM_ENGINE_DIR) {
      serviceEl.appendChild(_el("div", "card-meta", "LLM: " + envInfo.EDGE_VLM_LLM_ENGINE_DIR));
    }
    if (envInfo.EDGE_VLM_MULTIMODAL_ENGINE_DIR) {
      serviceEl.appendChild(_el("div", "card-meta", "MM: " + envInfo.EDGE_VLM_MULTIMODAL_ENGINE_DIR));
    }
    if (srvInfo.error && !reachable) {
      serviceEl.appendChild(_el("div", "card-meta muted", srvInfo.error));
    }

    // GPU card
    gpuEl.textContent = "";
    var gpuInfo = data.gpu || {};
    if (gpuInfo.available && gpuInfo.gpus && gpuInfo.gpus.length > 0) {
      gpuInfo.gpus.forEach(function(g) {
        var row = document.createElement("div");
        row.className = "card-meta";
        var namePart = _el("span", "gpu-name", g.name);
        row.appendChild(namePart);
        row.appendChild(document.createTextNode(
          " | Util: " + g.utilization_pct + "% | VRAM: " +
          g.memory_used_mib + "/" + g.memory_total_mib + " MiB"
        ));
        gpuEl.appendChild(row);
      });
    } else {
      gpuEl.appendChild(_el("span", "muted", gpuInfo.error || "No GPU detected"));
    }

    // Active ROS run card
    activeEl.textContent = "";
    var activeId = data.active_ros_run_id;
    if (activeId) {
      activeEl.appendChild(_badge("Running", "running"));
      activeEl.appendChild(_el("div", "card-meta", activeId));
    } else {
      activeEl.appendChild(_el("span", "muted", "None"));
    }
  } catch (e) {
    serviceEl.textContent = "Error: " + e;
    badge.textContent = "error";
    badge.className = "badge fail";
  }
}

/* ── standalone inference ────────────────────────────────────────────────── */

async function submitInfer() {
  var card = document.getElementById("infer-result-card");
  var rawDetails = document.getElementById("infer-raw-details");
  var rawOut = document.getElementById("infer-out");
  var form = document.getElementById("infer-form");

  card.textContent = "";
  card.style.display = "none";
  rawDetails.style.display = "none";

  var spinner = _el("div", "muted", "Running inference…");
  card.appendChild(spinner);
  card.style.display = "";

  try {
    var fd = new FormData(form);
    var resp = await fetch("/api/infer", { method: "POST", body: fd });
    var data = await resp.json();

    rawOut.textContent = JSON.stringify(data, null, 2);
    rawDetails.style.display = "";

    card.textContent = "";
    _renderInferResult(card, data, resp.ok);
    refreshHistory();
  } catch (e) {
    card.textContent = "";
    card.appendChild(_el("div", "badge fail", "Network error: " + e));
  }
}

function _renderInferResult(container, data, ok) {
  // Header row: badge + latency
  var header = document.createElement("div");
  header.className = "result-header";
  if (ok && data.success) {
    header.appendChild(_badge("Success", "ok"));
  } else {
    header.appendChild(_badge("Failed", "fail"));
  }
  var secs = data.inference_seconds;
  if (secs !== undefined) {
    header.appendChild(_el("span", "result-latency", secs.toFixed(3) + " s"));
  }
  container.appendChild(header);

  // Response text
  if (data.text) {
    container.appendChild(_el("div", "result-text", data.text));
  } else if (!ok) {
    container.appendChild(_el("div", "result-error", data.error || "No response text"));
  }

  // Concise settings row
  var parts = [];
  if (data.prompt) parts.push('prompt: "' + data.prompt + '"');
  if (data.max_generate_length !== undefined) parts.push("max_tokens: " + data.max_generate_length);
  if (data.temperature !== undefined) parts.push("temp: " + data.temperature);
  if (data.top_p !== undefined) parts.push("top-p: " + data.top_p);
  if (data.top_k !== undefined) parts.push("top-k: " + data.top_k);
  if (parts.length > 0) {
    container.appendChild(_el("div", "result-meta", parts.join(" | ")));
  }
}

/* ── ROS experiment ──────────────────────────────────────────────────────── */

var _rosLogInterval = null;

function showActiveRun(runId) {
  var banner = document.getElementById("ros-active");
  document.getElementById("ros-active-id").textContent = runId;
  banner.style.display = "";
  document.getElementById("ros-logs-area").style.display = "";
  _startLogPoll(runId);
}

function hideActiveRun() {
  document.getElementById("ros-active").style.display = "none";
  document.getElementById("ros-logs-area").style.display = "none";
  if (_rosLogInterval) { clearInterval(_rosLogInterval); _rosLogInterval = null; }
}

function _startLogPoll(runId) {
  if (_rosLogInterval) clearInterval(_rosLogInterval);
  _rosLogInterval = setInterval(function() { _fetchLogs(runId); }, 3000);
}

async function _fetchLogs(runId) {
  try {
    var resp = await fetch("/api/runs/" + runId + "/logs");
    var data = await resp.json();
    var el = document.getElementById("ros-logs");
    if (data.log_lines && data.log_lines.length > 0) {
      el.textContent = data.log_lines.join("\n");
      el.scrollTop = el.scrollHeight;
    }
    // Stop polling when the run has reached a terminal state.
    if (data.terminal) {
      if (_rosLogInterval) { clearInterval(_rosLogInterval); _rosLogInterval = null; }
      refreshHistory();
    }
  } catch (_) {}
}

async function pollLogs() {
  var runId = document.getElementById("ros-active-id").textContent;
  if (runId) _fetchLogs(runId);
}

async function startRos() {
  var startOut = document.getElementById("ros-start-out");
  var rawDetails = document.getElementById("ros-raw-details");
  var rawOut = document.getElementById("ros-out");
  startOut.style.display = "";
  startOut.textContent = "Starting ROS experiment…";
  rawDetails.style.display = "none";

  var params = {
    image_topic:                    document.getElementById("ros-topic").value,
    prompt:                         document.getElementById("ros-prompt").value,
    max_generate_length:            parseInt(document.getElementById("ros-max-gen").value, 10),
    instruction_delivery_mode:      document.getElementById("ros-delivery").value,
    observation_history_max_entries:parseInt(document.getElementById("ros-hist-entries").value, 10),
    observation_history_max_chars:  parseInt(document.getElementById("ros-hist-chars").value, 10),
    playback_duration:              parseInt(document.getElementById("ros-playback").value, 10),
    result_timeout:                 parseInt(document.getElementById("ros-timeout").value, 10),
    success_results_required:       parseInt(document.getElementById("ros-required").value, 10),
  };
  // Remove empty prompt to let the script use its default.
  if (!params.prompt) delete params.prompt;
  try {
    var resp = await fetch("/api/ros/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params: params }),
    });
    var data = await resp.json();
    rawOut.textContent = JSON.stringify(data, null, 2);
    rawDetails.style.display = "";
    if (resp.ok || resp.status === 202) {
      startOut.textContent = "Started run " + (data.run_id || "");
      var extWorker = data.external_worker;
      if (extWorker) {
        startOut.textContent += " (reusing external inference service)";
      }
      showActiveRun(data.run_id);
      refreshHistory();
    } else {
      startOut.textContent = "Error " + resp.status + ": " + (data.error || JSON.stringify(data));
    }
  } catch (e) {
    startOut.textContent = "Network error: " + e;
  }
}

async function stopRos() {
  var runId = document.getElementById("ros-active-id").textContent;
  if (!runId) return;
  var startOut = document.getElementById("ros-start-out");
  var rawOut = document.getElementById("ros-out");
  startOut.textContent = "Stopping ROS experiment…";
  startOut.style.display = "";
  try {
    var resp = await fetch("/api/ros/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runId }),
    });
    var data = await resp.json();
    rawOut.textContent = JSON.stringify(data, null, 2);
    startOut.textContent = "Stopped run " + runId;
    hideActiveRun();
    refreshHistory();
  } catch (e) {
    startOut.textContent = "Network error: " + e;
  }
}

/* ── run history ─────────────────────────────────────────────────────────── */

async function refreshHistory() {
  var el = document.getElementById("runs-table");
  try {
    var resp = await fetch("/api/runs");
    var data = await resp.json();
    if (!data.runs || data.runs.length === 0) {
      el.textContent = "";
      el.appendChild(_el("p", "muted", "No runs yet."));
      return;
    }

    var table = document.createElement("table");
    table.className = "runs";
    var thead = table.createTHead();
    var hrow = thead.insertRow();
    ["Run", "Kind", "Status", "Created", "Latency"].forEach(function(h) {
      var th = document.createElement("th");
      th.textContent = h;
      hrow.appendChild(th);
    });

    var tbody = table.createTBody();
    data.runs.forEach(function(r) {
      var row = tbody.insertRow();
      // Run ID link
      var idCell = row.insertCell();
      var link = document.createElement("a");
      link.href = "#";
      link.textContent = (r.run_id || "").substring(0, 8) + "…";
      link.addEventListener("click", function(e) {
        e.preventDefault();
        loadRun(r.run_id);
      });
      idCell.appendChild(link);
      // Kind
      row.insertCell().textContent = r.kind || "";
      // Status badge
      var statusCell = row.insertCell();
      if (r.success === true) {
        statusCell.appendChild(_badge("OK", "ok"));
      } else if (r.success === false) {
        statusCell.appendChild(_badge("FAIL", "fail"));
      } else if (r.status === "running" || r.status === "starting") {
        statusCell.appendChild(_badge("Running", "running"));
      }
      // Created (trim to seconds, replace T with space)
      var created = (r.created_at || "").substring(0, 19).replace("T", " ");
      row.insertCell().textContent = created;
      // Latency
      var latCell = row.insertCell();
      if (r.kind === "standalone" && r.inference_seconds !== undefined) {
        latCell.textContent = r.inference_seconds.toFixed(3) + " s";
      } else if (r.kind === "ros" && r.benchmark_summary && r.benchmark_summary.mean_inference_ms !== null) {
        latCell.textContent = r.benchmark_summary.mean_inference_ms.toFixed(0) + " ms";
      }
    });

    el.textContent = "";
    el.appendChild(table);
  } catch (e) {
    el.textContent = "";
    el.appendChild(_el("p", "muted", "Failed to load history: " + e));
  }
}

async function loadRun(runId) {
  var detailEl = document.getElementById("run-detail");
  var cardsEl = document.getElementById("run-detail-cards");
  var rawOut = document.getElementById("run-detail-out");
  detailEl.style.display = "";
  cardsEl.textContent = "";
  rawOut.textContent = "Loading…";
  try {
    var resp = await fetch("/api/runs/" + runId);
    var data = await resp.json();
    rawOut.textContent = JSON.stringify(data, null, 2);
    cardsEl.textContent = "";
    _renderRunDetail(cardsEl, data);
    detailEl.scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    cardsEl.appendChild(_el("p", "muted", "Error: " + e));
  }
}

function _renderRunDetail(container, data) {
  // Header: kind, status badge, run_id
  var hdr = document.createElement("div");
  hdr.className = "result-header";
  hdr.appendChild(_el("span", "run-kind", (data.kind || "unknown") + " run"));
  if (data.success === true) {
    hdr.appendChild(_badge("Success", "ok"));
  } else if (data.success === false) {
    hdr.appendChild(_badge("Failed", "fail"));
  } else if (data.status === "running" || data.status === "starting") {
    hdr.appendChild(_badge(data.status, "running"));
  }
  container.appendChild(hdr);

  if (data.kind === "standalone") {
    _renderStandaloneDetail(container, data);
  } else if (data.kind === "ros") {
    _renderRosDetail(container, data);
  }
}

function _renderStandaloneDetail(container, data) {
  // Latency + settings
  var meta = document.createElement("div");
  meta.className = "result-meta";
  if (data.inference_seconds !== undefined) {
    meta.appendChild(_el("span", "result-latency", data.inference_seconds.toFixed(3) + " s"));
  }
  var settingParts = [];
  if (data.max_generate_length !== undefined) settingParts.push("max_tokens: " + data.max_generate_length);
  if (data.temperature !== undefined) settingParts.push("temp: " + data.temperature);
  if (data.top_p !== undefined) settingParts.push("top-p: " + data.top_p);
  meta.appendChild(document.createTextNode(" " + settingParts.join(" | ")));
  container.appendChild(meta);

  // Prompt
  if (data.prompt) {
    var promptRow = document.createElement("div");
    promptRow.className = "result-meta";
    promptRow.appendChild(_el("span", "card-title", "Prompt: "));
    promptRow.appendChild(document.createTextNode(data.prompt));
    container.appendChild(promptRow);
  }

  // Response text
  if (data.text) {
    container.appendChild(_el("div", "result-text", data.text));
  } else if (data.error) {
    container.appendChild(_el("div", "result-error", data.error));
  }
}

function _renderRosDetail(container, data) {
  // Script manifest summary
  var sm = data.script_manifest;
  if (sm) {
    var smCard = document.createElement("div");
    smCard.className = "result-meta";
    var smParts = [];
    if (sm.successful_results_observed !== undefined) smParts.push("results: " + sm.successful_results_observed);
    if (sm.instruction_delivery_mode) smParts.push("mode: " + sm.instruction_delivery_mode);
    if (sm.max_generate_length) smParts.push("max_tokens: " + sm.max_generate_length);
    if (sm.git_commit) smParts.push("commit: " + sm.git_commit.substring(0, 8));
    smCard.textContent = smParts.join(" | ");
    container.appendChild(smCard);
  }

  // Benchmark summary
  var bs = data.benchmark_summary;
  if (bs) {
    var bsCard = document.createElement("div");
    bsCard.className = "result-meta";
    var bsParts = ["frames: " + bs.frame_count, "successful: " + bs.successful_frames];
    if (bs.mean_inference_ms !== null && bs.mean_inference_ms !== undefined) {
      bsParts.push("mean inference: " + bs.mean_inference_ms.toFixed(0) + " ms");
    }
    bsCard.textContent = bsParts.join(" | ");
    container.appendChild(bsCard);
  }

  // Result frames
  var frames = data.result_frames;
  if (frames && frames.length > 0) {
    var framesHdr = _el("div", "card-title", "Result Frames (" + frames.length + ")");
    container.appendChild(framesHdr);
    frames.forEach(function(frame, idx) {
      var fc = document.createElement("div");
      fc.className = "frame-card";
      // Frame header: index, success/fail badge
      var fhdr = document.createElement("div");
      fhdr.className = "result-header";
      fhdr.appendChild(_el("span", "card-title", "Frame " + (frame.frame_seq !== undefined ? frame.frame_seq : idx + 1)));
      if (frame.success === true) {
        fhdr.appendChild(_badge("OK", "ok"));
      } else if (frame.success === false) {
        fhdr.appendChild(_badge("FAIL", "fail"));
      }
      if (frame.latency_ms !== undefined) {
        fhdr.appendChild(_el("span", "result-latency", frame.latency_ms.toFixed(0) + " ms"));
      }
      if (frame.source_timestamp_ns !== undefined) {
        var ts = new Date(frame.source_timestamp_ns / 1e6);
        fhdr.appendChild(_el("span", "result-meta", ts.toISOString().replace("T", " ").replace("Z", " UTC")));
      }
      fc.appendChild(fhdr);
      if (frame.text) {
        fc.appendChild(_el("div", "result-text", frame.text));
      } else if (frame.error) {
        fc.appendChild(_el("div", "result-error", frame.error));
      }
      container.appendChild(fc);
    });
  }

  // Artifacts list
  var arts = data.artifacts;
  if (arts && arts.length > 0) {
    var artHdr = _el("div", "card-title", "Artifacts");
    container.appendChild(artHdr);
    var artList = document.createElement("ul");
    artList.className = "artifact-list";
    arts.forEach(function(p) {
      var li = document.createElement("li");
      li.textContent = p;
      artList.appendChild(li);
    });
    container.appendChild(artList);
  }

  // Log lines (collapsed)
  var logs = data.log_lines;
  if (logs && logs.length > 0) {
    var logDetails = document.createElement("details");
    logDetails.className = "raw-details";
    var logSummary = document.createElement("summary");
    logSummary.textContent = "Console log lines (" + logs.length + ")";
    logDetails.appendChild(logSummary);
    var logPre = document.createElement("pre");
    logPre.textContent = logs.join("\n");
    logDetails.appendChild(logPre);
    container.appendChild(logDetails);
  }
}

