/* Copyright 2025 edge_vlm_ros contributors */
/* Vanilla JS for the local web experiment console. No build step required. */

"use strict";

/* ── status ─────────────────────────────────────────────────────────────── */

async function refreshStatus() {
  const out = document.getElementById("status-out");
  const badge = document.getElementById("server-badge");
  out.textContent = "Loading…";
  try {
    const resp = await fetch("/api/status");
    const data = await resp.json();
    out.textContent = JSON.stringify(data, null, 2);
    const reachable = data.server && data.server.reachable;
    badge.textContent = reachable ? "server reachable" : "server unreachable";
    badge.className = "badge " + (reachable ? "ok" : "fail");
  } catch (e) {
    out.textContent = "Error: " + e;
    badge.textContent = "error";
    badge.className = "badge fail";
  }
}

/* ── standalone inference ────────────────────────────────────────────────── */

async function submitInfer() {
  const out = document.getElementById("infer-out");
  const form = document.getElementById("infer-form");
  out.textContent = "Running inference…";
  try {
    const fd = new FormData(form);
    const resp = await fetch("/api/infer", { method: "POST", body: fd });
    const data = await resp.json();
    if (resp.ok) {
      out.textContent =
        "Success (" + (data.inference_seconds || 0).toFixed(3) + " s)\n\n" +
        (data.text || "(empty response)") +
        "\n\n── full record ──\n" + JSON.stringify(data, null, 2);
    } else {
      out.textContent = "Error " + resp.status + ": " + (data.error || JSON.stringify(data));
    }
    refreshHistory();
  } catch (e) {
    out.textContent = "Network error: " + e;
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
    const resp = await fetch("/api/runs/" + runId + "/logs");
    const data = await resp.json();
    const el = document.getElementById("ros-logs");
    if (data.log_lines && data.log_lines.length > 0) {
      el.textContent = data.log_lines.join("\n");
      el.scrollTop = el.scrollHeight;
    }
  } catch (_) {}
}

async function pollLogs() {
  var runId = document.getElementById("ros-active-id").textContent;
  if (runId) _fetchLogs(runId);
}

async function startRos() {
  const out = document.getElementById("ros-out");
  out.textContent = "Starting ROS experiment…";
  const params = {
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
    const resp = await fetch("/api/ros/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params }),
    });
    const data = await resp.json();
    if (resp.ok || resp.status === 202) {
      out.textContent = "Started: " + JSON.stringify(data, null, 2);
      showActiveRun(data.run_id);
      refreshHistory();
    } else {
      out.textContent = "Error " + resp.status + ": " + (data.error || JSON.stringify(data));
    }
  } catch (e) {
    out.textContent = "Network error: " + e;
  }
}

async function stopRos() {
  const runId = document.getElementById("ros-active-id").textContent;
  if (!runId) return;
  const out = document.getElementById("ros-out");
  out.textContent = "Stopping ROS experiment…";
  try {
    const resp = await fetch("/api/ros/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runId }),
    });
    const data = await resp.json();
    out.textContent = JSON.stringify(data, null, 2);
    hideActiveRun();
    refreshHistory();
  } catch (e) {
    out.textContent = "Network error: " + e;
  }
}

/* ── run history ─────────────────────────────────────────────────────────── */

async function refreshHistory() {
  const el = document.getElementById("runs-table");
  try {
    const resp = await fetch("/api/runs");
    const data = await resp.json();
    if (!data.runs || data.runs.length === 0) {
      el.innerHTML = "<p class='muted'>No runs yet.</p>";
      return;
    }
    const rows = data.runs.map(function(r) {
      const id = r.run_id || "";
      const kind = r.kind || "";
      const created = r.created_at || "";
      const ok = r.success === true ? "<span class='badge ok'>OK</span>"
               : r.success === false ? "<span class='badge fail'>FAIL</span>"
               : "";
      return "<tr>" +
        "<td><a href='#' onclick=\"loadRun('" + id + "'); return false;\">" +
          id.substring(0, 8) + "…</a></td>" +
        "<td>" + kind + "</td>" +
        "<td>" + created + "</td>" +
        "<td>" + ok + "</td>" +
        "</tr>";
    }).join("");
    el.innerHTML = "<table class='runs'><thead><tr>" +
      "<th>Run</th><th>Kind</th><th>Created</th><th>Status</th>" +
      "</tr></thead><tbody>" + rows + "</tbody></table>";
  } catch (e) {
    el.innerHTML = "<p class='muted'>Failed to load history: " + e + "</p>";
  }
}

async function loadRun(runId) {
  const detailEl = document.getElementById("run-detail");
  const outEl = document.getElementById("run-detail-out");
  detailEl.style.display = "";
  outEl.textContent = "Loading…";
  try {
    const resp = await fetch("/api/runs/" + runId);
    const data = await resp.json();
    outEl.textContent = JSON.stringify(data, null, 2);
    detailEl.scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    outEl.textContent = "Error: " + e;
  }
}
