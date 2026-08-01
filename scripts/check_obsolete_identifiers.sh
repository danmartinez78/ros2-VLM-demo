#!/usr/bin/env bash
# check_obsolete_identifiers.sh
#
# Rejects any source, script, or documentation file that contains a generic
# infrastructure identifier from the cosmos_ros2_video_reasoner era.
#
# Narrow allowlist: files that may still reference old names for legitimate
# migration-context or historical-RCA reasons are listed per pattern below.
#
# Usage
# -----
#   bash scripts/check_obsolete_identifiers.sh
#
# Exit code: 0 if clean, 1 if any obsolete identifier is found.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# file extensions to scan
INCLUDE_GLOBS=(
  "*.cpp" "*.hpp" "*.h"
  "*.py" "*.sh"
  "*.md"
  "*.yaml" "*.yml" "*.xml"
  "CMakeLists.txt" "package.xml"
)

# Build the --include= arguments for grep
include_args=()
for g in "${INCLUDE_GLOBS[@]}"; do
  include_args+=("--include=${g}")
done

# ── patterns and their per-pattern allowlists ────────────────────────────────
#
# Format: each entry is "PATTERN|ALLOWLIST_FRAGMENT_1|ALLOWLIST_FRAGMENT_2|..."
# PATTERN is a grep -F fixed string.
# ALLOWLIST_FRAGMENT is a substring of the file path that exempts the file.
#
RULES=(
  # Generic COSMOS_* env var family renamed to EDGE_VLM_* — allowlisted only in
  # the migration section of deployment.md and the historical RCA document.
  "COSMOS_|docs/deployment.md|docs/thor-edge-llm-prefill-stall-rca.md"

  # Executable / ROS node name renamed from cosmos_reasoner to edge_vlm_ros_node
  "cosmos_reasoner|docs/deployment.md|docs/thor-edge-llm-prefill-stall-rca.md"

  # Class renamed from CosmosReasonerNode to VlmReasonerNode
  "CosmosReasonerNode"

  # Message renamed from VisionReasoningResult to VlmResult
  "VisionReasoningResult"

  # Topic renamed from /cosmos/reasoning to /vlm/result
  "/cosmos/reasoning"

  # Socket renamed from cosmos_edge_llm to edge_vlm
  "cosmos_edge_llm"

  # Executable/service renamed
  "cosmos_inference_worker"
  "cosmos_reasoning_cli"

  # CMake target renamed
  "cosmos_ipc_backend"

  # Package name — allowlisted in deployment.md (migration section) and RCA doc
  "cosmos_ros2_video_reasoner|docs/deployment.md|docs/thor-edge-llm-prefill-stall-rca.md"
)

failed=0

for rule in "${RULES[@]}"; do
  # Split pattern from allowlist fragments
  IFS='|' read -ra parts <<< "${rule}"
  pattern="${parts[0]}"
  allowlist=("${parts[@]:1}")

  # Grep for the pattern across the repo, excluding this script itself
  raw_matches=$(
    grep -rn --fixed-strings \
      "${include_args[@]}" \
      --exclude-dir=".git" \
      --exclude="check_obsolete_identifiers.sh" \
      "${pattern}" "${REPO_ROOT}" 2>/dev/null || true
  )

  if [[ -z "${raw_matches}" ]]; then
    continue
  fi

  # Filter out allowlisted paths
  filtered="${raw_matches}"
  for allow in "${allowlist[@]}"; do
    filtered=$(printf '%s\n' "${filtered}" | grep -v "${allow}" || true)
  done

  if [[ -n "${filtered}" ]]; then
    echo "FAIL: Obsolete identifier '${pattern}' found:"
    printf '%s\n' "${filtered}"
    echo ""
    failed=1
  fi
done

if [[ ${failed} -ne 0 ]]; then
  echo "Remove the above obsolete identifiers before merging."
  echo "See the PR description for the canonical rename table."
  exit 1
fi

echo "OK: no obsolete generic infrastructure identifiers found."
