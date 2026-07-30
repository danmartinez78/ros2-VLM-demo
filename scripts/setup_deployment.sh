#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${script_dir}/install_dependencies.sh" "$@"

if [[ ! -f "${script_dir}/cosmos_env.sh" ]]; then
  cp "${script_dir}/cosmos_env.sh.example" "${script_dir}/cosmos_env.sh"
  echo
  echo "Created ${script_dir}/cosmos_env.sh"
  echo "Review its model and engine paths, then run:"
  echo "  ${script_dir}/build_workspace.sh"
  exit 0
fi

"${script_dir}/build_workspace.sh"
"${script_dir}/verify_deployment.sh"
