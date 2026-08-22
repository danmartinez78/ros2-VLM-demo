#!/usr/bin/env bash

source_ros_setup_nounset_safe() {
  local setup_script="$1"
  local nounset_was_enabled=0

  [[ -f "${setup_script}" ]] || return 1

  case $- in
    *u*) nounset_was_enabled=1 ;;
  esac

  set +u
  # shellcheck disable=SC1090
  source "${setup_script}"
  local source_status=$?

  if [[ "${nounset_was_enabled}" -eq 1 ]]; then
    set -u
  fi

  return "${source_status}"
}
