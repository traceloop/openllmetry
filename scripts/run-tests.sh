#!/usr/bin/env bash

# Examples:
# Run every discovered package test suite:
#   ./scripts/run-tests.sh
# Run only tests marked with `@pytest.mark.fr` across matching packages:
#   ./scripts/run-tests.sh --fr
# Run all tests for packages whose basename matches a glob:
#   ./scripts/run-tests.sh --package '*openai*'
# Run only tests marked with `@pytest.mark.fr` for a selected package glob:
#   ./scripts/run-tests.sh --fr --package '*anthropic*'
# List all discovered packages without running tests:
#   ./scripts/run-tests.sh --list
# Override the per-package test timeout:
#   PACKAGE_TEST_TIMEOUT_SECONDS=1800 ./scripts/run-tests.sh --fr
# Forward extra pytest arguments after `--`:
#   ./scripts/run-tests.sh --fr -- -x

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGES_DIR="$ROOT_DIR/packages"
VENV_DIR="$ROOT_DIR/.venv"
REPORTS_ROOT="$ROOT_DIR/reports/test-run"
TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
REPORT_DIR="$REPORTS_ROOT/$TIMESTAMP"
STATE_DIR="$REPORT_DIR/state"
PACKAGE_TEST_TIMEOUT_SECONDS="${PACKAGE_TEST_TIMEOUT_SECONDS:-1200}"

MODE="all"
PACKAGE_FILTER=""
LIST_ONLY=0
PYTEST_ARGS=()

INSTALLED_PACKAGES=()
PACKAGE_NAMES=()

usage() {
  cat <<'EOF'
Usage: scripts/run-tests.sh [options] [-- <extra pytest args>]

Options:
  --fr               Run only tests marked with @pytest.mark.fr
  --package <glob>   Restrict packages by basename glob, e.g. "*openai*"
  --list             List discovered packages and exit
  -h, --help         Show this help

Examples:
  scripts/run-tests.sh
  scripts/run-tests.sh --fr
  scripts/run-tests.sh --package "*openai*"
  PACKAGE_TEST_TIMEOUT_SECONDS=1800 scripts/run-tests.sh --fr
  scripts/run-tests.sh --fr -- -x
EOF
}

log() {
  printf '[run-tests] %s\n' "$*"
}

die() {
  printf '[run-tests] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

state_file() {
  local package_name="$1"
  local field="$2"
  local package_key
  package_key="$(printf '%s' "$package_name" | tr '/ ' '__')"
  printf '%s/%s.%s' "$STATE_DIR" "$package_key" "$field"
}

set_state() {
  local package_name="$1"
  local field="$2"
  local value="${3:-}"
  printf '%s' "$value" > "$(state_file "$package_name" "$field")"
}

get_state() {
  local package_name="$1"
  local field="$2"
  local file_path
  file_path="$(state_file "$package_name" "$field")"
  if [[ -f "$file_path" ]]; then
    cat "$file_path"
  fi
}

mark_package_seen() {
  local package_name="$1"
  local seen
  for seen in "${PACKAGE_NAMES[@]:-}"; do
    if [[ "$seen" == "$package_name" ]]; then
      return 0
    fi
  done
  PACKAGE_NAMES+=("$package_name")
}

bootstrap_venv() {
  require_cmd python3
  require_cmd poetry

  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log "Creating shared virtualenv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  export POETRY_VIRTUALENVS_CREATE=false
  export PIP_DISABLE_PIP_VERSION_CHECK=1

  log "Bootstrapping shared virtualenv"
  python -m pip install --upgrade pip setuptools wheel >/dev/null
}

discover_packages() {
  find "$PACKAGES_DIR" -mindepth 2 -maxdepth 2 -type f -name pyproject.toml -print \
    | sed 's#/pyproject.toml$##' \
    | sort
}

matches_package_filter() {
  local package_dir="$1"
  local package_name
  package_name="$(basename "$package_dir")"

  if [[ -z "$PACKAGE_FILTER" ]]; then
    return 0
  fi

  [[ "$package_name" == $PACKAGE_FILTER ]]
}

has_tests_directory() {
  local package_dir="$1"
  [[ -d "$package_dir/tests" ]]
}

has_marker_in_tests() {
  local package_dir="$1"
  local marker="$2"

  if command -v rg >/dev/null 2>&1; then
    rg -l "pytest\\.mark\\.${marker}" "$package_dir/tests" >/dev/null 2>&1
    return $?
  fi

  grep -RIl -E "pytest\\.mark\\.${marker}" "$package_dir/tests" >/dev/null 2>&1
}

python_meta() {
  local package_dir="$1"
  local key="$2"

  python3 - "$package_dir" "$key" <<'PY'
import json
import pathlib
import sys
import tomllib

package_dir = pathlib.Path(sys.argv[1]).resolve()
key = sys.argv[2]
data = tomllib.loads((package_dir / "pyproject.toml").read_text())
project = data.get("project", {})
groups = data.get("dependency-groups", {})
optional = project.get("optional-dependencies", {})
uv_sources = (((data.get("tool") or {}).get("uv") or {}).get("sources") or {})

if key == "name":
    print(project.get("name", package_dir.name))
elif key == "install_target":
    extras = []
    if "instruments" in optional:
        extras.append("instruments")
    suffix = f"[{','.join(extras)}]" if extras else ""
    print(f".{suffix}")
elif key == "local_paths":
    for source in uv_sources.values():
        if isinstance(source, dict) and "path" in source:
            print((package_dir / source["path"]).resolve())
elif key == "test_deps":
    for dep in groups.get("test", []):
        print(dep)
PY
}

install_package() {
  local package_dir="$1"
  local owner_package_name="${2:-}"
  local package_name
  package_name="$(python_meta "$package_dir" name)"
  if [[ -z "$owner_package_name" ]]; then
    owner_package_name="$package_name"
  fi

  local installed
  for installed in "${INSTALLED_PACKAGES[@]:-}"; do
    if [[ "$installed" == "$package_dir" ]]; then
      return 0
    fi
  done

  if [[ -f "$(state_file "$owner_package_name" "status")" ]] && [[ "$(get_state "$owner_package_name" "status")" == "INSTALL_FAIL" ]]; then
    return 0
  fi

  local local_dep
  while IFS= read -r local_dep; do
    [[ -n "$local_dep" ]] || continue
    install_package "$local_dep" "$owner_package_name"
  done < <(python_meta "$package_dir" local_paths)

  log "Installing dependencies for $package_name"

  local install_target
  install_target="$(python_meta "$package_dir" install_target)"

  local -a pip_args
  pip_args=(-e "$install_target")
  while IFS= read -r dep; do
    [[ -n "$dep" ]] || continue
    pip_args+=("$dep")
  done < <(python_meta "$package_dir" test_deps)

  local install_log="$REPORT_DIR/${package_name}.install.log"
  set +e
  (
    cd "$package_dir"
    poetry run python -m pip install "${pip_args[@]}"
  ) > >(tee "$install_log") 2>&1
  local install_status=$?
  set -e

  if [[ $install_status -ne 0 ]]; then
    set_state "$owner_package_name" "status" "INSTALL_FAIL"
    set_state "$owner_package_name" "reason" "Dependency installation failed while preparing $package_name. See $install_log"
    return 1
  fi

  INSTALLED_PACKAGES+=("$package_dir")
}

parse_junit_summary() {
  local junit_xml="$1"

  python3 - "$junit_xml" <<'PY'
import json
import pathlib
import sys
import xml.etree.ElementTree as ET

xml_path = pathlib.Path(sys.argv[1])
summary = {
    "tests": 0,
    "passed": 0,
    "failed": 0,
    "errors": 0,
    "skipped": 0,
    "failing_tests": [],
}

if not xml_path.exists():
    print(json.dumps(summary))
    raise SystemExit(0)

root = ET.parse(xml_path).getroot()
tests = list(root.iter("testcase"))
summary["tests"] = len(tests)

for case in tests:
    is_failed = case.find("failure") is not None
    is_error = case.find("error") is not None
    is_skipped = case.find("skipped") is not None
    if is_failed or is_error:
        summary["failed"] += int(is_failed)
        summary["errors"] += int(is_error)
        classname = case.attrib.get("classname", "").strip()
        name = case.attrib.get("name", "").strip()
        if classname and name:
            summary["failing_tests"].append(f"{classname}::{name}")
        else:
            summary["failing_tests"].append(name or classname or "<unknown>")
    elif is_skipped:
        summary["skipped"] += 1

summary["passed"] = summary["tests"] - summary["failed"] - summary["errors"] - summary["skipped"]
print(json.dumps(summary))
PY
}

run_with_timeout() {
  local timeout_seconds="$1"
  shift

  python3 - "$timeout_seconds" "$@" <<'PY'
import subprocess
import sys

timeout_seconds = int(sys.argv[1])
command = sys.argv[2:]

try:
    completed = subprocess.run(command, timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    print(
        f"[run-tests] ERROR: command timed out after {timeout_seconds} seconds",
        file=sys.stderr,
    )
    raise SystemExit(124)

raise SystemExit(completed.returncode)
PY
}

record_test_summary() {
  local package_name="$1"
  local junit_xml="$2"

  local summary_json
  summary_json="$(parse_junit_summary "$junit_xml")"

  set_state "$package_name" "counts" "$(python3 - "$summary_json" <<'PY'
import json
import sys
data = json.loads(sys.argv[1])
print(f"passed={data['passed']} failed={data['failed']} errors={data['errors']} skipped={data['skipped']} total={data['tests']}")
PY
)"

  set_state "$package_name" "failures" "$(python3 - "$summary_json" <<'PY'
import json
import sys
data = json.loads(sys.argv[1])
print("\n".join(data["failing_tests"]))
PY
)"
}

run_package_tests() {
  local package_dir="$1"
  local package_name
  package_name="$(python_meta "$package_dir" name)"
  mark_package_seen "$package_name"

  if ! has_tests_directory "$package_dir"; then
    set_state "$package_name" "status" "SKIP"
    set_state "$package_name" "reason" "No tests directory"
    return 0
  fi

  if [[ "$MODE" == "fr" ]] && ! has_marker_in_tests "$package_dir" "fr"; then
    set_state "$package_name" "status" "SKIP"
    set_state "$package_name" "reason" "No FR-marked tests detected"
    return 0
  fi

  if ! install_package "$package_dir" "$package_name"; then
    return 0
  fi

  local junit_xml="$REPORT_DIR/${package_name}.xml"
  local test_log="$REPORT_DIR/${package_name}.test.log"
  local -a pytest_cmd
  if [[ "$MODE" == "fr" ]]; then
    pytest_cmd=(poetry run pytest -q --junitxml "$junit_xml" tests)
    pytest_cmd+=(-m fr)
  else
    pytest_cmd=(poetry run pytest -q --junitxml "$junit_xml" tests)
  fi
  pytest_cmd+=("${PYTEST_ARGS[@]:-}")

  log "Running tests for $package_name (timeout=${PACKAGE_TEST_TIMEOUT_SECONDS}s)"
  set +e
  (
    cd "$package_dir"
    run_with_timeout "$PACKAGE_TEST_TIMEOUT_SECONDS" "${pytest_cmd[@]}"
  ) > >(tee "$test_log") 2>&1
  local test_status=$?
  set -e

  if [[ $test_status -eq 5 ]]; then
    set_state "$package_name" "status" "SKIP"
    set_state "$package_name" "reason" "Pytest collected no matching tests"
    return 0
  fi

  record_test_summary "$package_name" "$junit_xml"

  if [[ $test_status -eq 0 ]]; then
    set_state "$package_name" "status" "PASS"
  elif [[ $test_status -eq 124 ]]; then
    set_state "$package_name" "status" "FAIL"
    set_state "$package_name" "reason" "Timed out after ${PACKAGE_TEST_TIMEOUT_SECONDS}s. See $test_log"
  else
    set_state "$package_name" "status" "FAIL"
    set_state "$package_name" "reason" "Test failures detected. See $test_log"
  fi
}

print_report() {
  local total=0
  local passed=0
  local failed=0
  local skipped=0
  local install_failed=0
  local tests_total=0
  local tests_passed=0
  local tests_failed=0
  local tests_skipped=0
  local package_name
  local package_counts
  local counts_field
  local counts_value
  local package_passed
  local package_failed
  local package_errors
  local package_skipped
  local package_total
  local -a failing_lines=()
  local -a skipped_lines=()
  local -a executed_lines=()

  printf '\n=== Consolidated Test Report ===\n'
  printf 'Mode: %s\n' "$MODE"
  printf 'Reports: %s\n\n' "$REPORT_DIR"

  for package_name in "${PACKAGE_NAMES[@]:-}"; do
    total=$((total + 1))
    case "$(get_state "$package_name" "status")" in
      PASS)
        passed=$((passed + 1))
        package_counts="$(get_state "$package_name" "counts")"
        executed_lines+=("$(printf 'PASS  %-50s %s' "$package_name" "$package_counts")")
        package_passed=0
        package_failed=0
        package_errors=0
        package_skipped=0
        package_total=0
        for counts_field in $package_counts; do
          counts_value="${counts_field#*=}"
          case "$counts_field" in
            passed=*) package_passed="$counts_value" ;;
            failed=*) package_failed="$counts_value" ;;
            errors=*) package_errors="$counts_value" ;;
            skipped=*) package_skipped="$counts_value" ;;
            total=*) package_total="$counts_value" ;;
          esac
        done
        tests_total=$((tests_total + package_total))
        tests_passed=$((tests_passed + package_passed))
        tests_failed=$((tests_failed + package_failed + package_errors))
        tests_skipped=$((tests_skipped + package_skipped))
        ;;
      FAIL)
        failed=$((failed + 1))
        package_counts="$(get_state "$package_name" "counts")"
        executed_lines+=("$(printf 'FAIL  %-50s %s' "$package_name" "$package_counts")")
        package_passed=0
        package_failed=0
        package_errors=0
        package_skipped=0
        package_total=0
        for counts_field in $package_counts; do
          counts_value="${counts_field#*=}"
          case "$counts_field" in
            passed=*) package_passed="$counts_value" ;;
            failed=*) package_failed="$counts_value" ;;
            errors=*) package_errors="$counts_value" ;;
            skipped=*) package_skipped="$counts_value" ;;
            total=*) package_total="$counts_value" ;;
          esac
        done
        tests_total=$((tests_total + package_total))
        tests_passed=$((tests_passed + package_passed))
        tests_failed=$((tests_failed + package_failed + package_errors))
        tests_skipped=$((tests_skipped + package_skipped))
        if [[ -n "$(get_state "$package_name" "failures")" ]]; then
          while IFS= read -r failing_test; do
            [[ -n "$failing_test" ]] || continue
            failing_lines+=("$package_name :: $failing_test")
          done <<< "$(get_state "$package_name" "failures")"
        fi
        ;;
      INSTALL_FAIL)
        install_failed=$((install_failed + 1))
        executed_lines+=("$(printf 'ERROR %-50s %s' "$package_name" "$(get_state "$package_name" "reason")")")
        failing_lines+=("$package_name :: [install] $(get_state "$package_name" "reason")")
        ;;
      *)
        skipped=$((skipped + 1))
        skipped_lines+=("$(printf 'SKIP  %-50s %s' "$package_name" "$(get_state "$package_name" "reason")")")
        ;;
    esac
  done

  printf 'Skipped packages:\n'
  if (( ${#skipped_lines[@]} > 0 )); then
    printf '%s\n' "${skipped_lines[@]}"
  else
    printf ' - none\n'
  fi

  printf '\nExecuted packages:\n'
  if (( ${#executed_lines[@]} > 0 )); then
    printf '%s\n' "${executed_lines[@]}"
  else
    printf ' - none\n'
  fi

  printf '\nOverall tests run: total = %d passed = %d failed = %d skipped = %d\n' \
    "$tests_total" "$tests_passed" "$tests_failed" "$tests_skipped"

  printf '\nPackages: total=%d passed=%d failed=%d install_failed=%d skipped=%d\n' \
    "$total" "$passed" "$failed" "$install_failed" "$skipped"

  if [[ ${#failing_lines[@]} -gt 0 ]]; then
    printf '\nFailing tests:\n'
    printf ' - %s\n' "${failing_lines[@]}"
  else
    printf '\nFailing tests:\n'
    printf ' - none\n'
  fi

  printf '\n'

  if (( failed > 0 || install_failed > 0 )); then
    return 1
  fi
  return 0
}

main() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --fr)
        MODE="fr"
        shift
        ;;
      --package)
        [[ $# -ge 2 ]] || die "--package requires a glob argument"
        PACKAGE_FILTER="$2"
        shift 2
        ;;
      --list)
        LIST_ONLY=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        PYTEST_ARGS=("$@")
        break
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done

  mkdir -p "$REPORT_DIR"
  mkdir -p "$STATE_DIR"
  bootstrap_venv

  local -a package_dirs=()
  local package_dir
  while IFS= read -r package_dir; do
    [[ -n "$package_dir" ]] || continue
    if matches_package_filter "$package_dir"; then
      package_dirs+=("$package_dir")
    fi
  done < <(discover_packages)

  (( ${#package_dirs[@]} > 0 )) || die "No packages matched the requested filters"

  if (( LIST_ONLY == 1 )); then
    printf '%s\n' "${package_dirs[@]}"
    exit 0
  fi

  local package_name
  for package_dir in "${package_dirs[@]}"; do
    package_name="$(python_meta "$package_dir" name)"
    mark_package_seen "$package_name"
    set_state "$package_name" "status" "SKIP"
    set_state "$package_name" "reason" "Not executed"
  done

  for package_dir in "${package_dirs[@]}"; do
    run_package_tests "$package_dir"
  done

  print_report
}

main "$@"
