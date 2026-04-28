#!/usr/bin/env bash
# line_count.sh
# -----------------------------------------------------------------------
# pgkg pitch: minimal "non-Postgres" footprint.
# This script shows how much logic lives in Python (app) vs SQL (Postgres).
# The goal: SQL does the heavy lifting; Python is just glue.
# -----------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Optional color output (falls back gracefully if tput unavailable)
if command -v tput &>/dev/null && tput setaf 1 &>/dev/null; then
    BOLD=$(tput bold)
    CYAN=$(tput setaf 6)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    RESET=$(tput sgr0)
else
    BOLD="" CYAN="" GREEN="" YELLOW="" RESET=""
fi

# print_section DIR EXT LABEL
# Prints per-file line counts and returns the total via global SECTION_TOTAL
print_section() {
    local dir="$1"
    local ext="$2"
    local label="$3"

    echo "${BOLD}${CYAN}${label}${RESET} (${dir#${ROOT}/})"

    if [ ! -d "${dir}" ]; then
        echo "  (directory not found — skipping)"
        echo ""
        SECTION_TOTAL=0
        return
    fi

    local files
    files=$(find "${dir}" -name "*.${ext}" 2>/dev/null | sort)

    if [ -z "${files}" ]; then
        echo "  (empty)"
        echo ""
        SECTION_TOTAL=0
        return
    fi

    local total=0
    while IFS= read -r f; do
        local rel="${f#${ROOT}/}"
        local lines
        lines=$(wc -l < "${f}" | tr -d ' ')
        printf "  %-60s %5d lines\n" "${rel}" "${lines}"
        total=$((total + lines))
    done <<< "${files}"

    echo "${YELLOW}  Total: ${total} lines${RESET}"
    echo ""
    SECTION_TOTAL=${total}
}

echo ""
echo "${BOLD}pgkg — Non-Postgres code (Python) vs Postgres code (SQL)${RESET}"
echo "================================================================"
echo "  Pitch: SQL does the heavy lifting inside Postgres."
echo "  Python is intentionally thin — routing, validation, glue."
echo "================================================================"
echo ""

print_section "${ROOT}/pgkg" "py" "Python source (pgkg/)"
python_lines=${SECTION_TOTAL}

print_section "${ROOT}/migrations" "sql" "SQL migrations (migrations/)"
sql_lines=${SECTION_TOTAL}

print_section "${ROOT}/bench" "py" "Bench harness (bench/)"
bench_lines=${SECTION_TOTAL}

print_section "${ROOT}/tests" "py" "Tests (tests/)"
test_lines=${SECTION_TOTAL}

echo "================================================================"
printf "${GREEN}  Python total (pgkg/) : %5d lines${RESET}\n" "${python_lines}"
printf "${GREEN}  SQL total            : %5d lines${RESET}\n" "${sql_lines}"
printf "${GREEN}  Bench harness        : %5d lines${RESET}\n" "${bench_lines}"
printf "${GREEN}  Tests                : %5d lines${RESET}\n" "${test_lines}"

if [ "${python_lines}" -gt 0 ] && [ "${sql_lines}" -gt 0 ]; then
    ratio=$(echo "scale=2; ${sql_lines} / ${python_lines}" | bc 2>/dev/null || echo "n/a")
    echo "  SQL:Python ratio = ${ratio}:1"
fi
echo "================================================================"
echo ""
