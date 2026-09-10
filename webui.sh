#!/usr/bin/env bash
###############################################################################
# Model Breeder — launcher (Linux / macOS)
#
# Mirrors the launcher pattern from AUTOMATIC1111/Forge (webui.sh) and
# ComfyUI's run scripts: creates a virtual environment on first run,
# installs dependencies only when requirements.txt has actually changed,
# creates the input/output workspace folders, then hands off to app.py
# with whatever flags you set in webui-user.sh.
#
# Customize launch behavior in webui-user.sh, not this file.
###############################################################################
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ---- pass-through flags just for this script (not app.py) -----------------
EXTRA_ARGS=()
FORCE_REINSTALL_FLAG=0
for arg in "$@"; do
    case "$arg" in
        --reinstall) FORCE_REINSTALL_FLAG=1 ;;
        --skip-venv) SKIP_VENV_FLAG=1 ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

# ---- load user overrides ---------------------------------------------------
if [[ -f webui-user.sh ]]; then
    # shellcheck disable=SC1091
    source webui-user.sh
fi

PYTHON="${python_cmd:-python3}"
VENV_DIR="${venv_dir:-venv}"
USE_VENV="${use_venv:-1}"
[[ "${SKIP_VENV_FLAG:-0}" == "1" ]] && USE_VENV=0
COMMANDLINE_ARGS="${COMMANDLINE_ARGS:-}"

echo "############################################################"
echo "  Model Breeder — Gradio Edition"
echo "############################################################"

# ---- sanity: python present -------------------------------------------
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: '$PYTHON' not found on PATH."
    echo "       Install Python 3.10+ and try again, or set python_cmd in"
    echo "       webui-user.sh to the right command for your system."
    exit 1
fi
echo "Python:  $("$PYTHON" --version 2>&1)"
echo "Workdir: $(pwd)"

# ---- sanity: this looks like a complete checkout -----------------------
# Checked here, before any venv creation or pip work, deliberately: a
# missing requirements.txt with --skip-venv used to get masked entirely by
# an unrelated "externally-managed-environment" error from pip's own PEP
# 668 guard (which fires before pip even looks at its arguments), and with
# the default venv path it wasted time creating a whole virtual
# environment before failing on the exact same missing file. Both
# reproduced directly before adding this check.
MISSING=()
for f in requirements.txt app.py setup_workspace.py; do
    [[ -f "$f" ]] || MISSING+=("$f")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: this doesn't look like a complete Model Breeder checkout —"
    echo "       missing: ${MISSING[*]}"
    echo "       Make sure you copied/cloned the whole project, not just"
    echo "       individual files, and that you're running this script"
    echo "       from inside that folder."
    exit 1
fi

# ---- virtual environment ---------------------------------------------
if [[ "$USE_VENV" == "1" ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "Creating virtual environment at ./$VENV_DIR ..."
        "$PYTHON" -m venv "$VENV_DIR" || {
            echo "ERROR: failed to create a virtual environment."
            echo "       On Debian/Ubuntu you may need: sudo apt install python3-venv"
            exit 1
        }
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PYTHON="python"
    echo "Virtual environment active: $VENV_DIR"
else
    echo "Skipping virtual environment (use_venv=0 or --skip-venv)"
fi

# ---- dependencies — only reinstall when requirements.txt changed ------
STATE_DIR="${VENV_DIR}"
[[ "$USE_VENV" == "1" ]] || STATE_DIR="."
mkdir -p "$STATE_DIR" 2>/dev/null || true
REQ_HASH_FILE="$STATE_DIR/.requirements_hash"

hash_requirements() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum requirements.txt 2>/dev/null | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 requirements.txt 2>/dev/null | awk '{print $1}'
    else
        # No hashing tool available — fall back to size+mtime as a
        # best-effort change signal rather than skipping the check outright.
        stat -c '%Y-%s' requirements.txt 2>/dev/null || stat -f '%m-%z' requirements.txt 2>/dev/null
    fi
}

NEW_HASH="$(hash_requirements)"
OLD_HASH="$(cat "$REQ_HASH_FILE" 2>/dev/null || true)"

if [[ -z "$NEW_HASH" || "$NEW_HASH" != "$OLD_HASH" || "${force_reinstall:-0}" == "1" || "$FORCE_REINSTALL_FLAG" == "1" ]]; then
    echo "Installing/updating dependencies (this can take a while the first time)..."
    "$PYTHON" -m pip install --upgrade pip --quiet 2>/dev/null || true
    INSTALL_LOG="$(mktemp)"
    if ! "$PYTHON" -m pip install -r requirements.txt 2>&1 | tee "$INSTALL_LOG"; then
        if [[ "$USE_VENV" != "1" ]] && grep -q "externally-managed-environment" "$INSTALL_LOG"; then
            # Only relevant outside a venv: modern Debian/Ubuntu (PEP 668)
            # refuse to let pip touch the system Python at all without this
            # flag. A venv is exempt from that restriction entirely, which
            # is exactly why it's the default — this fallback only matters
            # for someone who explicitly chose --skip-venv/use_venv=0.
            echo "  (externally-managed system Python detected — retrying with"
            echo "   --break-system-packages; use a virtual environment instead to avoid this)"
            if ! "$PYTHON" -m pip install -r requirements.txt --break-system-packages; then
                echo "ERROR: dependency installation failed even with --break-system-packages."
                rm -f "$INSTALL_LOG"
                exit 1
            fi
        else
            echo "ERROR: dependency installation failed — see the pip output above."
            rm -f "$INSTALL_LOG"
            exit 1
        fi
    fi
    rm -f "$INSTALL_LOG"
    echo "$NEW_HASH" > "$REQ_HASH_FILE" 2>/dev/null || true
else
    echo "Dependencies already up to date — skipping install."
    echo "(run ./webui.sh --reinstall to force a reinstall)"
fi

# ---- workspace folders --------------------------------------------------
"$PYTHON" setup_workspace.py

# ---- launch -----------------------------------------------------------
echo "############################################################"
echo "  Launching — press Ctrl+C to stop"
echo "############################################################"
# Guarding the array expansion rather than writing "${EXTRA_ARGS[@]}"
# directly: that idiom can raise "unbound variable" under `set -u` on
# bash 3.2 specifically (macOS's stock system bash — Apple never shipped
# a GPLv3 bash release, so this is still what `bash` resolves to on an
# unmodified Mac even today) when the array is declared but has zero
# elements, which is exactly the common case of running this script with
# no extra arguments. Checking the length first avoids the ambiguity
# entirely rather than assuming a bash version this script doesn't control.
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    exec "$PYTHON" app.py $COMMANDLINE_ARGS "${EXTRA_ARGS[@]}"
else
    exec "$PYTHON" app.py $COMMANDLINE_ARGS
fi
