#!/usr/bin/env bash
###############################################################################
# Model Breeder — user configuration (Linux / macOS)
#
# Edit this file to customize how webui.sh launches the app. webui.sh
# sources this file automatically every time you run it — you never need
# to touch webui.sh itself.
###############################################################################

# Command-line arguments passed straight to app.py. Uncomment/combine as
# needed. Common ones:
#   --share                   create a public gradio.live link (works
#                             anywhere, including a machine with no open
#                             ports — this is what most people want)
#   --listen                  also bind to 0.0.0.0 so other devices on your
#                             LAN can reach it directly (phone, another PC)
#   --port 7861               use a different port than the default 7860
export COMMANDLINE_ARGS=""

# Examples:
# export COMMANDLINE_ARGS="--share"
# export COMMANDLINE_ARGS="--listen --port 7861"

# Python executable used to create the virtual environment. Override this
# if your system's default `python3` isn't the one you want (e.g. a
# specific version installed via pyenv/conda).
#export python_cmd="python3.11"

# Set to 0 to skip the virtual environment entirely and install straight
# into whatever `python3` already resolves to on your system. Useful
# inside an environment that's already isolated for you (a fresh Colab
# runtime, a container built just for this).
#export use_venv=0

# Virtual environment folder name (created next to this script).
#export venv_dir="venv"

# Set to 1 to force a full dependency reinstall even if requirements.txt
# hasn't changed since the last run. Normally not needed — webui.sh only
# reinstalls when requirements.txt actually changes.
#export force_reinstall=0
