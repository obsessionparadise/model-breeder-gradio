@echo off
:: ===========================================================================
:: Model Breeder — user configuration (Windows)
::
:: Edit this file to customize how webui.bat launches the app. webui.bat
:: calls this file automatically every time you run it — you never need to
:: touch webui.bat itself.
:: ===========================================================================

:: Command-line arguments passed straight to app.py. Common ones:
::   --share                  create a public gradio.live link (works
::                            anywhere, no router/firewall setup needed)
::   --listen                 also bind to 0.0.0.0 so other devices on your
::                            LAN can reach it directly (phone, another PC)
::   --port 7861              use a different port than the default 7860
set COMMANDLINE_ARGS=

:: Examples:
:: set COMMANDLINE_ARGS=--share
:: set COMMANDLINE_ARGS=--listen --port 7861

:: Python launcher to use. Leave as "python" unless you have multiple
:: versions installed and need a specific one.
:: set PYTHON=python

:: Set to 0 to skip creating a virtual environment and install straight
:: into your system Python instead.
:: set USE_VENV=0

:: Virtual environment folder name (created next to this script).
:: set VENV_DIR=venv
