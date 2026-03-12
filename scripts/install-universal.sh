#!/bin/bash
# llm-expose Universal Installation Script
# This script detects the OS and runs the appropriate installer
# Usage: curl -fsSL https://raw.githubusercontent.com/edo0xff/llm-expose/main/scripts/install.sh | bash

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo "Detected Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo "Detected macOS"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    OS="windows"
    echo "Detected Windows"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

if [[ "$OS" == "windows" ]]; then
    echo "For Windows, please run:"
    echo "  powershell -ExecutionPolicy Bypass -Command \"iex (New-Object Net.WebClient).DownloadString('https://raw.githubusercontent.com/edo0xff/llm-expose/main/scripts/install-windows.ps1')\""
    exit 1
fi

# Execute the Unix installer
bash -c "$(curl -fsSL https://raw.githubusercontent.com/edo0xff/llm-expose/main/scripts/install.sh)"
