# Installation Scripts

Quick-start installation scripts for **llm-expose** on Linux, macOS, and Windows.

## Quick Install

### Linux & macOS

```bash
curl -fsSL https://raw.githubusercontent.com/edo0xff/llm-expose/main/scripts/install.sh | bash
```

### Windows (PowerShell)

Open PowerShell as Administrator and run:

```powershell
powershell -ExecutionPolicy Bypass -Command "iex (New-Object Net.WebClient).DownloadString('https://raw.githubusercontent.com/edo0xff/llm-expose/main/scripts/install-windows.ps1')"
```

Or using `curl` in PowerShell:

```powershell
curl -Uri "https://raw.githubusercontent.com/edo0xff/llm-expose/main/scripts/install-windows.ps1" -OutFile "install.ps1"; powershell -ExecutionPolicy Bypass -File "install.ps1"
```

## What These Scripts Do

### install.sh (Linux & macOS)

1. ✓ Detects your operating system
2. ✓ Checks for Python 3.11+ installation
3. ✓ Checks for pip availability
4. ✓ Optionally checks for git (for source installation)
5. ✓ Offers choice between source or PyPI installation
6. ✓ Installs dependencies
7. ✓ Verifies successful installation
8. ✓ Provides next steps

### install-windows.ps1 (Windows)

Same features as the Linux/macOS script, but:
- Uses PowerShell syntax
- Requires Administrator privileges
- Handles Windows path configuration

### install-universal.sh

A wrapper script that detects the OS and redirects to the appropriate installer.

## Requirements

- **Python**: 3.11 or higher
- **pip**: For package management
- **git** (optional): For source installation

## Features

- **Automatic OS detection**: Identifies Linux, macOS, or Windows
- **Dependency checking**: Verifies Python version and pip
- **Installation options**: Choose between GitHub source or PyPI package
- **Error handling**: Clear error messages and guidance
- **Colored output**: Easy-to-read installation progress
- **Verification**: Confirms successful installation

## Manual Installation

If you prefer not to use the automated scripts:

### From PyPI

```bash
pip install llm-expose
```

### From Source

```bash
git clone https://github.com/edo0xff/llm-expose.git
cd llm-expose
pip install -e .
```

### Development Setup

```bash
git clone https://github.com/edo0xff/llm-expose.git
cd llm-expose
pip install -e '.[dev]'
```

## Troubleshooting

### Python not found

Make sure Python 3.11+ is installed and in your PATH:

**Linux/macOS:**
```bash
python3 --version
```

**Windows:**
```powershell
python --version
```

### pip not found

Install pip or upgrade Python:

**Linux/macOS:**
```bash
sudo apt-get install python3-pip  # Debian/Ubuntu
brew install python               # macOS with Homebrew
```

**Windows:**
Download Python from https://www.python.org/downloads/ and ensure "Add Python to PATH" is checked.

### Permission denied (Linux/macOS)

Make the script executable:

```bash
chmod +x install.sh
./install.sh
```

### PowerShell execution policy (Windows)

If you see an execution policy error, run as Administrator and use:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope CurrentUser
```

## After Installation

Verify the installation:

```bash
llm-expose --version
llm-expose --help
```

For detailed setup instructions, see the [documentation](https://edo0xff.github.io/llm-expose).

## Support

If you encounter issues:

1. Check the [Troubleshooting Guide](https://edo0xff.github.io/llm-expose/guides/troubleshooting)
2. Open an issue on [GitHub](https://github.com/edo0xff/llm-expose/issues)
