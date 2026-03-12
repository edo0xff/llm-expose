# llm-expose Installation Script for Windows
# Run as Administrator: powershell -ExecutionPolicy Bypass -Command "iex (New-Object Net.WebClient).DownloadString('https://raw.githubusercontent.com/edo0xff/llm-expose/main/scripts/install-windows.ps1')"

# Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

# Colors
$colors = @{
    'header' = 'Cyan'
    'success' = 'Green'
    'error' = 'Red'
    'warning' = 'Yellow'
    'step' = 'Blue'
}

# Functions
function Write-Header {
    Write-Host ""
    Write-Host "═════════════════════════════════════" -ForegroundColor $colors['header']
    Write-Host "  llm-expose Installation Script" -ForegroundColor $colors['header']
    Write-Host "═════════════════════════════════════" -ForegroundColor $colors['header']
    Write-Host ""
}

function Write-Step {
    param([string]$Message)
    Write-Host "▶ $Message" -ForegroundColor $colors['step']
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor $colors['success']
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor $colors['error']
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor $colors['warning']
}

# Check if running as Administrator
function Test-Administrator {
    $user = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($user)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check Python installation
function Check-Python {
    Write-Step "Checking Python installation..."
    
    try {
        $pythonVersion = & python --version 2>&1
        $version = $pythonVersion -match '(\d+\.\d+\.\d+)' | ForEach-Object { $matches[1] }
        Write-Success "Python $version found"
        
        # Check version is 3.11+
        $parts = $version -split '\.'
        $major = [int]$parts[0]
        $minor = [int]$parts[1]
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
            Write-Error-Custom "Python 3.11 or higher is required (found $version)"
            exit 1
        }
        
        return $true
    } catch {
        Write-Error-Custom "Python is not installed or not in PATH"
        Write-Host ""
        Write-Host "Please install Python 3.11 or higher:"
        Write-Host "  Download from: https://www.python.org/downloads/"
        Write-Host "  Make sure to check 'Add Python to PATH' during installation"
        Write-Host ""
        exit 1
    }
}

# Check pip installation
function Check-Pip {
    Write-Step "Checking pip installation..."
    
    try {
        $pipVersion = & pip --version
        Write-Success "pip is available: $pipVersion"
        return $true
    } catch {
        Write-Error-Custom "pip is not available"
        exit 1
    }
}

# Check git installation (optional)
function Check-Git {
    Write-Step "Checking git installation..."
    
    try {
        $gitVersion = & git --version
        Write-Success "git is available: $gitVersion"
        return $true
    } catch {
        Write-Warning-Custom "git is not installed - you can only install from PyPI"
        return $false
    }
}

# Install from source
function Install-FromSource {
    Write-Step "Installing from source (GitHub)..."
    
    $installDir = "$env:USERPROFILE\.llm-expose"
    
    if (Test-Path $installDir) {
        Write-Warning-Custom "Installation directory already exists: $installDir"
        $overwrite = Read-Host "Overwrite? (y/N)"
        if ($overwrite -ne "y" -and $overwrite -ne "Y") {
            Write-Warning-Custom "Keeping existing installation"
            return
        }
        Remove-Item $installDir -Recurse -Force
    }
    
    Write-Step "Cloning repository..."
    & git clone https://github.com/edo0xff/llm-expose.git $installDir
    
    Push-Location $installDir
    
    Write-Step "Installing Python dependencies..."
    & pip install -e . --upgrade
    
    Pop-Location
    
    Write-Success "Installation complete!"
    Write-Host ""
    Write-Host "To get started, run:"
    Write-Host "  llm-expose --help"
}

# Install from PyPI
function Install-FromPyPI {
    Write-Step "Installing from PyPI..."
    
    & pip install llm-expose --upgrade
    
    Write-Success "Installation complete!"
    Write-Host ""
    Write-Host "To get started, run:"
    Write-Host "  llm-expose --help"
}

# Verify installation
function Verify-Installation {
    Write-Step "Verifying installation..."
    
    try {
        $version = & llm-expose --version 2>$null
        Write-Success "llm-expose is installed (version: $version)"
    } catch {
        Write-Warning-Custom "llm-expose command not found"
        Write-Host "You may need to restart PowerShell or add the Python Scripts directory to your PATH"
    }
}

# Main
function Main {
    Write-Header
    
    if (-not (Test-Administrator)) {
        Write-Error-Custom "This script requires Administrator privileges"
        Write-Host "Please run PowerShell as Administrator"
        exit 1
    }
    
    Write-Success "Running as Administrator"
    Write-Host ""
    
    Check-Python
    Check-Pip
    Write-Host ""
    
    $hasGit = Check-Git
    Write-Host ""
    
    if ($hasGit) {
        $choice = Read-Host "Install from source (recommended for development) or PyPI? [source/pypi]"
        if ($choice -eq "source" -or $choice -eq "s") {
            Install-FromSource
        } else {
            Install-FromPyPI
        }
    } else {
        Install-FromPyPI
    }
    
    Write-Host ""
    Verify-Installation
    
    Write-Host ""
    Write-Host "Installation complete!" -ForegroundColor $colors['success']
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Read the documentation: https://edo0xff.github.io/llm-expose"
    Write-Host "  2. Configure a provider: llm-expose add model --help"
    Write-Host "  3. Set up a channel: llm-expose add channel --help"
    Write-Host ""
}

Main
