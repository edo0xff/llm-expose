#!/bin/bash
# llm-expose Installation Script
# This script installs llm-expose on Linux and macOS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}═══════════════════════════════════${NC}"
    echo -e "${BLUE}  llm-expose Installation Script${NC}"
    echo -e "${BLUE}═══════════════════════════════════${NC}\n"
}

print_step() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        print_error "Unsupported OS: $OSTYPE"
        exit 1
    fi
}

# Check Python installation
check_python() {
    print_step "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        echo -e "\nPlease install Python 3.11 or higher:"
        if [[ "$OS" == "linux" ]]; then
            echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
            echo "  Fedora: sudo dnf install python3 python3-pip"
            echo "  Arch: sudo pacman -S python python-pip"
        elif [[ "$OS" == "macos" ]]; then
            echo "  Using Homebrew: brew install python@3.11"
            echo "  Or download from: https://www.python.org/downloads/"
        fi
        exit 1
    fi
    
    PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $PY_VERSION found"
    
    # Check if version is 3.11+
    PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
    PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)
    
    if [[ $PY_MAJOR -lt 3 ]] || [[ $PY_MAJOR -eq 3 && $PY_MINOR -lt 11 ]]; then
        print_error "Python 3.11 or higher is required (found $PY_VERSION)"
        exit 1
    fi
}

# Check pip installation
check_pip() {
    print_step "Checking pip installation..."
    
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed"
        exit 1
    fi
    
    pip3_version=$(pip3 --version)
    print_success "pip is available: $pip3_version"
}

# Check git installation
check_git() {
    print_step "Checking git installation..."
    
    if ! command -v git &> /dev/null; then
        print_warning "git is not installed - you can only install from PyPI"
        return 1
    fi
    
    print_success "git is available"
    return 0
}

# Clone and install from source
install_from_source() {
    print_step "Installing from source (GitHub)..."
    
    INSTALL_DIR="${HOME}/.llm-expose"
    
    if [ -d "$INSTALL_DIR" ]; then
        print_warning "Installation directory already exists: $INSTALL_DIR"
        read -p "Overwrite? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Keeping existing installation"
            return 0
        fi
        rm -rf "$INSTALL_DIR"
    fi
    
    print_step "Cloning repository..."
    git clone https://github.com/edo0xff/llm-expose.git "$INSTALL_DIR"
    
    cd "$INSTALL_DIR"
    
    print_step "Installing Python dependencies..."
    pip3 install -e . --upgrade
    
    print_success "Installation complete!"
    echo ""
    echo "To get started, run:"
    echo "  llm-expose --help"
}

# Install from PyPI
install_from_pypi() {
    print_step "Installing from PyPI..."
    
    pip3 install llm-expose --upgrade
    print_success "Installation complete!"
    echo ""
    echo "To get started, run:"
    echo "  llm-expose --help"
}

# Post-installation checks
verify_installation() {
    print_step "Verifying installation..."
    
    if ! command -v llm-expose &> /dev/null; then
        print_warning "llm-expose command not found in PATH"
        echo "You may need to restart your terminal or add ~/.local/bin to your PATH"
    else
        VERSION=$(llm-expose --version 2>/dev/null || echo "unknown")
        print_success "llm-expose is installed (version: $VERSION)"
    fi
}

# Main installation flow
main() {
    print_header
    
    detect_os
    print_success "Detected OS: $OS"
    echo ""
    
    check_python
    check_pip
    echo ""
    
    # Try to install from source if git is available
    if check_git; then
        read -p "Install from source (recommended for development) or PyPI? [source/pypi] " -r
        echo
        if [[ $REPLY =~ ^[Ss]$ ]]; then
            install_from_source
        else
            install_from_pypi
        fi
    else
        install_from_pypi
    fi
    
    echo ""
    verify_installation
    
    echo ""
    echo -e "${GREEN}Installation complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Read the documentation: https://edo0xff.github.io/llm-expose"
    echo "  2. Configure a provider: llm-expose add model --help"
    echo "  3. Set up a channel: llm-expose add channel --help"
    echo ""
}

main "$@"
