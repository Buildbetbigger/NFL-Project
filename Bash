#!/bin/bash

# ============================================================================
# DFS Showdown Optimizer - Setup Script
# ============================================================================

set -e  # Exit on error

echo "================================================"
echo "DFS Showdown Optimizer - Setup"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# FUNCTIONS
# ============================================================================

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "ℹ $1"
}

# ============================================================================
# CHECK PYTHON VERSION
# ============================================================================

print_info "Checking Python version..."

if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    print_error "Python not found. Please install Python 3.9 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    print_error "Python 3.9+ required. Found: $PYTHON_VERSION"
    exit 1
fi

print_success "Python $PYTHON_VERSION found"

# ============================================================================
# CREATE VIRTUAL ENVIRONMENT
# ============================================================================

print_info "Creating virtual environment..."

if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Skipping creation."
else
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
fi

# ============================================================================
# ACTIVATE VIRTUAL ENVIRONMENT
# ============================================================================

print_info "Activating virtual environment..."

if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash)
    source venv/Scripts/activate
else
    # macOS / Linux
    source venv/bin/activate
fi

print_success "Virtual environment activated"

# ============================================================================
# UPGRADE PIP
# ============================================================================

print_info "Upgrading pip..."

pip install --upgrade pip setuptools wheel

print_success "Pip upgraded"

# ============================================================================
# INSTALL REQUIREMENTS
# ============================================================================

print_info "Installing requirements..."

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Requirements installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# ============================================================================
# CREATE DIRECTORY STRUCTURE
# ============================================================================

print_info "Creating directory structure..."

mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/uploads
mkdir -p lineups
mkdir -p logs
mkdir -p exports
mkdir -p tmp

# Create .gitkeep files
touch data/.gitkeep
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch lineups/.gitkeep
touch logs/.gitkeep

print_success "Directory structure created"

# ============================================================================
# CREATE .env FILE
# ============================================================================

if [ ! -f ".env" ]; then
    print_info "Creating .env file from template..."
    
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_success ".env file created"
        print_warning "Please edit .env file and add your API keys"
    else
        print_warning ".env.example not found. Creating basic .env..."
        cat > .env << EOF
# DFS Optimizer Environment Variables
# Add your API keys here (NEVER commit this file to Git!)

# Anthropic API Key
ANTHROPIC_API_KEY=your_api_key_here

# Optional Settings
DEBUG_MODE=false
LOG_LEVEL=INFO
EOF
        print_success "Basic .env file created"
        print_warning "Please edit .env file and add your API keys"
    fi
else
    print_warning ".env file already exists. Skipping creation."
fi

# ============================================================================
# VERIFY INSTALLATION
# ============================================================================

print_info "Verifying installation..."

$PYTHON_CMD -c "
import sys
print(f'Python: {sys.version}')

# Check core packages
packages = ['pandas', 'numpy', 'pulp', 'streamlit', 'anthropic']
for package in packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        print(f'✗ {package} - NOT FOUND')
        sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_success "All packages verified"
else
    print_error "Package verification failed"
    exit 1
fi

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "================================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env file and add your Anthropic API key"
echo "  2. Run: source venv/bin/activate  (or venv\\Scripts\\activate on Windows)"
echo "  3. Run: ./run.sh  (or bash run.sh)"
echo ""
echo "Or simply run: bash run.sh"
echo ""
