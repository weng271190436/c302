#!/bin/bash

# Setup script for c302 (OpenWorm neural simulation)
# Run this from the c302 repository root

set -e  # Exit on error

echo "🐛 Setting up c302 virtual environment..."

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate it
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install c302 and dependencies
echo "📥 Installing c302 and dependencies (this may take a few minutes)..."
pip install .

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "    source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "    python -c \"import c302; print('c302 version:', c302.__version__)\""
echo ""
echo "🐛 Happy worm simulating!"
