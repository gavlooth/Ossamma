#!/bin/bash
#
# Vast.ai Setup Script for Ossamma Training
#
# Usage:
#   1. Rent a GPU on vast.ai (RTX 4090 or A100 recommended)
#   2. SSH into the instance
#   3. Run: curl -fsSL https://raw.githubusercontent.com/gavlooth/Ossamma/master/scripts/setup_vastai.sh | bash
#
# Or manually:
#   wget https://raw.githubusercontent.com/gavlooth/Ossamma/master/scripts/setup_vastai.sh
#   chmod +x setup_vastai.sh
#   ./setup_vastai.sh
#

set -e

echo "========================================"
echo "Ossamma Training Setup for Vast.ai"
echo "========================================"
echo ""

# Update system
echo "[1/5] Updating system..."
apt-get update -qq
apt-get install -y -qq git curl wget tmux htop

# Install Julia
echo "[2/5] Installing Julia 1.10.5..."
cd /tmp
wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.5-linux-x86_64.tar.gz
tar -xzf julia-1.10.5-linux-x86_64.tar.gz
mv julia-1.10.5 /opt/julia
ln -sf /opt/julia/bin/julia /usr/local/bin/julia
rm julia-1.10.5-linux-x86_64.tar.gz

echo "Julia version: $(julia --version)"

# Clone Ossamma
echo "[3/5] Cloning Ossamma..."
cd /workspace
git clone https://github.com/gavlooth/Ossamma.git
cd Ossamma

# Install Julia dependencies
echo "[4/5] Installing Julia dependencies..."
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Test trainability
echo "[5/5] Testing trainability..."
julia --project=. scripts/test_trainability.jl

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To start training:"
echo "  cd /workspace/Ossamma"
echo "  julia --project=. scripts/train_production.jl"
echo ""
echo "For long training, use tmux:"
echo "  tmux new -s train"
echo "  julia --project=. scripts/train_production.jl"
echo "  # Ctrl+B, D to detach"
echo "  # tmux attach -t train to reattach"
echo ""
