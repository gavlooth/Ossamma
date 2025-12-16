#!/bin/bash
set -e

echo "========================================"
echo "Installing Bun + Claude Code CLI"
echo "========================================"

# Install bun
if ! command -v bun &> /dev/null; then
    echo "Installing bun..."
    curl -fsSL https://bun.sh/install | bash
    export BUN_INSTALL="$HOME/.bun"
    export PATH="$BUN_INSTALL/bin:$PATH"
fi

echo "Bun version: $(bun --version)"

# Install Claude Code CLI globally
echo "Installing Claude Code CLI..."
bun install -g @anthropic-ai/claude-code

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "Run 'source ~/.bashrc' or start a new shell"
echo "Then run 'claude' to start"
