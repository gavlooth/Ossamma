#!/bin/bash
set -e

echo "========================================"
echo "Installing Claude Code CLI"
echo "========================================"

# Create non-root user for claude (root not allowed)
if [ "$(id -u)" = "0" ]; then
    echo "Running as root - creating claude user..."
    useradd -m -s /bin/bash claude 2>/dev/null || true

    # Install as claude user with proper HOME
    runuser -u claude -- bash -c 'cd ~ && curl -fsSL https://claude.ai/install.sh | bash'

    # Create wrapper script to run as claude user
    cat > /usr/local/bin/claude << 'WRAPPER'
#!/bin/bash
exec runuser -u claude -- /home/claude/.local/bin/claude "$@"
WRAPPER
    chmod +x /usr/local/bin/claude
else
    # Non-root install
    curl -fsSL https://claude.ai/install.sh | bash
fi

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "Run 'claude' to start"
