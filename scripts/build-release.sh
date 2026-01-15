#!/bin/bash
set -euo pipefail

# Remove [tool.uv.sources] section for release builds
# This ensures packages are fetched from PyPI instead of local paths
# Use Python to safely remove the TOML section
python3 << 'EOF'
import re

with open('pyproject.toml', 'r') as f:
    content = f.read()

# Remove [tool.uv.sources] section and all its entries
# Matches from [tool.uv.sources] until the next section header or end of file
content = re.sub(r'\n?\[tool\.uv\.sources\][^\[]*', '', content)

with open('pyproject.toml', 'w') as f:
    f.write(content)
EOF

# Build using uv (which uses hatchling as backend)
uv build
