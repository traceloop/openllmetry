if [ "$(uname)" = "Darwin" ]; then export SEP=" "; else SEP=""; fi
VERSION=$(poetry version | awk '{print $2}')
sed -i$SEP'' "s|{.*path.*|\"==$VERSION\"|" pyproject.toml
poetry build
