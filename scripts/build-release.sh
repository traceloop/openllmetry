if [ "$(uname)" = "Darwin" ]; then export SEP=" "; else SEP=""; fi
VERSION=$(python -c "import tomllib; f=open('pyproject.toml','rb'); data=tomllib.load(f); f.close(); print(data['project']['version'])")
sed -i$SEP'' "s|{.*path.*|\"==$VERSION\"|" pyproject.toml
uv build
