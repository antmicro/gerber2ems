#!/usr/bin/env bash

# Create virtual enviroment
python3 -m venv --system-site-packages .venv > /dev/null 2>&1

# Install packages inside virtual enviroment
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
deactivate
wait

# Create ems-kicad.sh
WD=$(pwd)
echo '#!/usr/bin/env bash' > ems-kicad.sh
echo "$WD/.venv/bin/python3 $WD/src/main.py \"\$@\"">> ems-kicad.sh

# Make kmake.sh executable
chmod +x ems-kicad.sh

# Create ems-kicad command in "$HOME/.local/bin/"
mkdir --parents "$HOME/.local/bin/"
ln -sf "$(pwd)/ems-kicad.sh" "$HOME/.local/bin/ems-kicad"

echo "Installed successfully. Type ems-kicad everywhere you want"
