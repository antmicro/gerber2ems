#!/usr/bin/env bash
ci_flag=
while getopts c name
do
    case $name in
    c) ci_flag=1;;
    ?) echo "Usage: $0: [-c]"
        exit 2;;
    esac
done

echo "This is ems-kicad installation script"

# Install python
if [[ -z "${ci_flag}" ]]; then
    sudo apt-get -qq install python3 python3-pip python3-venv
else
    apt-get -qqy install python3 python3-pip python3-venv
fi

# Create virtual enviroment
python3 -m venv --system-site-packages .venv > /dev/null 2>&1

# Install packages inside virtual enviroment
.venv/bin/python3 -m pip install --ignore-installed -r requirements.txt
wait




# Create ems-kicad.sh
WD=$(pwd)
echo '#!/usr/bin/env bash' > ems-kicad.sh
echo "$WD/.venv/bin/python3 $WD/main.py \"\$@\"">> ems-kicad.sh

# Make kmake.sh executable
chmod +x ems-kicad.sh

# Create ems-kicad command in "$HOME/.local/bin/"
mkdir --parents "$HOME/.local/bin/"
ln -sf "$(pwd)/ems-kicad.sh" "$HOME/.local/bin/ems-kicad"



echo "Installed successfully. Type ems-kicad everywhere you want"
