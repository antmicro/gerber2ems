#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
kmake pnp -v -e -o
kmake gerber -e
kmake stackup-export
$SCRIPT_DIR/main.py -g -d