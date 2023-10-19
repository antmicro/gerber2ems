#!/usr/bin/env bash

kmake pnp -v -e -o
kmake gerber -e
kmake stackup-export