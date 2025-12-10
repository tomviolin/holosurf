#!/usr/bin/env bash

# This script sets up a Python virtual environment and installs required packages.
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --seed -p python3.13
. ./.venv/bin/activate
pip install uv
pip install -r requirements-torch.txt
pip install -r requirements.txt


