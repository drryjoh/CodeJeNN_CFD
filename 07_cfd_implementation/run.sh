#!/bin/bash
# python3 viscosity.py
python3 train.py
# python3 plotT.py
# python3 plotH2.py
rm -rf __pycache__
python3 \
    $HOME/code/codejenn/src/api-core/main.py \
    --input="./model/" \
    --output="." \
    --backend="tensorflow" \
    --bit=64
rm -rf .vscode/ ../../src/api-core/__pycache__ ../../src/dump_model/__pycache__
