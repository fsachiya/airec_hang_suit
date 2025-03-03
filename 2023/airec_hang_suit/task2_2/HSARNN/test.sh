#!/bin/bash

for file in `ls ./log/`; do
    echo "python3 ./bin/test.py ./log/${file}"
    python3 ./bin/test.py ./log/${file}/SARNN*.pth 0
done
