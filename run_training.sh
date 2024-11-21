#!/bin/bash

# Load environment variables
set -a
source .env
set +a

python3 train.py