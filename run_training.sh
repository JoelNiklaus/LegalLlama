#!/bin/bash

# Make sure the ephemeral disk is writable
sudo chmod 777 /ephemeral

# Load environment variables
set -a
source .env
set +a

python3 train.py