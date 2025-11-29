#!/usr/bin/env bash

python -m venv macberth_env

source macberth_env/Scripts/activate || source macberth_env/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt
