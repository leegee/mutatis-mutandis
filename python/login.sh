#!/usr/bin/env bash

source macberth_env/Scripts/activate

python -m venv macberth_env

source macberth_env/Scripts/activate || source macberth_env/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt

source macberth_env/Scripts/activate
