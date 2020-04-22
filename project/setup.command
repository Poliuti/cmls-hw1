#!/bin/sh
cd "`dirname "$0"`"
python3 -m pip install --user pipenv
python3 -m pipenv sync
