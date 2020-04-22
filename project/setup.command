#!/bin/sh
cd "`dirname "$0"`"
python -m pip install --user pipenv
python -m pipenv sync

