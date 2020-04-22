#!/bin/sh
cd "`dirname "$0"`"
python3 -m pipenv run jupyter nbextension enable --py widgetsnbextension --sys-prefix
python3 -m pipenv run editor
