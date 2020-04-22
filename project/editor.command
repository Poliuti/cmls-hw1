#!/bin/sh
cd "`dirname "$0"`"
python -m pipenv run jupyter nbextension enable --py widgetsnbextension --sys-prefix
python -m pipenv run editor
