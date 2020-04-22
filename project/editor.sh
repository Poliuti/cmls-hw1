#!/bin/sh
python -m pipenv run jupyter nbextension enable --py widgetsnbextension --sys-prefix
python -m pipenv run editor
