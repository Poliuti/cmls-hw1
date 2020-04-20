#!/bin/sh
pipenv run jupyter nbextension enable --py widgetsnbextension --sys-prefix
pipenv run editor
