# CMLS HW1 - Music Emotion Recognition

## Document

	cd document/

You can use any method of your choice to compile the LaTeX source, the file to be compiled is `main.tex`.
Built files will be ignored by `git`, please don't track them.

## Project

	cd project/

1. Install Pipenv:

		pip3 install pipenv

2. Initialize environment:

		pipenv sync

3. Download dataset & unpack:

		pipenv run python download_dataset.py

4. Run entry point:

		pipenv run python mer