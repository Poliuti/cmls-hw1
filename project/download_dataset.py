from requests import get
from shutil import copyfileobj
from zipfile import ZipFile
from os import unlink

URL  = "https://polimi365-my.sharepoint.com/:u:/g/personal/10768481_polimi_it/ET_EMOV_tgBAm2yIQn4m4h0B8FxvxcDCJkpedf_3SRtLWw?download=1"
NAME = "dataset.zip"

## download
print("downloading...")
with get(URL, stream=True) as r:
	with open(NAME, 'wb') as f:
		copyfileobj(r.raw, f)

## unzip
print("unzipping...")
with ZipFile(NAME) as z:
	z.extractall()

## delete zip
unlink(NAME)

print("DONE!")
