# # Initialization

# ## Imports

# +
import numpy as np
import scipy as sp
import pandas as pd
import sklearn.linear_model

from tqdm import tqdm
from functools import lru_cache
from operator import itemgetter
from zipfile import ZipFile

import os
import urllib.request
# -

# ## Download Dataset

# +
DW_URL  = "https://polimi365-my.sharepoint.com/:u:/g/personal/10768481_polimi_it/ET_EMOV_tgBAm2yIQn4m4h0B8FxvxcDCJkpedf_3SRtLWw?download=1"
DW_PATH = "./dataset.zip"

if os.path.isdir(DW_PATH[:-4]):
    print("already downloaded.")
else:
    print("downloading...")
    urllib.request.urlretrieve(DW_URL, DW_PATH)
    print("unzipping...")
    with ZipFile(DW_PATH) as z:
        z.extractall()
    os.unlink(DW_PATH)
    print("DONE!")
# -

# # Model

# ## Extract Features

DATASET_PATH = "./dataset/"

# +
@lru_cache(maxsize=None)
def get_features(track_id):
    """returns a pandas matrix of averaged features for track `track_id`"""
    with open(os.path.join(DATASET_PATH, "features", f"{track_id}.csv")) as fin:
        feats = pd.read_csv(fin, header=0, index_col=0, sep=";", engine="c")
    #return pd.concat((feats.mean(), feats.std()), keys=["mean", "std"], axis=1, copy=False)
    # TODO: how to extract clip-level features from time-level features averaged with a moving average of 3?
    return feats.mean()

def get_all_features(length=None):
    """iterates over the dataset and extracts features of all tracks"""
    all_feats = []
    track_files = os.listdir(os.path.join(DATASET_PATH, "features/"))
    selected_tracks = sorted(map(lambda name: int(name.split(".")[0]), track_files))[:length]
    for track_id in tqdm(selected_tracks):
        feats = get_features(track_id)
        all_feats.append((track_id, feats))
    # NB: the upper limit is because we are only interested to the `2-2000` range.
    return pd.DataFrame(map(itemgetter(1), all_feats), index=map(itemgetter(0), all_feats)).loc[:2000]
# -

get_all_features(10)

# ## Extract Annotations

# +
@lru_cache(maxsize=None)
def get_annotations():
    """returns a pandas matrix of all the annotations of all tracks"""
    with open(os.path.join(DATASET_PATH, "annotations.csv")) as fin:
        return pd.read_csv(fin, header=0, index_col=0, sep=",\s*", engine="python")

def get_all_valence(length=None):
    """returns valence mean and std for every track"""
    return get_annotations().iloc[:length].loc[:, ["valence_mean", "valence_std"]]

def get_all_arousal(length=None):
    """returns arousal mean and std for every track"""
    return get_annotations().iloc[:length].loc[:, ["arousal_mean", "arousal_std"]]
# -

get_all_arousal(10)


# ## Regressor Training
# silly example to test

# +
l = 10
feats   = get_all_features(l)
arousal = get_all_arousal(l)
valence = get_all_valence(l)

## normalization
feats_m, feats_std = feats.mean(), feats.std()
feats_norm = (feats-feats_m)/feats_std

## regressor fitting
a_reg = sklearn.linear_model.LinearRegression()
a_reg.fit(feats_norm, arousal.loc[:, "arousal_mean"])
# -


