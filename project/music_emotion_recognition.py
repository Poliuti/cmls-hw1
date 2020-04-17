# -*- coding: utf-8 -*-
# # Initialization

# ## Imports

# +
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.model_selection
import sklearn.linear_model
import sklearn.svm

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
def feature_selector(name):
    if "_sma_de" in name:
        return False
    return True

@lru_cache(maxsize=None)
def get_features(track_id):
    """returns a pandas matrix of averaged features for track `track_id`"""
    with open(os.path.join(DATASET_PATH, "features", f"{track_id}.csv")) as fin:
        feats = pd.read_csv(fin, header=0, index_col=0, sep=";", engine="c")
    #return pd.concat((feats.mean(), feats.std()), keys=["mean", "std"], axis=1, copy=False)
    # TODO: how to extract clip-level features from time-level features averaged with a moving average of 3?
    return feats.loc[:, filter(feature_selector, feats.columns)].mean()

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

@lru_cache(maxsize=1)
def get_all_annotations(length=None):
    """returns a pandas matrix of all the annotations of all tracks"""
    with open(os.path.join(DATASET_PATH, "annotations.csv")) as fin:
        return pd.read_csv(fin, header=0, index_col=0, sep=",\s*", engine="python").iloc[:length]

get_all_annotations(10)


# ## Feature Visualization

# +
def frame_level_features(track_id):
    with open(os.path.join(DATASET_PATH, "features", f"{track_id}.csv")) as fin:
        return pd.read_csv(fin, header=0, index_col=0, sep=";", engine="c")

feats20 = frame_level_features(20)
F0 = feats20.loc[:, filter(lambda x: "F0final" in x, feats20.columns)]
F0
# -

plt.plot(F0)

# ### Valence characteristics

# +
tid_max_valence = get_all_annotations().loc[:, "valence_mean"].idxmax()
tid_min_valence = get_all_annotations().loc[:, "valence_mean"].idxmin()

pd.DataFrame((get_features(tid_max_valence), get_features(tid_min_valence)), index=["max", "min"])
# -

# ### Arousal characteristics

# +
tid_max_arousal = get_all_annotations().loc[:, "arousal_mean"].idxmax()
tid_min_arousal = get_all_annotations().loc[:, "arousal_mean"].idxmin()

pd.DataFrame((get_features(tid_max_arousal), get_features(tid_min_arousal)), index=["max", "min"])


# -

# # Regression

# +
def train_linear_regressor(features, annotations):
    reg = sklearn.linear_model.LinearRegression()
    reg.fit(features, annotations)
    return reg

def predict_regressor(reg, features):
    return pd.Series(reg.predict(features), features.index)
# -

# Extract N tracks from the dataset.

N       = 1000
feats   = get_all_features(N)
annots  = get_all_annotations(N)

# For SVM, the dataset should be normalized in order to have $\bar{X}=0$ and $\sigma_X=1$:

feats_m, feats_std = feats.mean(), feats.std()
feats_norm = (feats-feats_m)/feats_std

# Split the dataset in training set and testing set.

# +
(feats_train, feats_test,
 annots_train, annots_test) = sklearn.model_selection.train_test_split(feats_norm, annots)

print("Training set:", feats_train.index)
print("Testing set:", feats_test.index)
# -

# ### Linear Regression
# Using a linear regression for playing around.

# +
linear_predictions = pd.DataFrame()

for label in annots.columns:
    reg = train_linear_regressor(feats_train, annots_train.loc[:, label])
    pred = predict_regressor(reg, feats_test)
    pred.name = label
    linear_predictions = linear_predictions.join(pred, how="right")

linear_predictions
# -

# # Evaluation

def get_metrics(prediction, ground_truth):
    print("MSE:     ", sklearn.metrics.mean_squared_error(ground_truth, prediction))
    print("R² score:", sklearn.metrics.r2_score(ground_truth, prediction))


for label in annots.columns:
    print(f"=== metrics for {label} ===")
    get_metrics(linear_predictions.loc[:, label], annots_test.loc[:, label])
    print()
