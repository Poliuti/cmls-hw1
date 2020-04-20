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

from tqdm.notebook import tqdm
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
def get_frame_level_features(track_id):
    """returns a pandas matrix of all frame-level features for `track_id`"""
    with open(os.path.join(DATASET_PATH, "features", f"{track_id}.csv")) as fin:
        return pd.read_csv(fin, header=0, index_col=0, sep=";", engine="c")

def get_clip_level_features(track_id):
    """converts frame-level features to relevant clip-level features"""
    # just mean everything for now, will be redefined later in the notebook
    sr = get_frame_level_features(track_id).mean()
    sr.name = track_id
    return sr

def get_features(selected_tracks=None, length=None):
    """iterates over the dataset and return a pandas matrix of features for all/selected tracks"""
    if selected_tracks is None:
        track_files = os.listdir(os.path.join(DATASET_PATH, "features/"))
        selected_tracks = sorted(map(lambda name: int(name.split(".")[0]), track_files))[:length]
    all_feats = (get_clip_level_features(track_id) for track_id in tqdm(selected_tracks))
    # NB: the upper limit is set because we are only interested to the `2-2000` range.
    return pd.DataFrame(all_feats).loc[:2000]
# -

get_features(length=100)

# ## Extract Annotations

def get_annotations(length=None):
    """returns a pandas matrix of all the annotations of all tracks"""
    with open(os.path.join(DATASET_PATH, "annotations.csv")) as fin:
        return pd.read_csv(fin, header=0, index_col=0, sep=",\s*", engine="python").iloc[:length]

get_annotations(50)


# ## Feature Visualization

# ### Annotations splitting

# +
maxs = dict()
mins = dict()

for label in get_annotations().columns:
    annot = get_annotations().loc[:, label]
    maxs[label] = annot.loc[annot >= annot.mean()].sort_values(ascending=False).index
    mins[label] = annot.loc[annot < annot.mean()].sort_values(ascending=True).index


# -

def plot_feature_evolution(tracks, feature_name, time_slice=slice(None)):
    data = pd.concat((
        get_frame_level_features(i).loc[time_slice, feature_name]
        for i in tqdm(tracks)
        ), axis=1)
    plt.xlabel("time")
    plt.ylabel(feature_name)
    plt.plot(data)


# ### Feature names

with open("features.txt") as fin:
    print(fin.read())

# ### Harmonicity

# Plotting Harmonicity evolution for the first 5 tracks with **maximum**/**minumum** valence.

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.title("tracks with min. valence mean")
plot_feature_evolution(mins["valence_mean"][:5], "pcm_fftMag_spectralHarmonicity_sma_amean", slice(10,50))
plt.subplot(1,2,2)
plt.title("tracks with max. valence mean")
plot_feature_evolution(maxs["valence_mean"][:5], "pcm_fftMag_spectralHarmonicity_sma_amean", slice(10,50))

# Plotting Harmonicity evolution for the first 5 tracks with **maximum**/**minumum** arousal.

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.title("tracks with min. arousal mean")
plot_feature_evolution(mins["arousal_mean"][:5], "pcm_fftMag_spectralHarmonicity_sma_amean", slice(10,50))
plt.subplot(1,2,2)
plt.title("tracks with max. arousal mean")
plot_feature_evolution(maxs["arousal_mean"][:5], "pcm_fftMag_spectralHarmonicity_sma_amean", slice(10,50))


# # Regression

# +
def train_linear_regressor(features, annotations):
    reg = sklearn.linear_model.LinearRegression()
    reg.fit(features, annotations)
    return reg

def train_svm_regressor(features, annotations):
    reg = sklearn.svm.SVR()
    reg.fit(features, annotations)
    return reg

def predict_regressor(reg, features):
    return pd.Series(reg.predict(features), features.index)
# -

# Extract N tracks from the dataset.

N       = 2000
feats   = get_features(length=N)
annots  = get_annotations(length=N)

# Split the dataset in training set and testing set.

# +
(feats_train, feats_test,
 annots_train, annots_test) = sklearn.model_selection.train_test_split(feats, annots)

print("Training set:", feats_train.index)
print("Testing set:", feats_test.index)
# -

# Normalize training dataset to have $\bar{X}=0$ and $\sigma_X=1$:

feats_m, feats_std = feats_train.mean(), feats_train.std()
feats_train = (feats_train-feats_m)/feats_std
feats_test  = (feats_test-feats_m)/feats_std

# ### Linear Regression

# +
linear_predictions = pd.DataFrame()

for label in annots.columns:
    reg = train_linear_regressor(feats_train, annots_train.loc[:, label])
    pred = predict_regressor(reg, feats_test)
    pred.name = label
    linear_predictions = linear_predictions.join(pred, how="right")

linear_predictions
# -

# ## SVM Regression

# +
svm_predictions = pd.DataFrame()

for label in annots.columns:
    reg = train_svm_regressor(feats_train, annots_train.loc[:, label])
    pred = predict_regressor(reg, feats_test)
    pred.name = label
    svm_predictions = svm_predictions.join(pred, how="right")

svm_predictions


# -

# # Evaluation

def get_metrics(prediction, ground_truth):
    print("MSE:     ", sklearn.metrics.mean_squared_error(ground_truth, prediction))
    print("R² score:", sklearn.metrics.r2_score(ground_truth, prediction))


# ## Metrics for linear regression

for label in annots.columns:
    print(f"=== metrics for {label} ===")
    get_metrics(linear_predictions.loc[:, label], annots_test.loc[:, label])
    print()

# ## Metrics for SVM regression

for label in annots.columns:
    print(f"=== metrics for {label} ===")
    get_metrics(svm_predictions.loc[:, label], annots_test.loc[:, label])
    print()


