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
import sklearn.neighbors
import sklearn.feature_selection

from tqdm.notebook import tqdm
from functools import lru_cache
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

# ## Other stuff

# +
RUNTIME_DIR = "./run/"

try:
    os.mkdir(RUNTIME_DIR)
except FileExistsError:
    pass
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
    # just mean everything for now (except deltas), might be redefined later in the notebook
    sr = get_frame_level_features(track_id).mean()
    sr.name = track_id
    return sr.loc[filter(lambda c: not "_sma_de" in c, sr.index)]

def get_features(selected_tracks=None, length=None):
    """iterates over the dataset and return a pandas matrix of features for all/selected tracks"""
    if selected_tracks is None:
        track_files = os.listdir(os.path.join(DATASET_PATH, "features/"))
        selected_tracks = sorted(map(lambda name: int(name.split(".")[0]), track_files))[:length]
    all_feats = (get_clip_level_features(track_id) for track_id in tqdm(selected_tracks))
    # NB: the upper limit is set because we are only interested to the `2-2000` range.
    return pd.DataFrame(all_feats).loc[:2000]
# -

get_features(length=50)

# ## Extract Annotations

def get_annotations(length=None):
    """returns a pandas matrix of all the annotations of all tracks"""
    with open(os.path.join(DATASET_PATH, "annotations.csv")) as fin:
        return pd.read_csv(fin, header=0, index_col=0, sep=",\s*", engine="python").iloc[:length]

get_annotations(50)


# ## Feature Visualization

def get_clip_level_features(track_id):
    """converts frame-level features to relevant clip-level features"""
    # just mean everything for now (except deltas), might be redefined later in the notebook
    sr = get_frame_level_features(track_id).mean()
    sr.name = track_id
    return sr.loc[filter(lambda c: not "_sma_de" in c, sr.index)]


# ### Annotations splitting

# +
maxs = dict()
mins = dict()

for label in get_annotations().columns:
    annot = get_annotations().loc[:, label]
    maxs[label] = annot.loc[annot >= annot.mean()].sort_values(ascending=False).index
    mins[label] = annot.loc[annot < annot.mean()].sort_values(ascending=True).index


# -

# ### Helper functions

# +
def plot_feature_evolution(tracks, feature_name, time_slice=slice(None)):
    data = pd.concat((
        get_frame_level_features(i).loc[time_slice, feature_name]
        for i in tqdm(tracks)
        ), axis=1)
    plt.xlabel("time")
    plt.ylabel(feature_name)
    plt.plot(data)

def plot_feature_distribution(tracks, feature_name, x_axis=None):
    mean_std = get_features(sorted(tracks)).loc[:, [f"{feature_name}_amean", f"{feature_name}_stddev"]]
    up = mean_std.loc[:, f"{feature_name}_amean"].max() + mean_std.loc[:, f"{feature_name}_stddev"].max()
    low = mean_std.loc[:, f"{feature_name}_amean"].min() - mean_std.loc[:, f"{feature_name}_stddev"].max()
    if x_axis is None:
        x_axis = np.linspace(low, up, 100)
    dists = mean_std.apply(lambda row: sp.stats.norm(row[0], row[1]).pdf(x_axis), axis=1, result_type="expand")
    dists.columns = x_axis
    plt.xlabel(feature_name)
    plt.ylabel("p.d.f.")
    plt.plot(dists.T)


# -

# Functions for plotting feature-distribution for VA mean values.

def plot_va_means_distributions(feature_name, n_tracks, x_axis=None):
    plt.figure(figsize=(15,10))
    i = 1
    for label in ["valence_mean", "arousal_mean"]:
        plt.subplot(2,2,i*2-1)
        plt.title(f"tracks with min. {label}")
        plot_feature_distribution(mins[label][:n_tracks], feature_name, x_axis)
        plt.subplot(2,2,i*2)
        plt.title(f"tracks with max. {label}")
        plot_feature_distribution(maxs[label][:n_tracks], feature_name, x_axis)
        i += 1
    plt.savefig(os.path.join(RUNTIME_DIR, f"{feature_name}-dists.pdf"))


# Functions for plotting feature time-evolution for VA mean values.

def plot_va_means_evolution(feature_name, n_tracks, time_slice=slice(10,50)):
    plt.figure(figsize=(15,10))
    i = 1
    for label in ["valence_mean", "arousal_mean"]:
        plt.subplot(2,2,i*2-1)
        plt.title(f"tracks with min. {label}")
        plot_feature_evolution(mins[label][:n_tracks], feature_name, time_slice)
        plt.subplot(2,2,i*2)
        plt.title(f"tracks with max. {label}")
        plot_feature_evolution(maxs[label][:n_tracks], feature_name, time_slice)
        i += 1
    plt.savefig(os.path.join(RUNTIME_DIR, f"{feature_name}-time.pdf"))


# ### Feature names

with open("features.txt") as fin:
    print(fin.read())

# ### Feature distributions

plot_va_means_distributions("pcm_RMSenergy_sma", 30, np.linspace(0, 0.4, 100))

plot_va_means_distributions("F0final_sma", 20, np.linspace(0, 500, 100))

plot_va_means_distributions("pcm_fftMag_psySharpness_sma", 50, np.linspace(0, 2.5, 100))

plot_va_means_distributions("pcm_fftMag_spectralHarmonicity_sma", 30, np.linspace(0,5,100))

# ### Feature time-evolution

plot_va_means_evolution("pcm_zcr_sma_amean", 10)


# # Regression

# ## Preliminary manual feature selection

def get_clip_level_features(track_id):
    """converts frame-level features to relevant clip-level features"""
    flf = get_frame_level_features(track_id)
    feats = list()
    for func in ["mean", "std", "max", "min"]:
        feat = flf.__getattribute__(func)()
        feat.index = map(lambda i: f"{i}__{func}", feat.index)
        feats.append(feat)
    sr = pd.concat(feats)
    sr.name = track_id
    #return sr.loc[filter(lambda f: not "" in f, sr.index)]
    return sr


# ## Preparation

# Common procedure for regression training and testing.

def run_regression(reg, feats_train, feats_test, annots_train, feat_selector):
    predictions = pd.DataFrame()
    for label in annots_train.columns:
        selected_feats_train = feat_selector[label].transform(feats_train)
        selected_feats_test  = feat_selector[label].transform(feats_test)
        # regression fitting
        reg = reg.fit(selected_feats_train, annots_train.loc[:, label])
        # regression prediction
        pred = pd.Series(reg.predict(selected_feats_test), feats_test.index)
        pred.name = label
        predictions = predictions.join(pred, how="right")
    return predictions

# Extract N tracks from the dataset.

N       = 2000
feats   = get_features(length=N)
annots  = get_annotations(length=N)
print(f"shape of feats: {feats.shape}\nshape of annots: {annots.shape}")

# Filter features using k-best.

# +
feat_selector = dict()
k_best = 100

for label in annots.columns:
    feat_selector[label] = sklearn.feature_selection.SelectKBest(k=k_best).fit(feats, annots.loc[:, label])
# -

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

# ## Linear Regression

lin_reg = sklearn.linear_model.LinearRegression()
linear_predictions = run_regression(lin_reg, feats_train, feats_test, annots_train, feat_selector)
linear_predictions

# ## SVM Regression

svm_reg = sklearn.svm.SVR()
svm_predictions = run_regression(svm_reg, feats_train, feats_test, annots_train, feat_selector)
svm_predictions

# ## KN Regression

kn_reg = sklearn.neighbors.KNeighborsRegressor(10, "distance")
kn_predictions = run_regression(kn_reg, feats_train, feats_test, annots_train, feat_selector)
kn_predictions


# # Evaluation

def get_metrics(prediction, ground_truth):
    print("MSE:     ", sklearn.metrics.mean_squared_error(ground_truth, prediction))
    print("R² score:", sklearn.metrics.r2_score(ground_truth, prediction))


# ## Metrics for Linear regression

for label in annots.columns:
    print(f"=== metrics for {label} ===")
    get_metrics(linear_predictions.loc[:, label], annots_test.loc[:, label])
    print()

# ## Metrics for SVM regression

for label in annots.columns:
    print(f"=== metrics for {label} ===")
    get_metrics(svm_predictions.loc[:, label], annots_test.loc[:, label])
    print()

# ## Metrics for KN regression

for label in annots.columns:
    print(f"=== metrics for {label} ===")
    get_metrics(kn_predictions.loc[:, label], annots_test.loc[:, label])
    print()
