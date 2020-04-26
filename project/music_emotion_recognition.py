# -*- coding: utf-8 -*-
# # Initialization

# ## Imports

# +
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import requests

import sklearn.model_selection
import sklearn.linear_model
import sklearn.svm
import sklearn.neighbors
import sklearn.feature_selection

from tqdm.notebook import tqdm
from functools import lru_cache, partial
from operator import methodcaller
from zipfile import ZipFile
from hashlib import sha1

import concurrent.futures as cf
import os
import sys
import shutil
import inspect
import warnings
# -

# ## Download Dataset

# +
DW_URL  = "https://polimi365-my.sharepoint.com/:u:/g/personal/10768481_polimi_it/ET_EMOV_tgBAm2yIQn4m4h0B8FxvxcDCJkpedf_3SRtLWw?download=1"
DW_PATH = "./dataset.zip"

if os.path.isdir(DW_PATH[:-4]):
    print("already downloaded.")
else:
    print("downloading...")
    with requests.get(DW_URL, stream=True) as r:
        with open(DW_PATH, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
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

# Names of relevant features

# +
# for librosa
features_to_extract = {
    "feature": ["spectral_flatness", "tonnetz", "chroma_stft"],
    "effects": ["harmonic", "percussive"],
    "beat": ["tempo"]
}
feature_moments = {
    # pandas function → column name
    "mean": "amean",
    "std": "stddev",
} #, "max", "min", "kurtosis"]

# filter from dataset
features_to_select = [
    "F0final",
    "RMSenergy",
    "zcr",
    "spectralRollOff",
    "spectralFlux",
    "spectralCentroid",
    "spectralEntropy",
    "spectralVariance",
    "spectralSkewness",
    "spectralKurtosis",
    "spectralSlope",
    "psySharpness",
    "spectralHarmonicity",
    "mfcc",
]


# -

# ### Caching functions

# +
def get_cache_path(basename, *args):
    to_bytes = lambda sth: (inspect.getsource(sth) if callable(sth) else repr(sth)).encode()
    h = sha1(b"".join(map(methodcaller("digest"), map(sha1, map(to_bytes, args))))).hexdigest()[:6]
    return os.path.join(RUNTIME_DIR, f"{basename}@{h}.csv")

def load_cache(fname):
    try:
        with open(fname) as fin:
            return pd.read_csv(fin, header=0, index_col=0, sep=",", engine="c")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()

def save_cache(fname, df):
    with open(fname, "w") as fout:
        df.to_csv(fout)

def get_cached_features(track_ids, cache_path, extractor_function, bar_desc, pool=None):
    """
    iterate over `track_ids` and return a pandas matrix of features, as obtained by `extractor_function`,
    using caching mechanism and paralel extracting.
    a Future is returned instead if a PoolExecutor `pool` is provided.
    """
    # helper function
    def prepare_out(features):
        # make input always a Future
        if type(features) != cf.Future:
            r = features
            features = cf.Future()
            features.set_result(r)
        # output future
        out = cf.Future()
        # remember to re-transpose to get final matrix
        filter_selected = lambda all_features: out.set_result(all_features.result().T.loc[track_ids])
        features.add_done_callback(filter_selected)
        # check if we need to resolve the future or not
        return out.result() if pool is None else out
    
    # load cache
    features = load_cache(cache_path).T # transpose: appending columns to a dataframe is faster
    missing_tracks = set(track_ids) - set(features.columns)

    if len(missing_tracks) == 0:
        # -- cache hit --
        return prepare_out(features)

    # -- cache miss --
    # either create a new pool or use provided one
    p = pool if pool is not None else cf.ThreadPoolExecutor() # initializer=print)
    # helper function
    def cache_updater(extractor): # non-thread safe, run only once
        with tqdm(total=len(missing_tracks), desc=bar_desc, leave=False) as pbar:
            # as_completed returns futures, map is needed to extract resolved values
            for feats in map(methodcaller("result"), cf.as_completed(extractor)):
                features[feats.name] = feats
                save_cache(cache_path, features.T)
                pbar.update()
        return features
    # async-iterate over missing_tracks
    extractor = (p.submit(extractor_function, track) for track in missing_tracks)
    new_features = p.submit(cache_updater, extractor)
    # prepare returned object
    requested_features = prepare_out(new_features)
    # if pool was created here we need to clenup it
    if pool is None:
        p.shutdown()
    # return resolver/resolved
    return requested_features


# -

# ### Extract features using librosa

def extract_raw_features(track_id, duration=60):
    """returns a dictionary of extracted time-level features for track `track_id`"""
    path = os.path.join(DATASET_PATH, "audio", f"{track_id}.mp3")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(path, duration=duration)
    raw_features = dict()
    f_len = len([x for y in features_to_extract.values() for x in y])
    # extract features using librosa
    with tqdm(total=f_len, desc=f"extract_raw_features({track_id})", leave=False) as pbar:
        for feattype in features_to_extract.keys():
            for featname in features_to_extract[feattype]:
                raw_features[featname] = getattr(getattr(librosa, feattype), featname)(y=y)
                pbar.update()
    return raw_features


# +
#efeat10 = extract_raw_features(10)
#for k in efeat10.keys():
#    print(k, efeat10[k].shape)
# -

# convert raw time-level features to pandas series of clip-level features.

def extract_features(track_id):
    """return a pandas series of extracted clip-level features for track `track_id`"""
    raw_features = extract_raw_features(track_id)
    features = list()
    # extract relevant moments
    for featname in raw_features.keys():
        feats = raw_features[featname]
        # if we have a single row vector we can just drop the second dimension
        if len(feats.shape) > 1 and feats.shape[0] == 1:
            feats = feats.reshape(-1)
        # now, do we have a vector or a matrix?
        if len(feats.shape) == 1:
            if feats.shape[0] > 1:
                ## vector: time-level feature
                for moment in feature_moments.keys():
                    features.append(
                        pd.Series(getattr(pd.Series(feats), moment)(),
                                  index=[f"{featname}_{feature_moments[moment]}"])
                    )
            else:
                ## single value feature
                features.append(
                    pd.Series(feats, index=[featname])
                )
        else:
            ## multi-column feature
            for moment in feature_moments.keys():
                frame = pd.DataFrame(
                    feats,
                    index=[f"{featname}{i}_{feature_moments[moment]}" for i in range(feats.shape[0])]
                )
                features.append(
                    getattr(frame,moment)(axis=1) # time evolution is originally as →
                )
    # concat all features to a single pandas series
    sr = pd.concat(features)
    sr.name = track_id
    return sr


# +
#extract_features(10)
# -

# Use caching mechanism to avoid recomputing everything each time.

def get_extracted_features(track_ids, pool=None):
    """iterate over `track_ids` and return a pandas matrix of extracted features"""
    cache_path = get_cache_path("lrosa_features", features_to_extract, extract_features)
    return get_cached_features(track_ids, cache_path, extract_features, "get_extracted_features(...)", pool)


get_extracted_features([10, 12])


# ### Load provided features

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
    return sr.loc[filter(lambda c: not "_sma_de" in c and any((f in c for f in features_to_select)), sr.index)]


# -

# Caching mechanism.

def get_provided_features(track_ids, pool=None):
    """iterate over `track_ids` and return a pandas matrix of provided features"""
    cache_path = get_cache_path("provided_features", get_clip_level_features)
    return get_cached_features(track_ids, cache_path, get_clip_level_features,
                               "get_provided_features(...)", pool)


# Merge extracted features with provided features and iterate over dataset.

def get_features(selected_tracks=None, length=None):
    """iterates over the dataset and return a pandas matrix of features for all/selected tracks"""
    ## helper functions
    def concat_features_with_lock(lrosa_lock, track_id):
        return pd.concat((get_clip_level_features(track_id), get_extracted_features(track_id, lock=lrosa_lock)))
    ## read directory if no selected_tracks are provided
    if selected_tracks is None:
        track_files = os.listdir(os.path.join(DATASET_PATH, "features/"))
        selected_tracks = sorted(map(lambda name: int(name.split(".")[0]), track_files))[:length]
    all_feats = list()
    ## spawn a process pool for concurrent fetching
    with cf.ThreadPoolExecutor(max_workers=os.cpu_count()*2) as p:
        ## fetch provided features
        computed = get_extracted_features(selected_tracks, p)
        provided = get_provided_features(selected_tracks, p)
        # NB: the upper limit is set because we are only interested to the `2-2000` range.
        return (provided.result()).join(computed.result()).loc[:2000]


_=get_features()


# ## Extract Annotations

def get_annotations(length=None):
    """returns a pandas matrix of all the annotations of all tracks"""
    with open(os.path.join(DATASET_PATH, "annotations.csv")) as fin:
        return pd.read_csv(fin, header=0, index_col=0, sep=",\s*", engine="python").iloc[:length]

get_annotations(50)


# ## Feature Visualization

# +
#def get_clip_level_features(track_id):
#    """converts frame-level features to relevant clip-level features"""
#    # just mean everything for now (except deltas), might be redefined later in the notebook
#    sr = get_frame_level_features(track_id).mean()
#    sr.name = track_id
#    return sr.loc[filter(lambda c: not "_sma_de" in c, sr.index)]
# -

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
        for i in tqdm(tracks, leave=False)
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
    dists = mean_std.apply(lambda row: sp.stats.norm(row[0], row[1]).pdf(x_axis), axis=1, result_type="expand").T
    dists.index = x_axis
    plt.xlabel(feature_name)
    plt.ylabel("p.d.f.")
    plt.plot(dists)


# -

# Functions for plotting feature-distribution for VA mean values.

def plot_va_means_distributions(feature_name, n_tracks, x_axis=None):
    plt.figure(figsize=(15,10))
    i = 1
    with tqdm(total=4, leave=False) as pbar:
        for label in ["valence_mean", "arousal_mean"]:
            plt.subplot(2,2,i*2-1)
            plt.title(f"tracks with min. {label}")
            plot_feature_distribution(mins[label][:n_tracks], feature_name, x_axis)
            pbar.update()
            plt.subplot(2,2,i*2)
            plt.title(f"tracks with max. {label}")
            plot_feature_distribution(maxs[label][:n_tracks], feature_name, x_axis)
            pbar.update()
            i += 1
    plt.savefig(os.path.join(RUNTIME_DIR, f"{feature_name}-dists.pdf"))


# Functions for plotting feature time-evolution for VA mean values.

def plot_va_means_evolution(feature_name, n_tracks, time_slice=slice(10,50)):
    plt.figure(figsize=(15,10))
    i = 1
    with tqdm(total=4, leave=False) as pbar:
        for label in ["valence_mean", "arousal_mean"]:
            plt.subplot(2,2,i*2-1)
            plt.title(f"tracks with min. {label}")
            plot_feature_evolution(mins[label][:n_tracks], feature_name, time_slice)
            pbar.update()
            plt.subplot(2,2,i*2)
            plt.title(f"tracks with max. {label}")
            plot_feature_evolution(maxs[label][:n_tracks], feature_name, time_slice)
            pbar.update()
            i += 1
    plt.savefig(os.path.join(RUNTIME_DIR, f"{feature_name}-time.pdf"))


# ### Feature names

with open("features.txt") as fin:
    print(fin.read())

# ### Feature distributions

plot_va_means_distributions("spectral_flatness0", 6, np.linspace(0,0.1,100))

plot_va_means_distributions("pcm_RMSenergy_sma", 10, np.linspace(0, 0.4, 100))

plot_va_means_distributions("F0final_sma", 20, np.linspace(0, 500, 100))

plot_va_means_distributions("pcm_fftMag_psySharpness_sma", 50, np.linspace(0, 2.5, 100))

plot_va_means_distributions("pcm_fftMag_spectralHarmonicity_sma", 100, np.linspace(0,3,100))

plot_va_means_distributions("pcm_zcr_sma", 100, np.linspace(0, 0.25, 100))

# ### Feature time-evolution

plot_va_means_evolution("pcm_zcr_sma_amean", 10)


# # Regression

# ## Preliminary manual feature selection

# +
#relevant_features = ["zcr", "F0final", "mfcc", "spectralHarmonicity", "psySharpness", "spectralRollOff"]
#relevant_moments = ["mean"] # provide pandas function names

#def get_clip_level_features(track_id):
#    """converts frame-level features to relevant clip-level features"""
#    flf = get_frame_level_features(track_id)
#    feats = list()
#    for func in relevant_moments:
#        feat = flf.__getattribute__(func)()
#        feat.index = map(lambda i: f"{i}__{func}", feat.index)
#        feats.append(feat)
#    sr = pd.concat(feats)
#    sr.name = track_id
#    return sr.loc[filter(lambda f: any((x in f for x in relevant_features)), sr.index)]
# -

# ## Preparation

# Common procedure for regression training, testing, and cross-validation.

# +
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

def run_cross_validation(reg):
    pass #TODO
# -

# Extract N tracks from the dataset.

N       = 100
feats   = get_features(length=N)
annots  = get_annotations(length=N)
print(f"shape of feats: {feats.shape}\nshape of annots: {annots.shape}")

# Filter features using k-best.

# +
feat_selector = dict()
k_best = 50

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
