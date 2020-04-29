# -*- coding: utf-8 -*-
# # Initialization

# ## Generic imports

# +
import numpy as np
import scipy as sp
import pandas as pd
import librosa

from tqdm.notebook import tqdm
from functools import lru_cache, partial
from operator import methodcaller

import concurrent.futures as cf
import os
import sys
# -

# ## Download Dataset

# +
from requests import get as download
from shutil import copyfileobj
from zipfile import ZipFile

DW_URL  = "https://polimi365-my.sharepoint.com/:u:/g/personal/10768481_polimi_it/ET_EMOV_tgBAm2yIQn4m4h0B8FxvxcDCJkpedf_3SRtLWw?download=1"
DW_PATH = "./dataset.zip"

if os.path.isdir(DW_PATH[:-4]):
    print("already downloaded.")
else:
    print("downloading...")
    with download(DW_URL, stream=True) as r:
        with open(DW_PATH, 'wb') as f:
            copyfileobj(r.raw, f)
    print("unzipping...")
    with ZipFile(DW_PATH) as z:
        z.extractall()
    os.unlink(DW_PATH)
    print("DONE!")
# -

# ## Other stuff

# +
RUNTIME_DIR = "./run/"
CACHE_DIR = "./cache/"

try:
    os.mkdir(RUNTIME_DIR)
except FileExistsError:
    pass

try:
    os.mkdir(CACHE_DIR)
except FileExistsError:
    pass
# -

# # Model

# ## Extract Features

DATASET_PATH = "./dataset/"

# Names of relevant features

# for librosa
features_to_extract = {
    "feature": ["spectral_flatness", "tonnetz", "chroma_stft"] # "spectral_contrast"],
    "effects": ["harmonic", "percussive"],
    "beat": ["tempo"]
}
feature_moments = {
    # pandas function → column name
    "mean": "amean",
    "std": "stddev",
#    "max": "max",
#    "min": "min",
#    "kurtosis": "kurtosis",
}

# ### Caching functions

# +
from inspect import getsource
from hashlib import sha1


def get_cache_path(basename, *args):
    to_bytes = lambda sth: (getsource(sth) if callable(sth) else repr(sth)).encode()
    h = sha1(b"".join(map(methodcaller("digest"), map(sha1, map(to_bytes, args))))).hexdigest()[:6]
    return os.path.join(CACHE_DIR, f"{basename}@{h}.csv")

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

# +
import warnings

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
    cache_path = get_cache_path("lrosa_features", features_to_extract, feature_moments, extract_features)
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
    sr = get_frame_level_features(track_id).mean()
    sr.name = track_id
    return sr


# -

# Caching mechanism.

def get_provided_features(track_ids, pool=None):
    """iterate over `track_ids` and return a pandas matrix of provided features"""
    cache_path = get_cache_path("provided_features", get_clip_level_features)
    return get_cached_features(track_ids, cache_path, get_clip_level_features,
                               "get_provided_features(...)", pool)


# Merge extracted features with provided features and iterate over dataset.

def get_features(selected_tracks=None, length=None, filt=lambda x: True):
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
        computed  = get_extracted_features(selected_tracks, p)
        provided  = get_provided_features(selected_tracks, p)
        all_feats = (provided.result()).join(computed.result())
        # NB: the upper limit is set because we are only interested to the `2-2000` range.
        return all_feats.loc[:2000, filter(filt, all_feats.columns)]


get_features(length=10)


# ## Extract Annotations

def get_annotations(length=None):
    """returns a pandas matrix of all the annotations of all tracks"""
    with open(os.path.join(DATASET_PATH, "annotations.csv")) as fin:
        return pd.read_csv(fin, header=0, index_col=0, sep=",\s*", engine="python").iloc[:length]

get_annotations(length=10)


# ## Feature Visualization

import matplotlib.pyplot as plt

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

plot_va_means_distributions("spectral_flatness", 50, np.linspace(-0.001,0.08,100))

plot_va_means_distributions("chroma_stft2", 50, np.linspace(-0.2, 1, 100))

plot_va_means_distributions("pcm_RMSenergy_sma", 20, np.linspace(0, 0.4, 100))

plot_va_means_distributions("F0final_sma", 20, np.linspace(0, 500, 100))

plot_va_means_distributions("pcm_fftMag_psySharpness_sma", 50, np.linspace(0, 2.5, 100))

plot_va_means_distributions("pcm_fftMag_spectralHarmonicity_sma", 100, np.linspace(0,3,100))

plot_va_means_distributions("pcm_zcr_sma", 25, np.linspace(0, 0.15, 100))

# ### Feature time-evolution

plot_va_means_evolution("pcm_zcr_sma_amean", 10)

# # Regression

# +
from sklearn import preprocessing, model_selection, feature_selection, clone as skclone

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression, RidgeCV, SGDRegressor
from sklearn.svm import LinearSVR, SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
# -

# ## Preliminary manual feature selection

# +
features_to_select = [
    "spectral_flatness",
    "tonnetz",
    "chroma",
    "harmonic",
    "percussive",
    "tempo",
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
    "voicingFinalUnclipped",
    "jitterLocal",
    "jitterDDP",
    "shimmerLocal",
    "logHNR",
    "audspec_lengthL1norm",
    "audspecRasta_lengthL1norm",
    "Rfilt",
    "fftMag_fband",
]

def feature_filter(featname):
    return any((sel in featname for sel in features_to_select)) and not "_sma_de" in featname


# -

# ## Preparation

# Common functions for regression training, prediction, and cross-validation.

# +
def reg_to_regs(regs, keys):
    if type(regs) == dict:
        return regs
    reg = regs
    regs = dict()
    for label in keys:
        regs[label] = skclone(reg)
    return regs

def run_regression(regs, feats_train, feats_test, annots_train, feat_processor):
    regs = reg_to_regs(regs, annots_train.columns)
    predictions = pd.DataFrame()
    for label in tqdm(annots_train.columns, leave=False):
        selected_feats_train = feat_processor[label].transform(feats_train)
        selected_feats_test  = feat_processor[label].transform(feats_test)
        # regression fitting
        regs[label].fit(selected_feats_train, annots_train.loc[:, label])
        # regression prediction
        pred = pd.Series(regs[label].predict(selected_feats_test), feats_test.index)
        pred.name = label
        predictions = predictions.join(pred, how="right")
    return predictions

def cross_validation_score(regs, feats_train, annots_train, feat_processor):
    regs = reg_to_regs(regs, annots_train.columns)
    for label in tqdm(annots_train.columns, leave=False):
        selected_feats_train = feat_processor[label].transform(feats_train)
        scores = model_selection.cross_val_score(regs[label], selected_feats_train, 
                                                 annots_train.loc[:, label], cv=10)
        print(f"R² score: {scores.mean() :5.2f} (± {scores.std() * 2 :4.2f})  [{label}]")

def run_cross_validation(init_reg, params, feats_train, annots_train, feat_processor, force=False):
    regs = dict()
    for label in tqdm(annots_train.columns, leave=False):
        selected_feats_train = feat_processor[label].transform(feats_train)
        reg = model_selection.GridSearchCV(init_reg, params, n_jobs=-1, cv=10, refit=False, verbose=1)
        reg.fit(feats_train, annots_train.loc[:, label])
        print(reg.best_params_, file=sys.stderr)
        regs[label] = type(init_reg)(**reg.best_params_)
    return regs
# -

# Extract N tracks from the dataset.

N       = 2000
feats   = get_features(length=N, filt=feature_filter)
annots  = get_annotations(length=N)
print(f"shape of feats: {feats.shape}\nshape of annots: {annots.shape}")

# Split the dataset in training set and testing set.

# +
(feats_train, feats_test,
 annots_train, annots_test) = model_selection.train_test_split(feats, annots)

print("Training set:", feats_train.index)
print("Testing set:", feats_test.index)
# -

# Preproces features by scaling them to have $\bar{X} = 1$ and $\sigma^2_X = 1$. Then filter out unneeded or redundant features.

# +
feat_processor = dict()

for label in annots.columns:
    pl = make_pipeline(
        # --- standardize features ---
        preprocessing.StandardScaler(),
        # --- filter out features ---
        feature_selection.VarianceThreshold(0.9),
        #feature_selection.SelectKBest(feature_selection.mutual_info_regression, 20),
        #feature_selection.SelectKBest(feature_selection.f_regression, 50),
        #feature_selection.RFE(SVR(kernel="linear")),
        verbose = 1
    )
    feat_processor[label] = pl.fit(feats_train, annots_train.loc[:, label])
    # print some report
    n_before = feats_train.iloc[0:1].shape[1]
    n_after  = pl.transform(feats_train.iloc[0:1]).shape[1]
    print(f"{'-'*3}\nfeat_processor for {label} reduces features from {n_before} to {n_after}\n{'-'*80}")
# -

# ## Linear Regression

# Run cross-validation to find best parameters.

# +
#lin_param_grid = (
#    {
#        "loss": ("epsilon_insensitive",),
#        "alpha": (1e-4, 1e-3, 1e-2),
#        "epsilon": (1e-3, 1e-2, 1e-1),
#        "tol": (1e-4, 1e-3),
#    }
#)
#lin_reg = run_cross_validation(SGDRegressor(), lin_param_grid, feats_train, annots_train, feat_processor)

lin_reg = RidgeCV()
# -

cross_validation_score(lin_reg, feats_train, annots_train, feat_processor)

# Save final predictions for later evaluation.

linear_predictions = run_regression(lin_reg, feats_train, feats_test, annots_train, feat_processor)
linear_predictions

# ## SVM Regression

# Run cross-validation to find best parameters.

# +
# svm_param_grid = (
#   {'C': (1, 10, 100, 1000), 'kernel': ('linear',)},
#   {'C': (1, 10, 100, 1000), 'gamma': (0.001, 0.0001), 'kernel': ('rbf',)},
# )
# svm_reg = run_cross_validation(SVR(), svm_param_grid, feats_train, annots_train, feat_processor)

svm_reg = SVR()
cross_validation_score(svm_reg, feats_train, annots_train, feat_processor)
# -

# Save final predictions for later evaluation.

svm_predictions = run_regression(svm_reg, feats_train, feats_test, annots_train, feat_processor)
svm_predictions

# ## KN Regression

# Run cross-validation to find the best parameters.

kn_reg = KNeighborsRegressor(10, "distance")
cross_validation_score(kn_reg, feats_train, annots_train, feat_processor)

# Save final predictions for later evaluation.

kn_predictions = run_regression(kn_reg, feats_train, feats_test, annots_train, feat_processor)
kn_predictions

# # Final Evaluation

# +
from sklearn import metrics

def get_metrics(prediction, ground_truth):
    print("MSE:     ", metrics.mean_squared_error(ground_truth, prediction))
    print("R² score:", metrics.r2_score(ground_truth, prediction))


# -

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
