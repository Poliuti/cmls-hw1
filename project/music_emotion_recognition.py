# -*- coding: utf-8 -*-
# # Initialization

# ## Generic imports
#
# Other imports will be defined later on.

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
#
# We readapted the provided dataset in a single zip file for easier extraction and naming.

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
#
# Create some directories for cache and runtime temporary data (mainly the plots) that we are not willing to track in `git`.

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

# ## Feature Extraction

DATASET_PATH = "./dataset/"

# Define here which features we want to extract with `librosa` and which statistical moments we are interested in.
#
# The way the `features_to_extract` dictionary works is like this: if a librosa function is provided under `librosa.mod.func`, then we `features_to_extract["mod"].append("func")`, so then we can use cool introspection to iterate over them.

# +
features_to_extract = {
    "feature": ["spectral_flatness", "tonnetz", "chroma_stft", "spectral_contrast",
                "spectral_bandwidth", "tempogram"],
    "effects": ["harmonic", "percussive"],
    "beat": ["tempo"]
}

feature_moments = {
    # pandas function → column name
    "mean": "amean",
    "std": "stddev",
    "max": "max",
    "min": "min",
    "kurtosis": "kurtosis",
}
# -

# ### Caching functions
#
# > There are 2 hard problems in computer science: cache invalidation, naming things, and off-by-1 errors.
# > — Leon Bambrick
#
# As feature extraction takes time I figured out some way to cache things. It's not smart, but it gets the job done.

# +
from inspect import getsource
from hashlib import sha1


def get_cache_path(basename, *args):
    """compute a cache path with `basename` and append a hash computed on `*args`"""
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
    iterate over `track_ids` and return a pandas DataFrame of features, as obtained by `extractor_function`,
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

# ### Librosa features
#
# This is the part dealing with extracting features with `librosa`, ideally this function won't be needed as we already have the cache.

# +
import warnings

def extract_raw_features(track_id, duration=60):
    """returns a dictionary of extracted time-level features for track `track_id`"""
    path = os.path.join(DATASET_PATH, "audio", f"{track_id}.mp3")
    with warnings.catch_warnings():
        # drop warning for ".mp3" files
        warnings.simplefilter("ignore")
        y, sr = librosa.load(path, duration=duration)
    raw_features = dict()
    f_len = len([x for y in features_to_extract.values() for x in y])
    # extract features using librosa
    with tqdm(total=f_len, desc=f"extract_raw_features({track_id})", leave=False) as pbar:
        for feattype in features_to_extract.keys():
            for featname in features_to_extract[feattype]:
                extra_opts = dict()
                if featname == "tempogram":
                    extra_opts["win_length"] = 128
                # use introspection to call the relevant function in librosa module
                raw_features[featname] = getattr(getattr(librosa, feattype), featname)(y=y, **extra_opts)
                pbar.update()
    return raw_features


# +
#efeat10 = extract_raw_features(10)

# +
#efeat10["spectral_flatness"].shape
# -

# convert raw time-level features to pandas series of clip-level features.

def extract_features(track_id):
    """return a pandas series of extracted clip-level features for track `track_id`"""
    raw_features = extract_raw_features(track_id)
    features = list()
    # iterate over each matrix as returned by librosa
    for featname in raw_features.keys():
        feats = raw_features[featname]
        # if we have a one-row matrix we can just drop the second dimension
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
                    getattr(frame, moment)(axis=1) # time evolution in librosa is left→rigth
                )
    # concat all features to a single pandas series
    sr = pd.concat(features)
    sr.name = track_id
    return sr


# +
#extract_features(10)
# -

# The actual function we care about, it encapsulates cache logic defined above in a black-box manner.

def get_extracted_features(track_ids, pool=None):
    """iterate over `track_ids` and return a pandas DataFrame of extracted clip-level features"""
    # invalidate cache if we select other features to extract or change the way we condense them
    cache_path = get_cache_path("lrosa_features", features_to_extract, feature_moments, extract_features)
    return get_cached_features(track_ids, cache_path, extract_features, "get_extracted_features(...)", pool)


get_extracted_features([10])


# ### Provided features
#
# This part instead deals with processing the features which are provided in the dataset.

# +
@lru_cache(maxsize=None)
def get_provided_frame_level_features(track_id):
    """returns a pandas DataFrame of all provided frame-level features for `track_id`"""
    with open(os.path.join(DATASET_PATH, "features", f"{track_id}.csv")) as fin:
        return pd.read_csv(fin, header=0, index_col=0, sep=";", engine="c")

def get_clip_level_features_from_provided(track_id):
    """converts provided frame-level features to relevant clip-level features"""
    sr = get_provided_frame_level_features(track_id).mean()
    sr.name = track_id
    return sr


# -

# We can use some cache here as well to speed up things (instead of doing a lot of I/O over sparse .csv files).

def get_provided_features(track_ids, pool=None):
    """iterate over `track_ids` and return a pandas DataFrame of provided features"""
    # invalidate cache if we change the way we condense the provided features
    cache_path = get_cache_path("provided_features", get_clip_level_features_from_provided)
    return get_cached_features(track_ids, cache_path, get_clip_level_features_from_provided,
                               "get_provided_features(...)", pool)


get_provided_features([10])


# ### Get final features
#
# Merge extracted features with provided features and iterate over dataset.

def get_features(selected_tracks=None, length=None):
    """iterates over the dataset and return a pandas DataFrame of features for all/selected tracks"""
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
        return all_feats.loc[:2000]


# +
fff = get_features()

# some checks on data consistency
print(f"Any value is NaN? {fff.isnull().values.any()}")
print(f"Any value is Infinite? {not np.isfinite(fff.to_numpy()).all()}")


# -

# ## Annotations Extraction
#
# Annotations are in a simple .csv file, so this part is more straightforward.

def get_annotations(length=None):
    """returns a pandas DataFrame of all the annotations of all tracks"""
    with open(os.path.join(DATASET_PATH, "annotations.csv")) as fin:
        return pd.read_csv(fin, header=0, index_col=0, sep=",\s*", engine="python").iloc[:length]

# +
aaa = get_annotations()

# some checks on data consistency
print(f"Any value is NaN? {aaa.isnull().values.any()}")
print(f"Any value is Infinite? {not np.isfinite(aaa.to_numpy()).all()}")
# -


# ## Dataset Splitting
#
# To avoid over-fitting, every further consideration must be done on the **training set**, therefore the first step is to split the dataset.

from sklearn import model_selection, preprocessing

# Pick features and annotations from the dataset.

# +
N       = 2000 # actual number of tracks is lower (there are holes), but it's ok to provide a higher number
feats   = get_features(length=N)
annots  = get_annotations(length=N)
print(f"shape of feats: {feats.shape}\nshape of annots: {annots.shape}")

# standardize annotations (not strictly needed, but it's worth a try)
annots = pd.DataFrame(preprocessing.scale(annots), columns=annots.columns, index=annots.index)
# -

# Split dataset in training set and testing set.

# +
(feats_train, feats_test,
 annots_train, annots_test) = model_selection.train_test_split(feats, annots)

print("Training set:", feats_train.index)
print("Testing set:", feats_test.index)
# -

# # Feature Visualization
#
# In this part we play around with plotting feature distributions, etc...

import matplotlib.pyplot as plt

# ### Annotations splitting
#
# This was the first idea of visualizing things: separate tracks in two sets, a "high-annotation" set and a "low-annotation" set, and sort them accordingly (the first one descending, the second one ascending).
#
# Picking the first `N` tracks for e.g. "arousal_mean" means: *«select the first `N` tracks with highest `arousal_mean` and compare them to the first `N` tracks with lowest `arousal_mean`.»*

# +
annot_maxs = dict()
annot_mins = dict()

for label in annots_train.columns:
    annot = annots_train.loc[:, label]
    annot_maxs[label] = annot.loc[annot >= annot.mean()].sort_values(ascending=False).index
    annot_mins[label] = annot.loc[annot < annot.mean()].sort_values(ascending=True).index


# -

# ### Plotting functions

# +
def plot_feature_evolution(tracks, feature_name, time_slice=slice(None)):
    data = pd.concat((
        get_provided_frame_level_features(i).loc[time_slice, feature_name]
        for i in tqdm(tracks, leave=False)
        ), axis=1)
    plt.xlabel("time")
    plt.ylabel(feature_name)
    plt.plot(data)

def plot_feature_distribution(tracks, feature_name, x_axis=None):
    mean_std = get_features(sorted(tracks)).loc[:, [f"{feature_name}_amean", f"{feature_name}_stddev"]]
    plot_mean = mean_std.loc[:, f"{feature_name}_amean"].mean()
    plot_std = mean_std.loc[:, f"{feature_name}_stddev"].mean()
    if x_axis is None:
        x_axis = np.linspace(plot_mean - plot_std, plot_mean + plot_std, 100)
    dists = mean_std.apply(lambda row: sp.stats.norm(row[0], row[1]).pdf(x_axis), axis=1, result_type="expand").T
    dists.index = x_axis
    plt.xlabel(feature_name)
    plt.ylabel("p.d.f.")
    plt.plot(dists)

def plot_tempo_hist(tracks):
    tempos = get_features(sorted(tracks)).loc[:, "tempo"]
    plt.xlabel("bpm")
    plt.ylabel("# songs")
    plt.hist(tempos)


# -

# Functions for plotting 4 quadrants (low-valence, high-valence, low-arousal, high-arousal).

# +
def plot_va_distributions(feature_name, n_tracks, x_axis=None, annot_type="mean"):
    plt.figure(figsize=(15,10))
    i = 1
    with tqdm(total=4, leave=False) as pbar:
        for label in [f"valence_{annot_type}", f"arousal_{annot_type}"]:
            plt.subplot(2,2,i*2-1)
            plt.title(f"tracks with min. {label}")
            plot_feature_distribution(annot_mins[label][:n_tracks], feature_name, x_axis)
            pbar.update()
            plt.subplot(2,2,i*2)
            plt.title(f"tracks with max. {label}")
            plot_feature_distribution(annot_maxs[label][:n_tracks], feature_name, x_axis)
            pbar.update()
            i += 1
    plt.savefig(os.path.join(RUNTIME_DIR, f"va_{annot_type}-{feature_name}-dists.pdf"))

def plot_va_evolution(feature_name, n_tracks, time_slice=slice(10,50), annot_type="mean"):
    plt.figure(figsize=(15,10))
    i = 1
    with tqdm(total=4, leave=False) as pbar:
        for label in [f"valence_{annot_type}", f"arousal_{annot_type}"]:
            plt.subplot(2,2,i*2-1)
            plt.title(f"tracks with min. {label}")
            plot_feature_evolution(annot_mins[label][:n_tracks], feature_name, time_slice)
            pbar.update()
            plt.subplot(2,2,i*2)
            plt.title(f"tracks with max. {label}")
            plot_feature_evolution(annot_maxs[label][:n_tracks], feature_name, time_slice)
            pbar.update()
            i += 1
    plt.savefig(os.path.join(RUNTIME_DIR, f"va_{annot_type}-{feature_name}-time.pdf"))

def plot_va_tempos(n_tracks, annot_type="mean"):
    plt.figure(figsize=(15,10))
    i = 1
    with tqdm(total=4, leave=False) as pbar:
        for label in [f"valence_{annot_type}", f"arousal_{annot_type}"]:
            plt.subplot(2,2,i*2-1)
            plt.title(f"tracks with min. {label}")
            plot_tempo_hist(annot_mins[label][:n_tracks])
            pbar.update()
            plt.subplot(2,2,i*2)
            plt.title(f"tracks with max. {label}")
            plot_tempo_hist(annot_maxs[label][:n_tracks])
            pbar.update()
            i += 1
    plt.savefig(os.path.join(RUNTIME_DIR, f"va_{annot_type}-tempo.pdf"))

def plot_scatter(feature_name, limit=None, x_max=float("inf")):
    feats = feats_train.loc[feats_train.loc[:, feature_name] < x_max, feature_name]
    annots = annots_train.loc[feats_train.loc[:, feature_name] < x_max]
    plt.figure(figsize=(15,10))
    i = 1
    for label in tqdm(annots.columns, leave=False):
        plt.subplot(2,2,i)
        #plt.title(f"scatter for {label}")
        plt.xlabel(feature_name)
        plt.ylabel(label)
        plt.scatter(feats.iloc[:limit], annots.iloc[:limit].loc[:, label])
        i += 1
    plt.savefig(os.path.join(RUNTIME_DIR, f"scatter-{feature_name}.pdf"))


# -

# ### Provided-Feature names
#
# We manually opened a .csv file of provided features and copied the feature names in a .txt file, showing them here:

with open("features.txt") as fin:
    print(fin.read())

# ### Scatters
#
# A better way to visualize features might be to plot feature-vs-annotation scatters.

plot_scatter("pcm_fftMag_mfcc_sma[1]_amean")

plot_scatter("pcm_zcr_sma_amean")

plot_scatter("pcm_RMSenergy_sma_amean")

plot_scatter("spectral_contrast2_amean")

# ### Songs tempo

plot_va_tempos(50)

# ### Feature distributions

plot_va_distributions("spectral_flatness", 50, np.linspace(-0.01, 0.05, 100))

plot_va_distributions("chroma_stft2", 50, np.linspace(-0.2, 1, 100), annot_type="std")

plot_va_distributions("pcm_RMSenergy_sma", 20, np.linspace(0, 0.4, 100))

plot_va_distributions("F0final_sma", 20, np.linspace(0, 500, 100))

plot_va_distributions("pcm_fftMag_psySharpness_sma", 50, np.linspace(0, 2.5, 100))

plot_va_distributions("pcm_fftMag_spectralHarmonicity_sma", 100, np.linspace(0,3,100))

plot_va_distributions("pcm_zcr_sma", 25, np.linspace(0, 0.15, 100))

# ### Feature time-evolution

plot_va_evolution("pcm_zcr_sma_amean", 10, annot_type="mean")

# # Regression

# +
from sklearn import preprocessing, model_selection, feature_selection, clone as skclone

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression, RidgeCV, SGDRegressor
from sklearn.svm import LinearSVR, SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
# -

# ## Preliminary manual feature selection

# We have a Pipeline later on that select features based on this dictionary. In particular we select different features depending on the annotation to be predicted.
#
# e.g. a given feature `fullname_for_featX_with_suffix` for `valence_mean` is **not discarded** if `featX` (or any other substring) is listed among `features_to_select["valence_mean"]"` or `features_to_select["always"]`.

# +
features_to_select = {
    "valence_mean": [
        "voicing",
        "audspec_lengthL1norm",
        "RMSenergy",
        "spectralFlux",
        "psySharpness",
        "spectralHarmonicity_sma_amean",
    ],
    "valence_std": [
        "shimmerLocal_sma_stddev",
    ],
    "arousal_mean": [
        "voicing",
        "shimmerLocal_sma_amean",
        "audspec_lengthL1norm",
        "RMSenergy",
        "spectralFlux",
        "psySharpness",
    ],
    "arousal_std": [
        # seems like nothing is useful
    ],
    "always": [
        # -- provided --
        "logHNR",
        "zcr",
        #"Rfilt", # unsure about these ↓
        #"fftMag_fband",
        "spectralRollOff",
        "spectralCentroid",
        "spectralEntropy",
        "spectralVariance",
        #"spectralSkewness",
        #"spectralKurtosis",
        "spectralSlope",
        "mfcc", # to here ↑
        # -- librosa --
        "tonnetz",
        "chroma",
        #"harmonic",
        #"percussive",
        "tempo",
        "spectral_flatness",
        "spectral_contrast",
        "spectral_bandwidth",
        "tempogram",
    ],
}

def feature_filter(annot_name):
    def real_filter(featname):
        return not "_sma_de" in featname and any(
            (sel in featname for sel in (features_to_select[annot_name] + features_to_select["always"]))
        )
    return real_filter

def manual_feature_filter(annot_name):
    """given a pandas DataFrame of features, returns a pandas DataFrame of filtered features"""
    def real_filter(features):
        return features.loc[:, filter(feature_filter(annot_name), features.columns)]
    return real_filter


# -

# ## Preparation

# Common functions for regression training, prediction, and cross-validation. With these functions we take care of iterating over all four annotations and output back predictions in a pandas `DataFrame` (otherwise we would get just `numpy.ndarray`).

# +
def reg_to_regs(regs, keys):
    """
    given a regressor object, clone it four times in order to have four independent
    regressors for each annotation to be predicted. This is mostly useful in the case of regressors
    which include cross-validation inside, e.g. `RidgeCV`.
    """
    if type(regs) == dict:
        return regs
    reg = regs
    regs = dict()
    for label in keys:
        regs[label] = skclone(reg)
    return regs

def run_regression(regs, feats_train, feats_test, annots_train, feat_processor):
    """fit regressor (or regressor dictionary, if already distinct) to data and return predictions"""
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
    predictions.name = type(next(iter(regs.values()))).__name__
    return predictions

def cross_validation_score(regs, feats_train, annots_train, feat_processor):
    """print cross-validation scores for regressor (or regressor dictionary, if already distinct)"""
    regs = reg_to_regs(regs, annots_train.columns)
    for label in tqdm(annots_train.columns, leave=False):
        selected_feats_train = feat_processor[label].transform(feats_train)
        scores = model_selection.cross_val_score(regs[label], selected_feats_train, 
                                                 annots_train.loc[:, label], cv=10)
        print(f"R² score: {scores.mean() :5.2f} (± {scores.std() * 2 :4.2f})  [{label}]")

def run_cross_validation(init_reg, params, feats_train, annots_train, feat_processor, force=False):
    """perform grid-search to find best parameters for `init_reg` and
       return the best four regressors found in a dictionary annot_name → reg_obj"""
    regs = dict()
    for label in tqdm(annots_train.columns, leave=False):
        selected_feats_train = feat_processor[label].transform(feats_train)
        reg = model_selection.GridSearchCV(init_reg, params, n_jobs=-1, cv=10, refit=False, verbose=2)
        reg.fit(feats_train, annots_train.loc[:, label])
        print(reg.best_params_, file=sys.stderr)
        regs[label] = type(init_reg)(**reg.best_params_)
    return regs
# -

# Preprocess features:
#  - First we manually discard unneeded features. This is done first because we still have a pandas `DataFrame` at this stage, otherwise we would not be able to read feature names.
#  - Then we scale them to have $\bar{X} = 0$ and $\sigma^2_X = 1$.
#  - Finally we filter out unneeded or redundant features using automatic feature selection tools.

# +
feat_processor = dict()

for label in annots.columns:
    pl = make_pipeline(
        # --- manual feature selection ---
        preprocessing.FunctionTransformer(manual_feature_filter(label)),
        # --- standardize features ---
        preprocessing.StandardScaler(),
        # --- filter out features ---
        feature_selection.VarianceThreshold(1 - 1e-15),
        feature_selection.SelectKBest(feature_selection.f_regression, 150),
        #feature_selection.RFE(LinearSVR(), 50),
        verbose = 1
    )
    feat_processor[label] = pl.fit(feats_train, annots_train.loc[:, label])
    # print some report
    n_before = feats_train.iloc[0:1].shape[1]
    n_after  = pl.transform(feats_train.iloc[0:1]).shape[1]
    print(f"checking if all values are finite and not NaN: {np.isfinite(pl.transform(feats)).all()}")
    print(f"{'-'*3}\nfeat_processor for {label} reduces features from {n_before} to {n_after}\n{'-'*80}")
# -

# ## Linear Regression

# Run cross-validation to evaluate the simples linear regressor.

lin_reg = LinearRegression()
cross_validation_score(lin_reg, feats_train, annots_train, feat_processor)

# Use `GridSearchCV` to optimize another type of linear regressor (Stochastic Gradient Descent) and see if it performs better.

# +
sgd_param_grid = (
    {
        "loss": ("epsilon_insensitive",),
        "alpha": (1e-4, 1e-3, 1e-2),
        "epsilon": (1e-3, 1e-2, 1e-1),
        "tol": (1e-4, 1e-3),
    }
)
#sgd_reg = run_cross_validation(SGDRegressor(), lin_param_grid, feats_train, annots_train, feat_processor)

# to avoid running it again, these are the best parameters we found on one particular run
sgd_reg = {
    "valence_mean": SGDRegressor(loss="epsilon_insensitive", alpha=1e-3, epsilon=1e-3, tol=1e-4),
    "valence_std":  SGDRegressor(loss="epsilon_insensitive", alpha=1e-2, epsilon=1e-2, tol=1e-4),
    "arousal_mean": SGDRegressor(loss="epsilon_insensitive", alpha=1e-2, epsilon=1e-3, tol=1e-3),
    "arousal_std":  SGDRegressor(loss="epsilon_insensitive", alpha=1e-3, epsilon=1e-1, tol=1e-4),
}
# -

cross_validation_score(sgd_reg, feats_train, annots_train, feat_processor)

# Last, compare with a third linear regressor with integrated cross-validation optimization.

ridge_reg = RidgeCV()
cross_validation_score(ridge_reg, feats_train, annots_train, feat_processor)

# Use best linear regressor found and save final predictions for later evaluation.

linear_predictions = run_regression(ridge_reg, feats_train, feats_test, annots_train, feat_processor)

# ## SVM Regression

# Use `GridSearchCV` to optimize `NuSVR` over different kernels and free paramenters.

# +
nu_param_grid = (
    {
        "kernel": ("rbf", "sigmoid"),
        "C": (0.1, 1, 10),
        "nu": (0.25, 0.5, 0.75),
    },
)
#nu_reg = run_cross_validation(NuSVR(), nu_param_grid, feats_train, annots_train, feat_processor)

# to avoid running it again, these are the best parameters we found on one particular run
nu_reg = {
    "valence_mean": NuSVR(kernel="rbf", C=10,  nu=0.75),
    "valence_std":  NuSVR(kernel="rbf", C=0.1, nu=0.25),
    "arousal_mean": NuSVR(kernel="rbf", C=1,   nu=0.5),
    "arousal_std":  NuSVR(kernel="rbf", C=0.1, nu=0.75),
}
# -

cross_validation_score(nu_reg, feats_train, annots_train, feat_processor)

# Use `GridSearchCV` to optimize `SVR` over different kernels and free parameters.

# +
svr_param_grid = (
   {
       "kernel": ("rbf", "sigmoid"),
       "C": (0.1, 1, 10),
       "epsilon": (1e-3, 1e-2, 1e-1),
       "tol": (1e-4, 1e-3),
   },
)
#svr_reg = run_cross_validation(SVR(), svr_param_grid, feats_train, annots_train, feat_processor)

# to avoid running it again, this are the best parameters we found on one particular run
svr_reg = {
    "valence_mean": SVR(kernel="rbf", C=10,  epsilon=0.1,  tol=1e-3),
    "valence_std":  SVR(kernel="rbf", C=0.1, epsilon=0.1,  tol=1e-3),
    "arousal_mean": SVR(kernel="rbf", C=0.1, epsilon=0.1,  tol=1e-4),
    "arousal_std":  SVR(kernel="rbf", C=0.1, epsilon=0.01, tol=1e-3),
}
# -

cross_validation_score(svr_reg, feats_train, annots_train, feat_processor)

# Use best SVM regressor found and save final predictions for later evaluation.

svm_predictions = run_regression(nu_reg, feats_train, feats_test, annots_train, feat_processor)

# ## KN Regression

# Use `GridSearchCV` to find best parameters for KN.

# +
kn_param_grid = (
    {
        "weights": ("uniform", "distance"),
        "n_neighbors": (10, 50, 60, 70, 80, 90, 100, 150, 200),
        "n_jobs": (-1,),
    },
)
#kn_reg = run_cross_validation(KNeighborsRegressor(), kn_param_grid, feats_train, annots_train, feat_processor)

# to avoid running it again, this are the best parameters we found on one particular run
kn_reg = {
    "valence_mean": KNeighborsRegressor(weights="uniform", n_neighbors=80,  n_jobs=-1),
    "valence_std":  KNeighborsRegressor(weights="uniform", n_neighbors=200, n_jobs=-1),
    "arousal_mean": KNeighborsRegressor(weights="uniform", n_neighbors=80,  n_jobs=-1),
    "arousal_std":  KNeighborsRegressor(weights="uniform", n_neighbors=150, n_jobs=-1),
}
# -

cross_validation_score(kn_reg, feats_train, annots_train, feat_processor)

# Save final predictions for later evaluation.

kn_predictions = run_regression(kn_reg, feats_train, feats_test, annots_train, feat_processor)

# # Final Evaluation
#
# We check this stage at last, once we are satisfied with the previous cross-validation scores and cannot find any way to enhance them. We plot some scatters as well to visualize the predictions against the ground truth for each annotation.

# +
from sklearn import metrics

def print_metrics(predictions):
    for label in predictions.columns:
        gtru = annots_test.loc[:, label]
        pred = predictions.loc[:, label]
        print(f"=== metrics for {label} ===")
        print("MSE:     ", metrics.mean_squared_error(gtru, pred))
        print("R² score:", metrics.r2_score(gtru, pred))
        print()

def plot_results(predictions):
    plt.figure(figsize=(15,10))
    i = 1
    for label in predictions.columns:
        pred = predictions.loc[:, label]
        gtru = annots_test.loc[:, label]
        plt.subplot(2,2,i)
        plt.title(f"comparison for {label}")
        plt.xlabel("predictions")
        plt.ylabel("ground truth")
        plt.scatter(pred, gtru)
        i += 1
    plt.savefig(os.path.join(RUNTIME_DIR, f"predictions-{predictions.name}.pdf"))


# -

# ## Metrics for Linear regression

print_metrics(linear_predictions)

plot_results(linear_predictions)

# ## Metrics for SVM regression

print_metrics(svm_predictions)

plot_results(svm_predictions)

# ## Metrics for KN regression

print_metrics(kn_predictions)

plot_results(kn_predictions)
