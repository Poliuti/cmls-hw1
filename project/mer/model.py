from functools import lru_cache
from operator import itemgetter
from tqdm import tqdm

import pandas as pd
import numpy as np
import os

## feature extraction
@lru_cache(maxsize=None)
def get_features(track_id, dataset_path):
	"""returns a pandas matrix of averaged features for track `track_id`"""
	with open(os.path.join(dataset_path, "features", f"{track_id}.csv")) as fin:
		feats = pd.read_csv(fin, header=0, index_col=0, sep=";", engine="c")
	#return pd.concat((feats.mean(), feats.std()), keys=["mean", "std"], axis=1, copy=False)
	# TODO: how to extract clip-level features from time-level features averaged with a moving average of 3?
	return feats.mean()

def get_all_features(dataset_path, length=None):
	"""iterates over the dataset and extracts features of all tracks"""
	all_feats = []
	feature_files = sorted(os.listdir(os.path.join(dataset_path, "features/")), key=lambda name: int(name.split(".")[0]))[:length]
	for fname in tqdm(feature_files):
		track_id = fname.split(".")[0]
		feats = get_features(track_id, dataset_path)
		all_feats.append((track_id, feats))
	return pd.DataFrame(map(itemgetter(1), all_feats), index=map(itemgetter(0), all_feats))


## annotations extraction
@lru_cache(maxsize=None)
def get_annotations(dataset_path):
	"""returns a pandas matrix of all the annotations of all tracks"""
	with open(os.path.join(dataset_path, "annotations.csv")) as fin:
		return pd.read_csv(fin, header=0, index_col=0, sep=",\s*", engine="python")

def get_all_valence(dataset_path, length=None):
	"""returns valence mean and std for every track"""
	return get_annotations(dataset_path).iloc[:length].loc[:, ["valence_mean", "valence_std"]]

def get_all_arousal(dataset_path, length=None):
	"""returns arousal mean and std for every track"""
	return get_annotations(dataset_path).iloc[:length].loc[:, ["arousal_mean", "arousal_std"]]


## dataset splitting
def get_training_set(dataset_path):
	pass

def get_testing_set(dataset_path):
	pass
