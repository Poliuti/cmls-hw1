import pandas as pd

f1, f2 = None, None

def set_env(lf1, lf2):
	global f1, f2
	f1, f2 = lf1, lf2

def concat_features(track_id):
	global f1, f2
	return pd.concat((f1(track_id), f2(track_id)))
