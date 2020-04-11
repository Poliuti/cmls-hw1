from setuptools import setup

setup(
	name="mer",
	version="0.0.1",
	description="""Music Emotion Recognition project""",
	python_requires=">=3.7,<4",
	install_requires=["pandas", "numpy", "librosa", "scikit-learn", "matplotlib", "tqdm"],
	packages=["mer"]
)
