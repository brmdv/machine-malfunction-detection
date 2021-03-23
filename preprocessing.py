"""Script can be used to preprocess files from the MIMII Dataset."""

import zipfile
from sys import argv

import librosa
import numpy as np
import pandas as pd


def get_audio_features(wavefile: str) -> dict:
    """Extract features from a provided audio file.

    :param wavefile: path to audio file
    """
    librosa.load(wavefile)

    return {}


def extract_dataset(filepath: str, sound_func=None) -> pd.DataFrame:
    """

    :param filepath: Path to the zip file which holds all the wav files.
    :param sound_func: Function that extracts audio features from each wave file. 
    :return: Pandas DataFrame with all features 
    """
    with zipfile.ZipFile(filepath, "r") as file:
        print(file.infolist())


if __name__ == "__main__":
    # Read zipfile from command line.
    if len(argv) > 1:
        extract_dataset(argv[1])
