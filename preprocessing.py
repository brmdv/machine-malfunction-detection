"""Script can be used to preprocess files from the MIMII Dataset."""

import zipfile

import librosa
import numpy as np
import pandas as pd


def get_audio_features(wavefile) -> pd.DataFrame:
    """Extract features from a provides audio file. Return a Pandas Dataframe.

    :param wavefile: path to audio file
    """
    pass


if __name__ == "__main__":
    print(f"Preprocess MiMii data")
