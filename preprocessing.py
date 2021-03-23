"""Script can be used to preprocess files from the MIMII Dataset."""

import zipfile
from pathlib import Path
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
    """Extract data points from MIMII zip file. Every .wav file will become one
    data point. An function to extract data from the content of each file can
    be specified.

    :param filepath: Path to the zip file which holds all the wav files.
    :param sound_func: Function that extracts audio features from each wave file. 
    :return: Pandas DataFrame with all features 
    """
    # convert to Path object for easy path methods
    filepath = Path(filepath)
    setname = filepath.stem  # name of archive
    SNR = int(setname.split("_")[0])  # get dB from name -6_dB_slider → -6
    machine = setname.split("_")[2]

    data = {
        "dataset": filepath.name,
        "machine": machine,
        "SNR": SNR,
        "machine_id": [],
        "wavefile": [],
        "is_normal": [],
    }

    with zipfile.ZipFile(filepath, "r") as file:
        for soundfile in file.infolist():
            # loop through zip contents, only do .wav files
            if soundfile.filename.endswith(".wav"):
                # convert to Path for easy
                soundfilename = Path(soundfile.filename)
                # target feature: is normal or abnormal?
                is_normal = not soundfilename.parts[-2].endswith("abnormal")
                # machine id from folder name: id_01 → 1
                machine_id = int(soundfilename.parts[-3].split("_")[-1])

                # add row to data
                data["wavefile"].append(str(soundfilename))
                data["is_normal"].append(is_normal)
                data["machine_id"].append(machine_id)

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Read zipfile from command line.
    if len(argv) > 1:
        extract_dataset(argv[1])
