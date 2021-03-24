"""Script can be used to preprocess files from the MIMII Dataset."""

import zipfile
from collections import defaultdict
from os import PathLike
from pathlib import Path
from sys import argv

import librosa
import numpy as np
import pandas as pd


def get_audio_features(wavefile) -> dict:
    """Extract features from a provided audio file.

    :param wavefile: path to audio file
    """
    # load wave file, don't resample and don't merge 8 channels
    Y, sr = librosa.load(wavefile, sr=None, mono=False)
    return {"duration": librosa.get_duration(Y, sr)}


def extract_dataset(filepath: str, sound_func=None) -> pd.DataFrame:
    """Extract data points from MIMII zip file. Every .wav file will become one
    data point. An function to extract data from the content of each file can
    be specified.

    :param filepath: Path to the zip file which holds all the wav files.
    :param sound_func: Function that extracts audio features from each wave file. 
    :return: Pandas DataFrame with all features.
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

                # # process audio
                # if sound_func is not None:
                #     with file.open(soundfile, "r") as wavefile:
                #         sound_data = sound_func(wavefile)

    return pd.DataFrame(data)


def process_audio(
    dataframe: pd.DataFrame, func=get_audio_features, datadir: PathLike = "./"
):
    """Apply a given function to every wave file in the dataframe with sound
    files. The resulting features will be added as extra columns to the
    dataframe. 

    :param dataframe: input (and output) dataframe
    :param func: function to apply on, defaults to get_audio_features
    :param datadir: Directory that holds all dataset zipfiles, defaults to "./"
    """
    # path where to look for zipfiles
    if not isinstance(datadir, Path):
        datadir = Path(datadir)

    results = dataframe.copy()

    for datazip in pd.unique(dataframe["dataset"]):
        # open the correct zipfile
        with zipfile.ZipFile(datadir / datazip, "r") as opened_zipfile:

            def extract_apply(row):
                """Helper function to apply on extracted wavefile."""
                with opened_zipfile.open(row["wavefile"]) as opened_wavefile:
                    result = func(opened_wavefile)
                return result

            # apply on dataframe
            new_cols = dataframe[dataframe["dataset"] == datazip].apply(
                extract_apply, axis=1, result_type="expand"
            )

        # join the new columns wioth the result datafram
        results = results.join(new_cols)

    return results


if __name__ == "__main__":
    # Read zipfile from command line.
    if len(argv) > 1:
        path = Path(argv[1])
        df = extract_dataset(path)
        process_audio(df, datadir=path.parent)

        if len(argv) > 2:
            df.to_csv(argv[2])
