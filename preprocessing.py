"""Script can be used to preprocess files from the MIMII Dataset. 

Command line usage:
   python preprocess.py INPUT_DATA.zip OUTPUT_CSV.csv
"""

from os import PathLike
from pathlib import Path
from sys import argv
from zipfile import ZipFile

import librosa
from librosa.feature.spectral import spectral_flatness
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_audio_features(wavefile) -> dict:
    """Extract features from a provided audio file.

    :param wavefile: path to audio file
    """
    # load wave file, don't resample and don't merge 8 channels
    # Y, sr = librosa.load(wavefile, sr=None, mono=False)
    y, sr = librosa.load(wavefile, sr=None)  # don't resample, this is slow
    rms = librosa.feature.rms(y)
    zcr = librosa.feature.zero_crossing_rate(y)
    # first calculate spectogram, so following functions can reuse this
    S = np.abs(librosa.stft(y))

    mel = librosa.feature.melspectrogram(S=S, sr=sr)
    mfcc = librosa.feature.mfcc(sr=sr, S=S)
    freq_rms = librosa.feature.rms(S=S)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(sr=sr, S=S)
    spectral_contrast = librosa.feature.spectral_contrast(sr=sr, S=S)
    spectral_flatness = librosa.feature.spectral_flatness(S=S)
    spectral_rolloff = librosa.feature.spectral_rolloff(sr=sr, S=S)

    return {
        # "duration": librosa.get_duration(y, sr),
        "T_rms_mean": np.mean(rms),
        # "T_rms_median": np.median(rms),
        "T_rms_std": np.std(rms),
        "T_zcr_mean": np.mean(zcr),
        "T_zcr_std": np.mean(zcr),
        "F_mel_mean": np.mean(mel),
        # "F_mel_median": np.median(mel),
        "F_mel_std": np.std(mel),
        "F_mel_rms_mean": np.mean(freq_rms),
        # "F_mel_rms_median": np.median(spectral_rms),
        "F_mel_rms_std": np.std(freq_rms),
        "F_mfcc_mean": np.mean(mfcc),
        # "F_mfcc_median": np.median(mfcc),
        "F_mfcc_std": np.std(mfcc),
        "F_flatness_mean": np.mean(spectral_flatness),
        "F_flatness_std": np.std(spectral_flatness),
        "F_bandwidth_mean": np.mean(spectral_bandwidth),
        "F_bandwidth_std": np.std(spectral_bandwidth),
        "F_contrast_mean": np.mean(spectral_contrast),
        "F_contrast_std": np.std(spectral_contrast),
        "F_rolloff_mean": np.mean(spectral_rolloff),
        "F_rolloff_std": np.std(spectral_rolloff),
    }


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

    with ZipFile(filepath, "r") as file:
        for soundfile in file.infolist():
            # loop through zip contents, only do .wav files
            if soundfile.filename.endswith(".wav"):
                # convert to Path for easy
                soundfilename = Path(soundfile.filename)
                # target feature: is normal or abnormal?
                is_normal = 0 if soundfilename.parts[-2].endswith("abnormal") else 1
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
        # create progress bar
        tqdm.pandas(desc="Extracting audio features…")

        # open the correct zipfile
        with ZipFile(datadir / datazip, "r") as opened_zipfile:

            def extract_apply(row):
                """Helper function to apply on extracted wavefile."""
                with opened_zipfile.open(row["wavefile"]) as opened_wavefile:
                    result = func(opened_wavefile)
                return result

            # apply on dataframe
            new_cols = dataframe[dataframe["dataset"] == datazip].progress_apply(
                extract_apply, axis=1, result_type="expand"
            )

        # join the new columns with the result dataframe
        results = results.join(new_cols)

    return results


if __name__ == "__main__":
    # Read zipfile from command line.
    if len(argv) > 2:
        out_path = Path(argv[2])
        if out_path.exists():
            print(f"Warning, {out_path} will be overwritten.")

    if len(argv) > 1:
        path = Path(argv[1])
        # create dataframe
        df = extract_dataset(path)
        # apply audio feature extraction
        processed_df = process_audio(df, datadir=path.parent)

        # if outfile specified, write to csv
        if len(argv) > 2:
            processed_df.to_csv(out_path, index=False)

    else:
        print(__doc__)
