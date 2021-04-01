# %%
from pathlib import Path
import pandas as pd
from sys import argv

from preprocessing import extract_dataset, get_audio_features, process_audio


def get_all_training_data(machine, datafolder):
    folder = Path(datafolder)

    # collect data from filenames
    data = [
        extract_dataset(folder / f"0_dB_{machine}.zip"),
        extract_dataset(folder / f"-6_dB_{machine}.zip"),
        extract_dataset(folder / f"6_dB_{machine}.zip"),
    ]
    print(f"Read zip files.")

    # get all audio features
    print("Process all audio features")
    processed_data = [process_audio(data_x, datadir=datafolder) for data_x in data]

    # concat and save to csv
    processed_data_all = pd.concat(processed_data)
    outpath = Path("processed_data") / f"{machine}_all.csv.xz"
    processed_data_all.to_csv(outpath, index=False)
    print(f"Saved to {outpath}")


if __name__ == "__main__":
    if len(argv) == 3:
        get_all_training_data(argv[1], argv[2])