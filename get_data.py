# %%
from pathlib import Path
import pandas as pd

from preprocessing import extract_dataset, get_audio_features, process_audio

# All slider data
# %%
slider_data = [
    extract_dataset("/home/bram/datasets/0_dB_slider.zip"),
    extract_dataset("/home/bram/datasets/-6_dB_slider.zip"),
    extract_dataset("/home/bram/datasets/6_dB_slider.zip"),
]

# %%
processed_slider = [
    process_audio(slider_data_x, datadir="/home/bram/datasets")
    for slider_data_x in slider_data
]
# %%
processed_slider_all = pd.concat(processed_slider)
processed_slider_all.to_csv("slider_all.csv.xz", index=False)

# %% Do the same for valve data
valve_data = pd.concat(
    [
        pd.read_csv("./processed_data/valve_0dB.csv.xz"),
        pd.read_csv("./processed_data/valve_-6dB.csv.xz"),
        pd.read_csv("./processed_data/valve_6dB.csv.xz"),
    ]
)
valve_data.reset_index(drop=True, inplace=True)
# %% save to csv
valve_data.to_csv("processed_data/valve_all.csv.xz")

# %%
get_audio_features("test_audio/00000000.wav")
# %%
def get_all_training_data(machine, datafolder):
    folder = Path(datafolder)

    # collect data from filenames
    data = [
        extract_dataset(folder / f"0_dB_{machine}.zip"),
        extract_dataset(folder / f"-6_dB_{machine}.zip"),
        extract_dataset(folder / f"6_dB_{machine}.zip"),
    ]

    # get all audio features
    processed_data = [process_audio(data_x, datadir=datafolder) for data_x in data]

    # concat and save to csv
    processed_data_all = pd.concat(processed_data)
    processed_data_all.to_csv(
        Path("processed_data") / f"{machine}_all.csv.xz", index=False
    )


# %%
pump_data = pd.concat(
    [
        pd.read_csv("./processed_data/0dB_pump.csv"),
        pd.read_csv("./processed_data/6dB_pump.csv"),
        pd.read_csv("./processed_data/pump_-6db.csv"),
    ]
)
pump_data.reset_index(drop=True, inplace=True)

# %% save to csv
pump_data.to_csv("processed_data/pump_all.csv.xz")

# %%
