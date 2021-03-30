# %%
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
