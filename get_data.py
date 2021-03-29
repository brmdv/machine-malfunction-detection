# %%
import pandas as pd

from preprocessing import extract_dataset, process_audio

# All slider data
# %%
slider_data = pd.concat(
    [
        extract_dataset("/home/bram/datasets/0_dB_slider.zip"),
        extract_dataset("/home/bram/datasets/-6_dB_slider.zip"),
        extract_dataset("/home/bram/datasets/6_dB_slider.zip"),
    ]
)
slider_data.reset_index(inplace=True, drop=True)
slider_data

# %%
processed_slider = process_audio(slider_data, datadir="/home/bram/datasets")
# %%
processed_slider.to_csv("slider_all.csv.xz", index=False)

# load from file
# %%
processed = pd.read_csv("./slider_all.csv.xz")

# %%
