# %%
import pandas as pd

from preprocessing import extract_dataset, process_audio

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

# %%
