import pickle
from builtins import FileNotFoundError, RuntimeError, hasattr, isinstance
from pathlib import Path
from sys import argv

import numpy as np

from preprocessing import get_audio_features


def predict_failure(sound, machine_type, model=None):
    """Predict whether the provides (list of) sound files is functioning normally. 

    :param sound: Sound file that will be analysed. Can also be a list of paths.
    :param machine_type: Type of machine. One of "slider", "valve", "fan", "pump"
    :param model: Optional, select pickled model file.
    :return: Array with 0 or 1 (abnormal / normal) for each soundfile.
    """
    # select and load trained model
    if model is None:
        if machine_type in ["fan", "slider", "pump", "valve"]:
            model_path = Path(f"./saved_model/Predict_{machine_type}_sound_type.sav")
        else:
            raise RuntimeError("machine_type should be slider, fan, pump, or valve.")
    else:
        model_path = Path(model)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model pickel file not found: {model_path}")

    with model_path.open("rb") as model_file:
        trained_model = pickle.load(model_file)

    # load features from audio file
    if not hasattr(sound, "__iter__") or isinstance(sound, str):
        sound = [sound]

    features = [get_audio_features(soundfile) for soundfile in sound]

    selected_features = (
        [
            "T_rms_mean",
            "T_rms_std",
            "T_zcr_mean",
            "F_mel_mean",
            "F_mel_std",
            "F_mel_rms_mean",
            "F_mel_rms_std",
            "F_mfcc_mean",
            "F_mfcc_std",
            "F_flatness_mean",
            "F_bandwidth_mean",
            "F_bandwidth_std",
            "F_contrast_mean",
            "F_rolloff_mean",
            "F_rolloff_std",
        ]
        if machine_type != "valve"
        else [
            "T_rms_mean",
            "T_rms_std",
            "T_zcr_mean",
            "T_zcr_std",
            "F_mel_mean",
            "F_mel_std",
            "F_mel_rms_mean",
            "F_mel_rms_std",
            "F_mfcc_mean",
            "F_mfcc_std",
            "F_flatness_mean",
            "F_flatness_std",
            "F_bandwidth_mean",
            "F_bandwidth_std",
            "F_contrast_mean",
            "F_contrast_std",
            "F_rolloff_mean",
            "F_rolloff_std",
        ]
    )

    # funnel features into same order as model
    X = np.array(
        [[feature_i[name] for name in selected_features] for feature_i in features]
    )

    # make prediction
    y_pred = trained_model.predict(X).ravel()

    return y_pred


def prediction_test_helper(machine):
    prediction = predict_failure(
        [f"test_audio/{machine}_abnormal.wav", f"test_audio/{machine}_normal.wav",],
        f"{machine}",
    )
    assert prediction[0] == 0 and prediction[1] == 1
    print(f"{machine:>10}: {prediction}")


def test_predictions():
    prediction_test_helper("valve")
    prediction_test_helper("pump")
    prediction_test_helper("fan")
    prediction_test_helper("slider")


if __name__ == "__main__":
    # command line functionality
    if len(argv) == 3:
        prediction = predict_failure(argv[1], argv[2])
        print(["ABNORMAL", "NORMAL"][prediction])
    else:
        print("USAGE:\n", "python predict.py SOUND_FILE MACHINE_TYPE")
