# Detecting malfunctioning machinery with ML

Industrial machines need maintenance from time to time. 
In this project we investigate if machine learning classification techniques can be applied to detect malfunctioning machines. 
We use sound recordings of several types of industrial machines, as available in the academic MIMII dataset.  



## About the MIMII dataset 

The [MIMII data](https://arxiv.org/abs/1909.09347) is [available](https://zenodo.org/record/3384388) as twelve large zip files, that each contains a multitude of sound recordings in WAVE format. 
There are 4 machine types (valves, fans, slide rails and pumps), each  with 6 different models. 
For each model there is a set of sounds form normal functioning and abnormal functioning. 


## Data preprocessing

In order to convert sound files to useable data points, we had to extract a number of discrete feautures from each audio file. 
We used the Python library [Librosa](https://librosa.org/doc/latest/index.html), which can be used to read and analyse audio. 
Librosa provides a multitude of processing functions, in the time domain, and predominently in the frequency domain.
After some experimentation, we selected about ten features, most of them based on the [MEL spectrogram](https://en.wikipedia.org/wiki/Mel_scale) of each audio file -- a representation that is commonly used for ML in audio. 

Because most of this audio processing gives higher-dimensional results. decisions about how to collapse them into discrete numbers had to be made.
Because of time restrictions, and because of promising results with the simple selection of mean, standard deviation and median (the latter we later dropped because of similarity to the mean) gave us good-enough results.  
However, we are aware that we loose some information, and not all subtitlities will be characterized.
This is a domain in which futher imporvements could be reached, for example by representing some features by other, more suitable statistical aggregations.

All the data preprocessing is done inside [_preprocessing.py_](preprocessing.py), which holds functions to read the MIMII zip files, and apply Librosa processing on each wave file inside. 


## Classification


## Usage


The trained models are saved as pickled Skikit-Learn files and can be found in the folder *saved_model/*. 

In [_predict.py_](predict.py), a function `predict_failure()` is available to predict whether a provided sound file is _normal_ or _abnormal_. If the machine type is provided, the correct pre-trained model will be selected automatically.
The file can also be used dreclty in the terminal, to classify a sound file, e.g.
```
$ python predict.py test_file.wav slider
ABNORMAL
```

Required packages are listed in [_requirements.txt_](requirements.txt). 

