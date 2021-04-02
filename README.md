# Detecting malfunctioning machinery with ML

Industrial machines need maintenance from time to time. 
In this project we investigate if machine learning classification techniques can be applied to detect malfunctioning machines. 
We use sound recordings of several types of industrial machines, as available in the academic MIMII dataset.  

## Project overview

- Duration: `2 weeks`
- Deadline: `04/02/2021 10:00 AM`
- 'Team challenge'
- Team Members:
	- [Bram De Vroey](https://github.com/brmdv)
	- [Sijal Kumar Joshi](https://github.com/sijal001)

### Mission objectives

* Create a Machine Learning model that predicts when a machine will fail, based on the current sound.
* Extra: Model that can categorize the failures. This help to do more targeted maintenance.

### Learning Objectives

* To be able to work in a team 
* To be able to complete task in given timeline
* To be able to understand business needs and problems
* To be able to present model to customer.
* To be able to present a final product.

## About the MIMII dataset 

The [MIMII data](https://arxiv.org/abs/1909.09347) is [available](https://zenodo.org/record/3384388) as twelve large zip files, that each contains a multitude of sound recordings in WAVE format. 
There are 4 machine types (valves, fans, slide rails and pumps), each  with 6 different models. 
For each model there is a set of sounds form normal functioning and abnormal functioning. 

To resemble a real-life scenario, various anomalous sounds were recorded (e.g. contamination, leakage, rotating unbalance, and rail damage). 
Also, the background noise recorded in multiple real factories was mixed with the machine sounds. 
The sounds were recorded by eight-channel microphone array with 16 kHz sampling rate and 16 bit per sample. 
The MIMII dataset assists benchmark for sound-based machine fault diagnosis. 
Users can test the performance for specific functions e.g., unsupervised anomaly detection, transfer learning, noise robustness, etc.


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


--- 

## Classification

The classification of (ab)normal sounds is done with a Random Forrest classification.  

The dataset was inbalanced, i.e. there were much more normal than abnormal sound files. In order to resolve these issues, the training data was upsampled with the library `imblearn`.

MOre detailed descriptions of the training process can be found in the notebooks in the directory _notebooks/_, and in the branch `sijal` of this repository.

---

## Usage

The trained models are saved as pickled Skikit-Learn files and can be found in the folder *saved_model/*. 

In [_predict.py_](predict.py), a function `predict_failure()` is available to predict whether a provided sound file is _normal_ or _abnormal_. If the machine type is provided, the correct pre-trained model will be selected automatically.
The file can also be used dreclty in the terminal, to classify a sound file, e.g.
```
$ python predict.py test_file.wav slider
ABNORMAL
```

Required packages are listed in [_requirements.txt_](requirements.txt). 
### Requirements


**Imporant Libaries:**

| Library       | Used to                                        |
| ------------- | :----------------------------------------------|
| numpy		| to work around multi-dimensional of generic data |
| pandas	| to remove, move, copy files.					 |
| matplotlib     |to genereate ploting.		                 |
| sklearn	| Machine learning library for the Python. 	     |
| imblearn	| offering a number of re-sampling techniques.	 |


**Note:** Just use command `pip install -r requirements.txt` to install the required libaries with correct version and run the program smoothly.

### Project files


```
codit-usecase
│
│   README.md               :explains the project
│   requirements.txt        :packages to install to run the program
│   .gitignore              :specifies which files to ignore when pushing to the repository
│__   
│  Data_Model_analysis      :directory contain all the main .ipynb that create the machine model that train test and creates a pickel files.
│   │
│   │ Fan_data_analysis     :notebook that contain data, Machine learning model, metric,statics, etc. fan.
│   │ Pump_data_analysis    :notebook that contain data, Machine learning model, metric,statics, etc. pump.
│   │ Slider_data_analysis  :notebook that contain data, Machine learning model, metric,statics, etc. slider.
│   │ Valve_data_analysis   :notebook that contain data, Machine learning model, metric,statics, etc. valve.
│   │
│   │ dataset		    :directory contains all .pynb file that does the preprocessing and fearure extration.:
│     │__
│	 processed_data     :directory contains the .csv files that contains the main machine features and information.
│	 preprocessing.py   :
│	 get_data.py        :
│__   
│  main		    	    :directory contain all the main .ipynb that create the machine model that train test and creates a pickel files.
│   │
│   │ Pump_ML_model.ipynb   :notebook that contain trainning, pickel creatin script and abnormal sound clusting for machine pump.
│   │ Slider_ML_model.ipynb :notebook that contain trainning, pickel creatin script and abnormal sound clusting for machine slider.
│   │ Fan_ML_model.ipynb    :notebook that contain trainning, pickel creatin script and abnormal sound clusting for machine fan.
│   │ Valve_ML_model.ipynb  :notebook that contain trainning, pickel creatin script and abnormal sound clusting for machine valve.
│   │
│   │ saved_model      	    :directory contains all saved pickel files of the machine learning model.
│   │ dataset		    :directory contains all .pynb file that does the preprocessing and fearure extration.:
│     │__
│	 processed_data     :directory contains the .csv files that contains the main machine features and information.
│	 preprocessing.py   :
│	 get_data.py        :
```

---



## Next Steps

* Optimize the model with better data set and feature to generate better predicitions.
* Improve 
* Improving end user experiance.