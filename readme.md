# Machine Learning
---
## Malfunctioning Industrial Machine Investigation and Inspection Using Sound

- Repository: `repo_name`
- Duration: `2 weeks`
- Deadline: `04/02/2021 10:00 AM`


# Mission objectives

* Create a Machine Learning model that predicts when a machine will fail, based on the current sound.
* Extra: Model that can categorize the failures. This help to do more targeted maintenance.


# Learning Objectives

* To be able to work in a team 
* To be able to complete task in given timeline
* To be able to understand business needs and problems
* To be able to present model to customer.
* To be able to present a final product.


# The Mission

This dataset is a sound dataset for malfunctioning industrial machine investigation and inspection (MIMII dataset). It contains the sounds generated from four types of industrial machines, i.e. valves, pumps, fans, and slide rails. Each type of machine includes seven individual product models*1, and the data for each model contains normal sounds (from 5000 seconds to 10000 seconds) and anomalous sounds (about 1000 seconds). To resemble a real-life scenario, various anomalous sounds were recorded (e.g., contamination, leakage, rotating unbalance, and rail damage). Also, the background noise recorded in multiple real factories was mixed with the machine sounds. The sounds were recorded by eight-channel microphone array with 16 kHz sampling rate and 16 bit per sample. The MIMII dataset assists benchmark for sound-based machine fault diagnosis. Users can test the performance for specific functions e.g., unsupervised anomaly detection, transfer learning, noise robustness, etc.

---

# About Running the Program

* **Python version:** `3.8.8`

**Imporant Libaries:**

| Library       | Used to                                        |
| ------------- | :----------------------------------------------|
| numpy		|to work around multi-dimensional of generic data|
| os		|to work around system path.			 |
| matplotlib	|to genereate ploting.		                 |
| pandas	|to remove, move, copy files.			 |
| shutil	|to remove, move, copy files.			 |
| pickel	|to remove, move, copy files.			 |
| imblearn	|offering a number of re-sampling techniques.	 |
| warnings	|to remove, move, copy files.			 |
| sklearn	|Machine learning library for the Python. 	 |
| bram		|info						 |


**Note:** Just use command below to install the required libary with correct version to run the program smoothly.

"pip install -r requiement.txt"


# **MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection**

* **Storage Requirement:** 100 GB 

* ***Method :*** Manually Download all the file from link save to the respective folder.
    * https://zenodo.org/record/3384388#.YGXS5ntR3-j


# Architecture

```
codit-usecase
│
│   README.md               :explains the project
│   requirements.txt        :packages to install to run the program
│   .gitignore              :specifies which files to ignore when pushing to the repository
│__   
   main		    	    :directory contain all the main .ipynb that create the machine model that train test and creates a pickel files.
    │
    │ Pump_ML_model.ipynb   :notebook that contain trainning, pickel creatin script and abnormal sound clusting for machine pump.
    │ Slider_ML_model.ipynb :notebook that contain trainning, pickel creatin script and abnormal sound clusting for machine slider.
    │ Fan_ML_model.ipynb    :notebook that contain trainning, pickel creatin script and abnormal sound clusting for machine fan.
    │ Valve_ML_model.ipynb  :notebook that contain trainning, pickel creatin script and abnormal sound clusting for machine valve.
    │
    │ saved_model      	    :directory contains all saved pickel files of the machine learning model.
    │ dataset		    ::directory contains all .pynb file that does the preprocessing and fearure extration .
      │__
	 processed_data     :directory contains the .csv files that contains the main machine features and information.
	 preprocessing.py   :
	 get_data.py        :

    
```

---

# Instruction
#### How to get 3d_location ploting

1. Setup python environment  `3.8.0`
2. Install all libaries `pip install -r requirements.txt`
3. Download important "sound data" files
4. Run the `.ipynb` in jupyter notebook
5. 
6. 
7. 

---
# Next Step

* Optimize the model with better data set and feature to genereate better predicition.
* Improving end user experiance.