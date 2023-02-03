# Voice Disorder Detection System

In this project there is an Automatic Voice Disorder Detection (AVDD) system. 
* **Frontend** ComParE feature set from OpenSMILE (https://audeering.github.io/opensmile/)
* **Backend** SVM classifier

## Databases
So far the databases included are the following (both are free available):
* **SVD** Saarbruecken Voice Database
(http://www.stimmdatenbank.coli.uni-saarland.de/help_en.php4)
* **AVFAD** Advanced Voice Function Assessment Database
(http://acsa.web.ua.pt/AVFAD.htm)

## Usage
For running an experiment you need to clone the repo

### Installation
1. Clone repo

```
git clone https://github.com/dayanavivolab/voicedisorder.git
```

2. Create and activate environment (Python >= 3.6)

```
python -m venv /scratch/user/miniconda3/envs/voicedisorder
source /scratch/user/miniconda3/envs/voicedisorder/bin/activate
```

### Run experiment
For running experiments you can use:
```
python bin/run_voicedisorder.py list_path kfold audio_type
```

```
Where: 
- audio_type: combiantion between audiotype and gender (audiotype_gender) 
audiotype (SVD): a_n, aiu, phrase, multi_a_n, multi_aiu, multi_phrase (multi is for multiple classification)
audiotype (AVFAD): aiu, phrase, read, spontaneous, multi_aiu, multi_phrase, multi_read, multi_spontaneous
gender: male, female, both
- kfold: number of folds
- list_path: path to the list with audios for processing 

Example: python bin/run_voicedisorder.py data/lst/Saarbruecken 5 phrase_both
```




