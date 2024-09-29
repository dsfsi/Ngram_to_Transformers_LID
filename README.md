DSFSI South African Language Identification (za-lid) Githup Repository
==============================

_This documentation is aimed to help provide information that explains what a project is about._

Last updated: September 2024

## Table of contents 

1. [Project Description](#project-description) 
2. [Getting Started](#getting-started)
3. [Authors](#authors)
4. [More Information](#more-information)

## Project Description 
-----------

This Github Repository contains datasets extracted from Vuk'zenzele () used to train various Language identification (LID) models such as N-grams, Machine Learning models (e.g SVM, Logistic Regression, K Nearest Neighbor, and Naive Bayes), and Transformer models (BERT, DistilBERT, mBERT, RemBERT, XLMr, AfroLM, Afro-XLMR, AfriBERTa, Serengeti, etc). The repo also contains code on how to use available LID models such as GlotLID, OpenLID, AfroLIF, and CLD V3.

## Getting Started
-----------
_This section provides the necessary information for a user to be able to run the code locally._

### Prerequisites 

All code is developed using Python.  : 

- Python 3.* 

### Installation 

1. Run the requirements.txt to install all the required libraries, modules, and packages.  

```
Run
pip install -r requirements.txt 
If all dependencies did not install successfully, or having compatability issues, the dependencies you need are:
sklearn
pandas
seaborn
matplotlib
numpy 
torch
transformers
nltk
tqdm
seqeval

```

### Usage 

All code and datasets is contained inside the src folder: 

1. To use the code , follow the steps: 

```
* For each model category (N-grams, ML, or Transformers) ensure all dependencies are installed
* For each Categoory of models there is script folder  (E.g LID_Toold/scripts). This folder contains a bash file that runs the appropriate python file five times and saves results in a destination folder (may need to change the destination folder)
* To run the bash simply run nohup bash 'script_name.sh' > 'output_text_file.txt' & . This line ensures the execution does not stop even if termibal is closed.
* Once run is complete all output files, plots, etc, will be saved to a destination folder for you to view.
* NB: For files with no script, you may ned to run the python file directly

```

## Authors 
-----------

* Written by : Thapelo Sindane And Vukosi Marivate
* Contact details : sindane.thapelo@tuks.co.za

### Contributions  

This is optional and provides information about which  and how each of the developers contributed. 

## How to Reference 
---------

## Licence

DSFSI South African Language Identification (za-lid) © 2024 by Thapelo Sindane, Vukosi Marivate is licensed under CC BY-SA 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by-sa/4.0/
