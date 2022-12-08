# Latent XRD - EECS 182/282A - Class Project  

## Introduction

This repository contains the codebase of the class project which was done for the EECS 182/282 - Deep Neural Network at UC Berkeley. 

This project aims to derive a learned representation of crystalline materials that can be used for downstream models. In particular, we aim to use a compressed latent representation of a powder X-ray diffraction spectra as a chemical fingerprint.

### Objectives
* Learn a compact representation of X-Ray Diffraction data on crystals
* Validate the representation with a straightforward classification task.

## Installation
The `requirements.txt` file contains all Python libraries that the models are dependents on,
and they can be installed by:

```
pip install -r requirements.txt
```

## Usage

 We have incorporated three different types of models for getting the latent space representation of the XRD data. 1. Vanilla Autoencorder, 2, PerceiverIO 3, ViTMAE

 All these models were run on the Perlmutter Supercomputer at Lawrence Berkeley National Lab on Nvidia A100 GPUs with a VRAM of 40 GB. 

 ### File Structure

 ```
├── Vanilla_Autoencoder
│   ├── dataloader.py
│   ├── vanilla
│   └── vanilla.py
├── Perceiver_model
|   ├── dataloader.py
│   ├── perceiver
│   └── perceiver.py
├── VITMAE_model
│   ├── classification_xrd
│   ├── dataloader.py
│   ├── test_accuracy.ipynb
│   ├── vit_classification
│   ├── vit_classification.py
│   ├── vitmae_to_vitclassification.py
│   ├── vitmae_xrd
│   └── vitmae_xrd.py
├── readme.md
├── requirements.txt
 ```
 ### Trained Models

 Trained models for all the three types of architectures can be accessed through this [Google Drive Link](https://drive.google.com/drive/folders/14-6RnIkezw5N3i-KYK_dyL3ktxFABYgK?usp=sharing) (needs to be logged in to bMail to access this link). It has the same folder tree as the GitHub repo. 

 All three models are on it's respective folder containg the relevent dataloader functions. Job script used for submitting the python scripts on Perlmutter using SLURM is also included on the folders with the same name as python scripts. 

 Note -  Before running them please change the relevent data file path in `dataloader.py` and model save file path on the relevent python script according to your local configuration.

 All three models can be trained on randomly generated binary data set by  choosing the `binary_dataloader` in the `train_model` function in `\Vanilla_Autoencoder\vanilla.py` and `\Perceiver_model\perceiver.py`. And choosing `square_binary_dataloader` in the `train_model` function in `\ViTMAE_model\vitmae.py`

 ### 1. Vanilla Autoencoder 

 This model consists of Multilayer Perceptron based encoder and decoder architecture with ReLU activation fuctions. \
 You can run this model by following command :-
 ```
 python3 vanilla.py
 ```
  ### 2. Perceiver 

  This model is based on the Perceiver architecture which was presented in  [Jaegle et al. 2021](https://arxiv.org/pdf/2103.03206.pdf). This architecture builds upon Transformers with an “asymmetric attention mechanism”.
 You can run this model by following command :
 ```
 python3 perceiver.py
 ```
 ### 3. ViTMAE

 This model is based on the ViT architecture which was presented in [He et al. 2022](https://arxiv.org/pdf/2111.06377v2.pdf)

 * `square_xrd_dataloader` - XRD spectra is arranged as 100*100 array consisting 10 *10 patches 
 * `square_xrd_dataloader_gaussian` - XRD spectra is arranged as 100*100 array consisting 10 *10 patches blured with gaussian noise
 * `square_binary_dataloader` - Randomly generated binary data arranged as 100*100 array
 * `square_binary_dataset_gaussian` - Randomly generated binary data arranged as 100*100 array blured with gaussian noise 
 * `square_xrd_classification_dataloader` - XRD spectra is arranged as 100*100 array consisting 10 *10 patches with the respective classification of crystal (ranges 0,6)
 You can run this model by following command :-
 ```
 python3 vitmae.py
 ```
 
 Once the ViTMAE model is trained you can use the saved model for classification task through `ViTForImageClassification`. \
 You can run this model by following command to generate the `ViTForImageClassification` and the model will be saved on `./ViTMAE_model/classification_xrd/`:-
 ```
 python3 vitmae_to_vitclassification.py
 ```
 Once you save the model with encorder and the classification head you can  train `ViTForImageClassification` by running the following command
  ```
 python3 vit_classification.py
 ```
 When you have trained the `ViTForImageClassification`, you can run the trained model on your test data set to see the accuracy by using the `ViTMAE_model/test_accuracy.ipynb` Jupyter notebook


## Contributors
Orion Archer Cohen - orioncohen@berkeley.edu \
Shreyas Krishnaswamy - shrekris@berkeley.edu  \
Hasitha Sithadara Wijesuriya - hasitha@berkeley.edu