# Analysis of Glyph and Writing System Similarities using Siamese Neural Networks

## About
This repository includes the code and the data associated to the preprint **Analysis of Glyph and Writing System Similarities using Siamese Neural Networks**. 

Here is the abstract of the article:
*In this paper we use siamese neural networks to compare glyphs and writing systems. These deep learning models define distance-like functions and are used to explore and visualize the space of scripts by performing multidimensional scaling and clustering analyses. From 51 old and actual European, Mediterranean and Middle Eastern alphabets, we use a Ward-linkage hierarchical clustering and obtain 10 clusters of scripts including three isolated alphabets. To collect the glyph database we use the Noto family fonts that encode in a standard form the Unicode character repertoire. This approach has the potential to reveal connections among scripts and civilizations and to help the deciphering of ancient scripts.*

## Requirements
To run this project we recommend to create a new python environment and install the following python packages (see [requirements.txt](requirements.txt)):
```
keras==2.15.0
matplotlib==3.7.2
numpy==1.23.5
opencv_python==4.7.0.72
Pillow==10.2.0
scipy==1.12.0
skimage==0.0
tensorflow==2.15.0
tensorflow_intel==2.15.0
```

## Content description
The raw and the processed data used in this work are located in the folder [data](data). Here are the descriptions of the raw data:
* [fonts](data/raw/fonts) is composed of NotoSans font ttf files necessary to create the database of 51 writing systems.
* [omniglot_invented](data/raw/omniglot_invented) is composed of the invented scripts of the omniglot database (see https://github.com/brendenlake/omniglot).

Here are the descriptions of the processed data:
* [alphabets](data/processed/alphabets) is composed of the 51 writing systems in numpy arrays that are created from the font files.
* [distances](data/processed/distances) is composed of the distances between the scripts obtained with the siamese neural network.
* [omniglot_invented_augmented](data/processed/omniglot_invented_augmented) is composed of the omniglot database augmented by rotations, shears, zooms and shits that are used to train the model.

The python scripts located in the folder  [src](src) permit to recreate the processed data and to train the model. They are include for reproducibility but it is not necessary to run them to use the notebooks. Here are the descriptions of the scripts:
* [creation_alphabets_from_fonts.py](src/creation_alphabets_from_fonts.py) permits to create alphabets from font ttf files and to export them as numpy arrays.
* [dictionary_alphabets.py](src/dictionary_alphabets.py) stores the capital letter unicodes and the names of the font files of the scripts.
* [distance_functions.py](src/distance_functions.py) defines the siamese-based distance between writing systems.
* [model.py](src/model.py) defines the siamese neural network model.
* [model_prediction.py](src/model_prediction.py) predicts the distances between alphabets using the siamese neural network model.
* [model_training.py](src/model_training.py) trains the siamese neural network model and save it in the folder [models](models) for reuse.
* [omniglot_data_augmentation.py](src/omniglot_data_augmentation.py) uses rotations, shears, zooms and shits to augment the Omniglot dataset that will be use to train the siamese neural network model.

The two notebooks located in the folder [notebooks](notebooks) permit to produce the scientific results of the paper. Here are the descriptions of the notebooks:
* [1.space_glyphs.ipynb](notebooks/1.space_glyphs.ipynb) imports pairs of scripts to visualize them with multidimensional scaling analysis.
* [2.clustering_scripts.ipynb](notebooks/2.clustering_scripts.ipynb) imports of the distances between scripts to perform a clustering of the writing systems.

To get the fitted siamese neural network, the weights have to be downloaded from https://drive.google.com/file/d/1RpFisWyN2Q7BEJ3q76oXrC_g6mpHIqId/view?usp=sharing and extracted in the folder [models](models).

## Author
ANONYMIZED
