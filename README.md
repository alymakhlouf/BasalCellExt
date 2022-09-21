# BasalCellExt
Code for analysing 2D and 3D confocal microscopy images

## Installation

git clone git@github.com:alymakhlouf/BasalCellExt.git  
cd BasalCellExt  
conda env create -f environment.yml

## How to run the scripts  

Spheroid, lumen and cell segmentations (object prediction files) were generated in [ilastik](https://www.ilastik.org/index.html). Object prediction files were then read and analysed in Python. <ins>**Important Note**</ins>: Membrane segmentations were only used in 2D analysis scripts. 

Nuclear segmentations (label files) were generated in Python using [StarDist](https://github.com/stardist/stardist). Label files were then imported in ilastik to train a positive/negative nuclear marker classifier. Classifier files were then read and analysed in Python.

The following script can be run directly:
- StarDist2D_2D_Monolayer.py

Otherwise, you need to separately generate segmentations in [ilastik](https://www.ilastik.org/index.html) for the spheroid, the lumen and the membranes, before running the scripts. First, you will need to run the 'Pixel Classification' workflow to generate a set of 'Pixel Prediction Map' files (https://www.ilastik.org/documentation/pixelclassification/pixelclassification). For spheroids and lumens, input these 'Pixel Prediction Map' files into an 'Object Classification' workflow [Inputs: Raw Data, Pixel Prediction Map] (https://www.ilastik.org/documentation/objects/objects). For membranes, input these 'Pixel Prediction Map' files into a 'Multicut Segmentation' workflow (https://www.ilastik.org/documentation/multicut/multicut). This will generate a set of 'Object Prediction Map' files in H5 format. 

At this point, 
The following script can be run directly, making sure all relevant directories in the script map to the relevant 'Object Prediction Map' files:
- StarDist3D.py

The following scripts have a corresponding '...Pre-Processing' script that needs to be run first:

- StarDist2D.py
- StarDist3D_Basic.py
- StarDist3D_Density.py

The 'Pre-Processing' scripts will generate cropped TIF files of individual spheroids in each image, as well as a nuclear label mask in HDF5 format for each spheroid, using [StarDist](https://github.com/stardist/stardist). You will need to input these cropped TIF files and their corresponding nuclear label files into an 'Object Classification' Workflow [Inputs: Raw Data, Segmentation] (https://www.ilastik.org/documentation/objects/objects) and use this to train a positive/negative nuclear marker classifier. This will generate a set of 'Object Prediction Map' files in H5 format. 

At this point, 
The following scripts can be run directly, making sure all relevant directories in the scripts map to the relevant 'Object Prediction Map' files:

- StarDist2D.py
- StarDist3D_Basic.py
- StarDist3D_Density.py

