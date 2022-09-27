# BasalCellExt
Code for analysing 2D and 3D confocal microscopy images of 3D EpiSC cultures

## Overview

Spheroid, lumen and membrane segmentations (object prediction files) were generated in [ilastik](https://www.ilastik.org/index.html). Object prediction files were then read and analysed in Python. <ins>**Important Note**</ins>: Membrane segmentations were only used in 2D analysis scripts. 

Nuclear segmentations (label files) were generated in Python using [StarDist](https://github.com/stardist/stardist). Label files were then imported in ilastik to train a positive/negative nuclear marker classifier. Classifier files were then read and analysed in Python.

## Installation

```console
git clone git@github.com:alymakhlouf/BasalCellExt.git  
cd BasalCellExt  
conda env create -f environment.yml
```

## Segmentation
### Monolayer Segmentation

```console
python StarDist2D_2D_Monolayer.py
```

### 3D EpiSC Segmentation

1. Generate spheroid, lumen and membrane segmentations in [ilastik](https://www.ilastik.org/index.html):
   1. run the [Pixel Classification](https://www.ilastik.org/documentation/pixelclassification/pixelclassification) workflow to generate a set of 'Pixel Prediction Map' files.
   2. For spheroids and lumens, input these 'Pixel Prediction Map' files into an [Object Classification Workflow (Inputs: Raw Data, Pixel Prediction Map)](https://www.ilastik.org/documentation/objects/objects). 
   3. For membranes, input these 'Pixel Prediction Map' files into a [Multicut Segmentation workflow](https://www.ilastik.org/documentation/multicut/multicut). This will generate a set of 'Object Prediction Map' files in H5 format. 
2. Choose between `Stardist2D, Stardist3D, StarDist3D_Basic` and `StarDist3D_Density` processing methods according to your image dimensions, now referred to as `<method>`
3. Preprocessing: run 
`python <method>_Pre-Processing.py`
unless your are using the `StarDist3D` method. This will generate cropped TIF files of individual spheroids in each image, as well as a nuclear label mask in HDF5 format for each spheroid, using [StarDist](https://github.com/stardist/stardist). You will need to input these cropped TIF files and their corresponding nuclear label files into the ilastik [Object Classification Workflow (Inputs: Raw Data, Segmentation)](https://www.ilastik.org/documentation/objects/objects) and use this to train a positive/negative nuclear marker classifier. This will generate a set of 'Object Prediction Map' files in H5 format.
4. Analyse the segmentations `python <method>.py`. Note: ensure that all relevant directories in the script map to the relevant 'Object Prediction Map' files. 

