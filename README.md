# BasalCellExt
Code for analysing 2D and 3D confocal microscopy images

Spheroid, lumen and membrane segmentations (object prediction files) were generated in ilastik. Object prediction files were then read and analysed in Python. 

Nuclear segmentations (label files) were generated in Python using StarDist. Label files were then imported in ilastik to train the positive/negative nuclear marker classifier. Classifier files were then read and analysed in Python.

## Installation

git clone git@github.com:alymakhlouf/BasalCellExt.git  
cd BasalCellExt  
conda env create -f environment.yml

## How to run the scripts (2D analysis)  

If you have a single (representative) 2D-plane of 3D EpiSC-cultured cells, you need to separately generate segmentations in ilastik for the spheroid, the lumen and the cells. You will first need to run the 'Pixel Classification' workflow to generate a 'Pixel Prediction Map', and then input this 'Pixel Prediction Map' into an 'Object Classification' workflow to generate an 'Object Prediction Map' (see https://www.ilastik.org/documentation/pixelclassification/pixelclassification and https://www.ilastik.org/documentation/objects/objects for more information). 



