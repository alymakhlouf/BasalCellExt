# BasalCellExt
Code for analysing 2D and 3D confocal microscopy images

Spheroid, lumen and membrane segmentations (object prediction files) were generated in ilastik. Object prediction files were then read and analysed in Python. 

Nuclear segmentations (label files) were generated in Python using StarDist. Label files were then imported in ilastik to train the positive/negative nuclear marker classifier. Classifier files were then read and analysed in Python.

# Installation

git clone git@github.com:alymakhlouf/BasalCellExt.git

cd BasalCellExt

conda env create -f environment.yml
