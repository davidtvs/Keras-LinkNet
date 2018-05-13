# Supported datasets

- CamVid
- Cityscapes

Note: When referring to the number of classes, the void/unlabelled class is excluded.

## CamVid Dataset

The Cambridge-driving Labeled Video Database (CamVid) is a collection of over ten minutes of high-quality 30Hz footage with object class semantic labels at 1Hz and in part, 15Hz. Each pixel is associated with one of 32 classes.

The CamVid dataset supported here is a 11 class version built from [``camvid``](https://github.com/davidtvs/camvid). The split between train, validation, and testing datasets mirrors the one used on [SegNet](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid).

Detailed information about the CamVid dataset can be found [here](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/).

## Cityscapes

Cityscapes is a set of stereo video sequences recorded in streets from 50 different cities with 34 different classes. There are 5000 images with fine annotations and 20000 images coarsely annotated.

The version supported here is the finely annotated one with 19 classes.

For more detailed information see the official [website](https://www.cityscapes-dataset.com/) and [repository](https://github.com/mcordts/cityscapesScripts).

The dataset can be downloaded from https://www.cityscapes-dataset.com/downloads/. At this time, a registration is required to download the data.
