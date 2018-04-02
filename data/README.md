# Supported datasets

- CamVid

Note: When referring to the number of classes, the void/unlabelled class is excluded.

## CamVid Dataset

The Cambridge-driving Labeled Video Database (CamVid) is a collection of over ten minutes of high-quality 30Hz footage with object class semantic labels at 1Hz and in part, 15Hz. Each pixel is associated with one of 32 classes.

The CamVid dataset supported here is a 12 class version developed by the authors of SegNet. [Download link here](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid). For actual training, an 11 class version is used - the "road marking" class is combined with the "road" class.

More detailed information about the CamVid dataset can be found [here](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) and on the [SegNet GitHub repository](https://github.com/alexgkendall/SegNet-Tutorial).
