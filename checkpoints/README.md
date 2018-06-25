# Pre-trained encoder (ResNet18)

The encoder in LinkNet is based on ResNet18. The pre-trained ImageNet weights are stored in: [``linknet_encoder_weights.h5``].

# Checkpoints

Keras-LinkNet has been trained on the CamVid and Cityscapes datasets. The models and weights can be downloaded from this [Google Drive link](https://drive.google.com/drive/folders/1BrvYVoRr7mhkSHTkdeQQ-Ih-ty0eGWO9?usp=sharing).

|                                Dataset                               | Classes <sup>1</sup> | Input resolution | Batch size | Mean IoU (%) |
|:--------------------------------------------------------------------:|:--------------------:|:----------------:|:----------:|:------------:|
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |          12          |      960x480     |      2     |     47.15<sup>2</sup>    |
|           [Cityscapes](https://www.cityscapes-dataset.com/)          |          20          |     1024x512     |      2     |     53.37<sup>3</sup>    |

<sup>1</sup> Includes the unlabeled/void class.<br/>
<sup>2</sup> Test set.<br/>
<sup>3</sup> Validation set.
