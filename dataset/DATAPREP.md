# Data Preperation
## CARLA
For the CARLA dataset ensure the following folder structure:
```
dataset
└───Carla
│   └───train
│   └───test
│   └───val
│   │   └───range
│   |   │   *.png
│   │   └───rgb
│   |   │   *.png
```
The sample in train are used for training, the samples in val are used for validation during the training and the samples in test are used for testing in test_mono3D.py.
## Fukuhoka Dataset for Place Categorization
For the Fukuhoka dataset we need to to some preprocessing first. Ensure the following datastructure:
```
dataset
└───JPN_Dataset
│   └───Coast
│   └───Parking
│   └───...
```
Than run the following command:
```
python dataset/MPO_JPN_Preprocess.py
```
This results in a datastructure for the dataloader:
```
dataset
└───JPN_Dataset
│   └───refined
│   │   └───range
│   |   │   *.png
│   │   └───rgb
│   |   │   *.png
```
> [!NOTE]
> We did not define a train/test/val split for this dataset, since it is not used in our paper!
> Feel free to shuffle the samples on your own demands!
> We also dont provide a dsignated script for training. You can take inspiration from the train_mono3D_CARLA.py to write your own.
## Matterport 360
```
dataset
└───Matterport
│   └───Train
│   └───Test
│   └───Val
│   │   └───1LXtFkjw3qL
│   │   │   *_rgb.png
│   │   │   *_rgb.dpt
```
> [!NOTE]
> We did not define a train/test/val split for this dataset, since it is not used in our paper!
> Feel free to shuffle the sequences on your own demands!
> We also dont provide a dsignated script for training. You can take inspiration from the train_mono3D_CARLA.py to write your own.
