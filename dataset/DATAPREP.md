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
The sample in train are used for training, the samples in val are used for validation during the training in train_mono3d_CARLA.py and the samples in test are used for testing in test_mono3D.py.
> [!NOTE]
> We wrote a designated dataloader in src/dataset/dataloader_CARLA.py
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

> [!NOTE]
> We wrote a designated dataloader in src/dataset/dataloader_JPN.py
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

> [!NOTE]
> We wrote a designated dataloader in src/dataset/dataloader_Matterport.py
