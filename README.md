# Monocular 3D with Diverse Camera Models
This code is for non-commercial use; please see the [license file](LICENSE) for terms.

## Introduction
Mono 3D from a single image is challenging due to ambiguities in scale. Camera intrinsic parameters are critically important for the
metric estimation from a single image, as otherwise, the
problem is ill-posed. In this work we aim to compare various approaches for camera intrinsic encodings.

<img src="/Images/IntrinsicVForward.png" width="400">

## Approach
The training data of many previous approaches contains limited images and types of cameras,
which challenges data diversity and network capacity. 
We propose a data augmentation pipeline from high-resolution equirectangular panoramas to generate diverse types of cameras during and mounting angles training and testing.
We use the [OmniCV-Lib](https://github.com/kaustubh-sadekar/OmniCV-Lib) (with some modifications) for our augmentation.

<img src="/Images/augmentation_ladybug.png" width="400">

You can control the range of parameters tuning the configs in the train_mono3D_*.py script.
Just pass lists over the parameter range you are interested in.
We use the FOV camera model to define diverse camera intrinsics with only a handful of parameters.

```python
config["FOV"] = list(range(20,70,1)) # linear FOV
config["H"] = [256, 512, 1024] # image hight 
config["PITCH"] = list(range(0,360,1)) # sensor pitch
config["ROLL"] = [0] # sensor roll
config["YAW"] =[0] # sensor yaw
config["DISTORTION"] = np.linspace(0.01, 1.0, 100).tolist() # non linear distortion
config["ASPECT"] = [1.0] # aspect ratio
config["FLIP"] = [True, False] # horizontal flip
```

> [!NOTE]
> We only support batch size of one for heterogenous image hights and aspect ratios

## Datasets
### CARLA 9.14
The CARLA dataset is a synthetic dataset containing equirectangular images, segmentation masks (semantic & instance), and range measures from simulated traffic scenes.
You can generate a carla dataset by using the following repo:
https://github.com/kav-hareichert/CARLA_MV_RIG

Or you can use our pre-generated [CARLA Panorama](https://drive.google.com/drive/folders/1WFCy2XeJugk82qbQTRmHuzYb6c8x4TqY?usp=sharing) dataset.

### Matterport 360
The Matterport Dataset provieds equirectangular images with range measured captured in indoor scenes by a Matterport 3D sensor.
You can download the [Matterport 360](https://researchdata.bath.ac.uk/1126/) [6] dataset.

### Fukuhoka Dataset for Place Categorization
The Fukuhoka Dataset is a high fidelity dataset for place categorization. It containes various static scenes from traffic related environemnts captured by a FARO 3D scanner and a equirectangular camera.
You can download the [Fukuhoka Dataset](http://robotics.ait.kyushu-u.ac.jp/kyushu_datasets/outdoor_dense.html) [8].

## Development environment:

### VS-Code:
The project is designed to be developed within vs-code IDE using remote container development.

### Setup Docker Container
In docker-compse.yaml all parameters are defined.
```bash
# Enable xhost in the terminal
sudo xhost +

# Add user to environment
sh setup.sh

# Build the image from scratch using Dockerfile, can be skipped if the image already exists or is loaded from docker registry
docker-compose build --no-cache

# Start the container
docker-compose up -d

# Stop the container
docker compose down
```
> [!CAUTION]
> xhost + is not a safe operation!


## Training
You can modify the training parameter using the argparse.

```
Supported backbone types: 'resnet18', 'resnet34', 'resnet50', 'regnet_y_400mf','regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'.
Supported sensor encodings: 'CoordConv'[2], 'CAMConv'[3], 'CameraTensor'[4], 'UnitVec','Deflection'[1], 'SHE'[5,7], and 'None'.
```


For CARLA run the training by:
```bash
python src/train_mono3D_CARLA.py --model_type resnet34 --encoding CameraTensor --learning_rate 0.001 --num_epochs 1000 --batch_size 8 --num_workers 16 --visualization
```
## License
Copyright © Technische Hochschule Aschaffenburg / University of Applied Sciences Aschaffenburg 2024. Patent Pending.
All rights reserved.
Please see the [license file](LICENSE) for terms.

## References
[1] Reichert, Hannes and Hetzel, Manuel and Hubert, Andreas and Doll, Konrad and Sick, Bernhard,
    "Sensor Equivariance: A Framework for Semantic Segmentation with Diverse Camera Models.", in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition     (CVPR) Workshops, June 2024, pages 1254-1261.

[2] Liu, Rosanne and Lehman, Joel and Molino, Piero and Petroski Such, Felipe and Frank, Eric and Sergeev, Alex, and Yosinski, Jason, "An intriguing failing of convolutional neural networks and the CoordConv solution.", in Proceedings of the 32nd International Conference on Neural Information Processing Systems (NIPS'18). 2019, pages 9628–9639.

[3] Facil, Jose M. and Ummenhofer, Benjamin and Zhou, Huizhong and Montesano, Luis and Brox, Thomas and Civera, Javier "CAM-Convs: Camera-Aware Multi-Scale Convolutions for Single-View Depth.", in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2019.

[4] Ravi Kumar, Varun et al. “OmniDet: Surround View Cameras Based Multi-Task Visual Perception Network for Autonomous Driving.” in IEEE Robotics and Automation Letters 6 (2021): pages 2830-2837.

[5] Wang, Yifan et al. “Input-level Inductive Biases for 3D Reconstruction.” IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021): pages 6166-6176.

[6] Rey-Area, Manuel and Mingze, Yuan and Christian, Richardt. “360MonoDepth: High-Resolution 360° Monocular Depth Estimation.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022): pages 3752-3762.

[7] Piccinelli, Luigi et al.  “UniDepth: Universal Monocular Metric Depth Estimation“ 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024): pages 10106-10116.

[8] Martinez Mozos O, Nakashima K, Jung H, Iwashita Y, Kurazume R. "Fukuoka datasets for place categorization" 2019 The International Journal of Robotics Research. pages 38(5):507-517. doi:10.1177/0278364919835603


