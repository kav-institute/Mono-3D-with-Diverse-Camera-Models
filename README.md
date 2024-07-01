# Monocular 3D with Diverse Camera Models

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

## Datasets
### CARLA 9.14
You can generate a carla dataset by using the following repo:
https://github.com/kav-hareichert/CARLA_MV_RIG

Or you can use our pre-generated CARLA Panorama dataset:

### Training
You can modify the training parameter using the argparse.

```
Supported backbone types: 'resnet18', 'resnet34', 'resnet50', 'regnet_y_400mf','regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'.
Supported sensor encodings: 'CoordConv'[3], 'CAMConv'[4], 'CameraTensor'[5], 'UnitVec','Deflection'[1], 'UnitVec_Fourier'[6], and 'None'.
```


For CARLA run the training by:
```bash
python src/train_mono3D_CARLA.py --model_type resnet34 --encoding CameraTensor --learning_rate 0.001 --num_epochs 1000 --batch_size 8 --num_workers 16 --visualization
```




### References
[1] Reichert, Hannes and Hetzel, Manuel and Hubert, Andreas and Doll, Konrad and Sick, Bernhard,
    "Sensor Equivariance: A Framework for Semantic Segmentation with Diverse Camera Models.", in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition     (CVPR) Workshops, June 2024, pages 1254-1261.

[2] Semih, Orhan and Semih, Bastanlar, "Semantic segmentation of outdoor panoramic images.", in Springer SIViP 16, April 2022, pages 643–650. https://doi.org/10.1007/s11760-021-02003-3

[3] Liu, Rosanne and Lehman, Joel and Molino, Piero and Petroski Such, Felipe and Frank, Eric and Sergeev, Alex, and Yosinski, Jason, "An intriguing failing of convolutional neural networks and the CoordConv solution.", in Proceedings of the 32nd International Conference on Neural Information Processing Systems (NIPS'18). 2019, pages 9628–9639.

[4] Facil, Jose M. and Ummenhofer, Benjamin and Zhou, Huizhong and Montesano, Luis and Brox, Thomas and Civera, Javier "CAM-Convs: Camera-Aware Multi-Scale Convolutions for Single-View Depth.", in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2019.

[5] Ravi Kumar, Varun et al. “OmniDet: Surround View Cameras Based Multi-Task Visual Perception Network for Autonomous Driving.” in IEEE Robotics and Automation Letters 6 (2021): pages 2830-2837.

[6] Wang, Yifan et al. “Input-level Inductive Biases for 3D Reconstruction.” IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021): pages 6166-6176.



