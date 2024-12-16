import glob
import argparse
import torch
from torch.utils.data import DataLoader
from dataset.dataloader_CARLA import CarlaEquirectangular
from dataset.dataloader_Ouster import OusterEquirectangular
from dataset.dataloader_JPN import JPNEquirectangular
from dataset.dataloader_Matterport import MatterportEquirectangular
from models.semanticFCN import RangeNetWithFPN
import torch.optim as optim
import tqdm
import time
import numpy as np
import cv2
import os
import open3d as o3d
import copy
from dataset.definitions import meta_channel_dict
from models.trainer import Trainer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visualize_semantic_segmentation_cv2(mask, class_colors):
    """
    Visualize semantic segmentation mask using class colors with cv2.

    Parameters:
    - mask: 2D NumPy array containing class IDs for each pixel.
    - class_colors: Dictionary mapping class IDs to BGR colors.

    Returns:
    - visualization: Colored semantic segmentation image in BGR format.
    """
    h, w = mask.shape
    visualization = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        visualization[mask == class_id] = color

    return visualization

def main(args):
    config = {}
    config["FOV"] = list(range(15,60,1))
    config["H"] = [512]#[32*i for i in range(2,32,4)]
    config["PITCH"] = list(range(0,360,1))
    config["ROLL"] = [0]
    config["YAW"] =[0]
    config["DISTORTION"] = np.linspace(0.01, 1.5, 100).tolist() 
    config["ASPECT"] = [1.0]
    config["FLIP"] = [True, False]
    config["MAX_RANGE"]  = 20.0
    config["MIN_RANGE"] = 0.1
    config["SENSOR_ENCODING"] =  args.encoding #"UnitVec"

    config_test = copy.deepcopy(config)
    config_test["FOV"] = [35]
    config_test["H"] = [512]#[32*i for i in range(2,32,4)]
    config_test["PITCH"] = [0]
    config_test["DISTORTION"] = [0.01]
    config_test["ASPECT"] = [1.0]
    config_test["MAX_RANGE"] = 10.0
    config_test["MIN_RANGE"] = 0.1
    config_test["SENSOR_ENCODING"] = args.encoding #"UnitVec"

    # DataLoader
    data_path_train = [(bin_path, bin_path.replace("_rgb.png", "_depth.dpt"))  for bin_path in glob.glob(f"/home/appuser/data/Matterport/data/*/*_rgb.png")]
    depth_dataset_train = MatterportEquirectangular(data_path_train, config=config)
    dataloader_train = DataLoader(depth_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    data_path_test = [(bin_path, bin_path.replace("_rgb.png", "_depth.dpt"))  for bin_path in glob.glob(f"/home/appuser/data/Matterport/data/*/*_rgb.png")]
    depth_dataset_test = MatterportEquirectangular(data_path_test, config=config)
    dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=True, num_workers=args.num_workers)

    # Depth Estimation Network
    model = RangeNetWithFPN(backbone=args.model_type, meta_channel_dim=meta_channel_dict[config["SENSOR_ENCODING"]])

    num_params = count_parameters(model)
    print("num_params", count_parameters(model))
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # TensorBoard
    save_path ='/home/appuser/data/train_depth_Matterport/{}_{}/'.format(args.model_type, args.encoding)

    # Training loop
    num_epochs = args.num_epochs

    trainer = Trainer(model, optimizer, save_path, visualize = True, max_range_vis=config["MAX_RANGE"])
    trainer(dataloader_train, dataloader_test, num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train script for SemanticKitti')
    parser.add_argument('--model_type', type=str, default='resnet34',
                        help='Type of the model to be used (default: resnet50)')
    parser.add_argument('--encoding', type=str, default="UnitVec",
                        help='Type of the model to be used (default: CameraTensor)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the model (default: 0.001)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs for training (default: 50)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of data loading workers (default: 1)')
    parser.add_argument('--rotate', action='store_true',
                        help='Whether to apply rotation augmentation (default: False)')
    parser.add_argument('--flip', action='store_true',
                        help='Whether to apply flip augmentation (default: False)')
    parser.add_argument('--visualization', action='store_true',
                        help='Toggle visualization during training (default: False)')
    args = parser.parse_args()

    main(args)

