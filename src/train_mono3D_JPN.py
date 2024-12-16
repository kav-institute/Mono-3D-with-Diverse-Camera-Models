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
from models.tester import Tester
import json
import random

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    config = {}
    config["FOV"] = list(range(15,70,1))
    config["H"] = [512]#[32*i for i in range(2,32,4)]
    config["PITCH"] = [0]#list(range(0,360,1))
    config["ROLL"] = [0]
    config["YAW"] =[0]
    config["DISTORTION"] = np.linspace(0.01, 1.5, 100).tolist() 
    config["ASPECT"] = [1.0]
    config["FLIP"] = [True, False]
    config["MAX_RANGE"]  = 100.0
    config["MIN_RANGE"] = 0.1
    config["SENSOR_ENCODING"] =  args.encoding #"UnitVec"
    config["BACKBONE"] = args.model_type
    config["INTERPOLATION"] = 'nearest'
    config["USE_ATTENTION"] = True
    config["SAVE_PATH"] = '/home/appuser/data/train_depth_JPN/{}_{}/model_final.pth'.format(args.model_type, args.encoding)

    config_test = copy.deepcopy(config)
    config_test["FOV"] = [35]
    config_test["H"] = [512]#[32*i for i in range(2,32,4)]
    config_test["PITCH"] = [0]
    config_test["DISTORTION"] = [0.01]
    config_test["ASPECT"] = [1.0]
    config_test["MAX_RANGE"] = 80.0
    config_test["MIN_RANGE"] = 0.1
    config_test["SENSOR_ENCODING"] = args.encoding #"UnitVec"
    # DataLoader
    data_path = list(glob.glob(f"/home/appuser/data/JPN_Dataset/refined/rgb/*.png"))
    random.shuffle(data_path)
    
    data_path_train = [(bin_path, bin_path.replace("rgb", "range"))  for bin_path in glob.glob(f"/home/appuser/data/Carla/train/rgb/*.png")]

    depth_dataset_train_CARLA = CarlaEquirectangular(data_path_train, config=config)

    data_path_train = [(bin_path, bin_path.replace("rgb", "range"))  for bin_path in data_path[0:int(0.7*len(data_path))]]
    data_path_test = [(bin_path, bin_path.replace("rgb", "range"))  for bin_path in data_path[int(0.7*len(data_path)):]]
    
    depth_dataset_train_JPN = JPNEquirectangular(data_path_train, config=config)
    depth_dataset_train  = torch.utils.data.ConcatDataset([depth_dataset_train_JPN])
    dataloader_train = DataLoader(depth_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    depth_dataset_test = JPNEquirectangular(data_path_test, config=config_test)
    dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Depth Estimation Network
    model = RangeNetWithFPN(backbone=args.model_type, meta_channel_dim=meta_channel_dict[config["SENSOR_ENCODING"]], attention=config["USE_ATTENTION"], interpolation_mode=config["INTERPOLATION"], learn_skymask=True)
    model.load_state_dict(torch.load("/home/appuser/data/train_depth_CARLA/{}_{}/model_final.pth".format(args.model_type, args.encoding)), strict=False)

    num_params = count_parameters(model)
    print("num_params", count_parameters(model))
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.1)

    # TensorBoard
    save_path = os.path.dirname(config["SAVE_PATH"])
    os.makedirs(save_path, exist_ok=True)

    # Save Config
    # save results dict as json
    with open(os.path.join(save_path, "config.json"), 'w') as fp:
        json.dump(config, fp)

    # Training loop
    num_epochs = args.num_epochs


    trainer = Trainer(model, optimizer, save_path, scheduler = scheduler, visualize = True, max_range_vis=config["MAX_RANGE"])
    trainer(dataloader_train, dataloader_test, num_epochs)

    # # run final test
    # tester = Tester(model, save_path=os.path.join(save_path, "model_final.pth"), config=config, load=False)
    # data_path_test = [(bin_path, bin_path.replace("rgb", "range"))  for bin_path in glob.glob(f"/home/appuser/data/JPN/test/rgb/*.png")]
    # tester(CarlaEquirectangular, data_path_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train Script for Monocular 3D')
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
