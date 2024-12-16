import glob
import argparse
import torch
from torch.utils.data import DataLoader
from dataset.dataloader_CARLA import CarlaEquirectangular
from models.semanticFCN import RangeNetWithFPN


import numpy as np

from dataset.definitions import meta_channel_dict

import numpy as np

import matplotlib
import matplotlib as mpl
from models.tester import Tester
import json, os

def main(args):

    with open(args.save_path) as json_data:
        config = json.load(json_data)
        json_data.close()

    model = RangeNetWithFPN(backbone=config["BACKBONE"], meta_channel_dim=meta_channel_dict[config["SENSOR_ENCODING"]], attention=config["USE_ATTENTION"], interpolation_mode=config["INTERPOLATION"])
    # run final test
    tester = Tester(model, save_path=config["SAVE_PATH"], config=config, load=True)
    data_path_test = [(bin_path, bin_path.replace("rgb", "range"))  for bin_path in glob.glob(f"/home/appuser/data/Carla/val/rgb/*.png")][0:1]
    #tester(CarlaEquirectangular, data_path_test, test_list_FOV = list(reversed(np.linspace(25,60,8).tolist())), test_list_DISTORTION = list(np.linspace(0.01, 1.5, 8).tolist()))
    tester.create_inference_plot(CarlaEquirectangular, data_path_test, test_list_FOV = list(reversed(np.linspace(25,60,4).tolist())), test_list_DISTORTION = list(np.linspace(0.01, 1.5, 4).tolist()))


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test Script For Monocular 3D')
    parser.add_argument('--save_path', type=str, default='/home/appuser/data/train_depth_CARLA/resnet34_UnitVec/config.json',
                        help='path to config.json')
    args = parser.parse_args()

    main(args)

