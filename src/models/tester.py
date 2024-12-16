
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
import cv2
import os
import open3d as o3d
import copy
import json
import glob

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl
from tqdm import tqdm

from models.evaluator import DepthEvaluator
from dataset.utils import build_normal_xyz

def plot_heatmap(heatmap, save_path, xaxes_labels, yaxes_labels, titel="rmse", norm=True):
    fig, ax = plt.subplots()
    if norm:
        im = ax.imshow(heatmap, vmin=2.0, vmax=10.0, cmap='Reds')
    else:
        im = ax.imshow(heatmap, vmin=0.0, vmax=1.0, cmap='Blues')
    fig.colorbar(im)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(yaxes_labels)), labels=yaxes_labels)
    ax.set_xticks(np.arange(len(xaxes_labels)), labels=xaxes_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(yaxes_labels)):
        for j in range(len(xaxes_labels)):
            text = ax.text(j, i, np.round(heatmap[i, j],2),
                        ha="center", va="center", color="k")

    ax.set_title(titel)
    plt.ylabel("Linear FOV")
    plt.xlabel("Distortion FOV")
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, "{}.png".format(titel)))
    plt.savefig(os.path.join(save_path, "{}.pdf".format(titel)))
    #plt.savefig(os.path.join(save_path, "{}.pgf".format(titel)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Tester:
    def __init__(self, model, save_path, config, load=False):

        self.model = model
        if load:
            self.model.load_state_dict(torch.load(save_path))

        self.config = config
        self.save_path = os.path.dirname(save_path)
        time.sleep(3)

        # Timer
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        
        # Evaluator
        self.depth_eval = DepthEvaluator()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def create_inference_plot(self, DatasetEquirectangular, data_path_test, test_list_FOV = list(reversed(np.linspace(15,60,8).tolist())), test_list_DISTORTION = list(np.linspace(0.01, 1.5, 8).tolist())):
        
        test_list_FOV_str =  [str(np.round(float(i),2)) for i in test_list_FOV]
        test_list_DISTORTION_str =  [str(np.round(np.rad2deg(float(i)),2)) for i in test_list_DISTORTION]

        heatmaps = {"rmse": np.zeros((len(test_list_FOV),len(test_list_DISTORTION))), 
                    "a1": np.zeros((len(test_list_FOV),len(test_list_DISTORTION)))}
        
        h_list_rgb = []  
        h_list_pred = []
        h_list_gt = [] 
        h_list_normals = [] 
        h_list_normals_gt = [] 
        for u_coord, FOV in enumerate(test_list_FOV):
            v_list_rgb = []  
            v_list_pred = []
            v_list_gt = [] 
            v_list_normals = [] 
            v_list_normals_gt = [] 
            for v_coord, DISTORTION in enumerate(test_list_DISTORTION): 
                config= copy.deepcopy(self.config)
                config["FOV"] = [FOV]
                config["H"] = [512]#[32*i for i in range(2,32,4)]
                config["PITCH"] = [0]#list(range(0,360,1))
                config["ROLL"] = [0]
                config["YAW"] =[0]
                config["DISTORTION"] = [DISTORTION]
                config["ASPECT"] = [1.0]
                config["FLIP"] = [True]
                config["MAX_RANGE"]  = 100.0 # 80.0
                config["MIN_RANGE"] = 0.1
                config["MAX_DEPTH"] = 80.0

                depth_dataset_test = DatasetEquirectangular(data_path_test, config=config)
                dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=False, num_workers=1)

                self.model.eval()
                self.depth_eval.reset()
                # train one epoch
                for batch_idx, (color_img, range_img, unit_vec, normals, encoding) in tqdm(enumerate(dataloader_test)): #enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                    color_img, range_img, unit_vec, normals, encoding = color_img.to(self.device), range_img.to(self.device), unit_vec.to(self.device), normals.to(self.device), encoding.to(self.device) 
                    # run forward path
                    start_time = time.time()
                    self.start.record()
                    #outputs_prob, outputs_semantic = nocs_model(color_img, encoding)
                    outputs_mu  = self.model(color_img, encoding)
                    self.end.record()
                    curr_time = (time.time()-start_time)*1000
            
                    # Waits for everything to finish running
                    torch.cuda.synchronize()
                    
                    

                    

                    
                    #l1_diff_vis = torch.minimum(l1_diff, 5.0)/5.0

                    M = (range_img > 0).to(torch.float32)
                    xyz_img = M*outputs_mu
                    xyz_img_gt = M*unit_vec*range_img
                    l1_diff = M*torch.sqrt(torch.square(xyz_img)-torch.square(xyz_img_gt))

                    # build depth images
                    depth = M*xyz_img[:,2:3,...]
                    depth_gt = M*xyz_img_gt[:,2:3,...]
                    
                    M1 = (depth_gt > 0).to(torch.float32) * (depth_gt < config["MAX_DEPTH"]).to(torch.float32)
                    M2 = (depth > 0).to(torch.float32) * (depth_gt < config["MAX_DEPTH"]).to(torch.float32)
                    

                    rgb_img = color_img.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                    M1 = M1.permute(0, 2, 3, 1)[0,...,0].cpu().detach().numpy()
                    M2 = M2.permute(0, 2, 3, 1)[0,...,0].cpu().detach().numpy()
                    range_img  = depth_gt.permute(0, 2, 3, 1)[0,...,0].cpu().detach().numpy()
                    outputs_img = depth.permute(0, 2, 3, 1)[0,...,0].cpu().detach().numpy()
                    l1_diff = l1_diff.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                    l1_diff = np.uint8(255*np.minimum(l1_diff, 10.0)/10.0)

                    gt_img = cv2.applyColorMap(np.uint8(255*range_img/config["MAX_RANGE"]), cv2.COLORMAP_JET)#[::4,::4]
                    pred_img = cv2.applyColorMap(np.uint8(255*outputs_img/config["MAX_RANGE"]), cv2.COLORMAP_JET)#[::4,::4]

                    xyz_img = xyz_img.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                    xyz_img_gt = xyz_img_gt.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()

                    normals = build_normal_xyz(xyz_img)
                    normals_gt = build_normal_xyz(xyz_img_gt)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_img.reshape(-1,3))
                    pcd.colors = o3d.utility.Vector3dVector(np.float32(rgb_img[...,::-1].reshape(-1,3))/255.0)

                    pcdgt = o3d.geometry.PointCloud()
                    pcdgt.points = o3d.utility.Vector3dVector(xyz_img_gt[::4,::4].reshape(-1,3))


                    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
                    #o3d.visualization.draw_geometries([pcd, pcdgt])

                    v_list_rgb.append(np.uint8(rgb_img))
                    v_list_gt.append(gt_img)
                    v_list_pred.append(pred_img)
                    v_list_normals.append(normals)
                    v_list_normals_gt.append(normals_gt)
                    # cv2.imshow("rgb_img", np.uint8(rgb_img)[::4,::4])
                    # # cv2.imshow("normal_img", np.uint8(255*0.5*(normal_img+1))[::4,::4])
                    # # cv2.imshow("normal_gt", np.uint8(255*0.5*(normals+1))[::4,::4])
                    # cv2.imshow("gt", gt_img)
                    # cv2.imshow("pred", pred_img)
                    # cv2.waitKey(5)
            h_list_rgb.append(np.hstack(v_list_rgb))
            h_list_gt.append(np.hstack(v_list_gt))
            h_list_pred.append(np.hstack(v_list_pred))
            h_list_normals.append(np.hstack(v_list_normals))
            h_list_normals_gt.append(np.hstack(v_list_normals_gt))
        rgb_img_stacked = np.vstack(h_list_rgb)
        gt_img_stacked = np.vstack(h_list_gt)
        pred_img_stacked = np.vstack(h_list_pred)
        normal_img_stacked = np.vstack(h_list_normals)
        normal_img_gt_stacked = np.vstack(h_list_normals_gt)
        cv2.imwrite(os.path.join(self.save_path, "rgb_sample.png"), rgb_img_stacked)
        cv2.imwrite(os.path.join(self.save_path, "gt_sample.png"), gt_img_stacked)
        cv2.imwrite(os.path.join(self.save_path, "pred_sample.png"), pred_img_stacked)
        cv2.imwrite(os.path.join(self.save_path, "normals_sample.png"), np.uint8(255*(normal_img_stacked+1)/2))
        cv2.imwrite(os.path.join(self.save_path, "normals_gt_sample.png"), np.uint8(255*(normal_img_gt_stacked+1)/2))

    
    def __call__(self, DatasetEquirectangular, data_path_test, test_list_FOV = list(reversed(np.linspace(25,60,8).tolist())), test_list_DISTORTION = list(np.linspace(0.01, 1.5, 8).tolist())):
        result_dict = {}
        
        test_list_FOV_str =  [str(np.round(float(i),2)) for i in test_list_FOV]
        test_list_DISTORTION_str =  [str(np.round(np.rad2deg(float(i)),2)) for i in test_list_DISTORTION]

        heatmaps = {"rmse": np.zeros((len(test_list_FOV),len(test_list_DISTORTION))), 
                    "a1": np.zeros((len(test_list_FOV),len(test_list_DISTORTION)))}
        
                    
        for u_coord, FOV in enumerate(test_list_FOV):
            result_dict[FOV] = {}
            for v_coord, DISTORTION in enumerate(test_list_DISTORTION): 
                config= copy.deepcopy(self.config)
                config["FOV"] = [FOV]
                config["H"] = [512]#[32*i for i in range(2,32,4)]
                config["PITCH"] = [0]#list(range(0,360,1))
                config["ROLL"] = [0]
                config["YAW"] =[0]
                config["DISTORTION"] = [DISTORTION]
                config["ASPECT"] = [1.0]
                config["FLIP"] = [True]
                config["MAX_RANGE"]  = 100.0 # 80.0
                config["MIN_RANGE"] = 0.1
                config["MAX_DEPTH"] = 50.0

                depth_dataset_test = DatasetEquirectangular(data_path_test, config=config)
                dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=False, num_workers=16)

                self.model.eval()
                self.depth_eval.reset()
                # train one epoch
                for batch_idx, (color_img, range_img, unit_vec, normals, encoding) in tqdm(enumerate(dataloader_test)): #enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                    color_img, range_img, unit_vec, normals, encoding = color_img.to(self.device), range_img.to(self.device), unit_vec.to(self.device), normals.to(self.device), encoding.to(self.device) 
                    # run forward path
                    start_time = time.time()
                    self.start.record()
                    #outputs_prob, outputs_semantic = nocs_model(color_img, encoding)
                    outputs_mu  = self.model(color_img, encoding)
                    self.end.record()
                    curr_time = (time.time()-start_time)*1000
            
                    # Waits for everything to finish running
                    torch.cuda.synchronize()
                    
                    

                    xyz_img = outputs_mu
                    xyz_img_gt = unit_vec*range_img


                    # build depth images
                    depth = xyz_img[:,2:3,...]
                    depth_gt = xyz_img_gt[:,2:3,...]
                    
                    M = (depth_gt > 0).to(torch.float32) * (depth_gt < config["MAX_DEPTH"]).to(torch.float32)

                    self.depth_eval.update(depth, depth_gt, M)
                    

                        
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = self.depth_eval.compute_final_metrics()
                #result_dict[FOV] = {DISTORTION: {}}
                result_dict[FOV][DISTORTION] = {}
                result_dict[FOV][DISTORTION]["abs_rel"] = abs_rel
                result_dict[FOV][DISTORTION]["sq_rel"] = sq_rel
                result_dict[FOV][DISTORTION]["rmse"] = rmse
                result_dict[FOV][DISTORTION]["rmse_log"] = rmse_log
                result_dict[FOV][DISTORTION]["a1"] = a1
                result_dict[FOV][DISTORTION]["a2"] = a2
                result_dict[FOV][DISTORTION]["a3"] = a3
                heatmaps["rmse"][u_coord,v_coord]=rmse
                heatmaps["a1"][u_coord,v_coord]=a1
                print("metrics: FOV: {}, DIST: {}, abs_rel: {}, sq_rel: {}, rmse: {}, rmse_log: {}, a1: {}, a2: {}, a3: {}".format(FOV, DISTORTION, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3))
        # save results dict as json
        with open(os.path.join(self.save_path, "results.json"), 'w') as fp:
                        json.dump(result_dict, fp)
        # plot heatmaps
        plot_heatmap(heatmaps["rmse"], self.save_path, test_list_DISTORTION_str, test_list_FOV_str, titel="RMSE", norm=True)
        plot_heatmap(heatmaps["a1"], self.save_path, test_list_DISTORTION_str, test_list_FOV_str, titel="l1", norm=False)