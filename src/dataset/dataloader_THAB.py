from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np

try:
    from dataset.ray_casting import EquirectangularRayCasting
    from dataset.projection_utils import equirect2Fisheye_FOV
except:
    from ray_casting import EquirectangularRayCasting
    from projection_utils import equirect2Fisheye_FOV

import open3d as o3d

import cv2 as cv
import glob
from sklearn import mixture
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch.distributions as D



def sample_from_cfg(param_list):
    param = np.random.choice(param_list)
    return param



class THABEquirectangular(Dataset):
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # Add more transformations if needed
        ])



    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):

        # Load Augmentation parameters
        FOV = sample_from_cfg(self.config["FOV"])
        
        FLIP = sample_from_cfg(self.config["FLIP"])
        
        h = int(sample_from_cfg(self.config["H"]))
        
        pitch = sample_from_cfg(self.config["PITCH"])
        roll = sample_from_cfg(self.config["ROLL"])
        yaw = sample_from_cfg(self.config["YAW"])
        d = sample_from_cfg(self.config["DISTORTION"])
        
        aspect = sample_from_cfg(self.config["ASPECT"])
        
        

        frame_path, range_FG_path, range_BG_path = self.data_path[idx]
        rgb_img = cv.imread(frame_path)

        # convert range to meter
        range_img_FG = cv.imread(range_FG_path, cv.IMREAD_UNCHANGED)/100.0
        range_img_BG = cv.imread(range_BG_path, cv.IMREAD_UNCHANGED)/100.0

        range_img = np.where(range_img_FG > 0, range_img_FG, range_img_BG)
        # clip range
        range_img = np.where(range_img>self.config["MAX_RANGE"], 0, range_img)
        range_img = np.where(range_img<self.config["MIN_RANGE"], 0, range_img)

        range_img_FG = np.where(range_img_FG>self.config["MAX_RANGE"], 0, range_img_FG)
        range_img_FG = np.where(range_img_FG<self.config["MIN_RANGE"], 0, range_img_FG)


        # build w from h and aspect ratio
        w = int(aspect*h) 
        # convert FOV to focal length
        f = 0.5 * w/np.tan(np.deg2rad(FOV))
        angle1, angle2 ,angle3 = roll, pitch, yaw

        outShape = (h,w)

        rgb_img, unit_vec = equirect2Fisheye_FOV(rgb_img, outShape,
                            f=f,
                            w_=d,
                            angles=[angle1, angle2 ,angle3],
                            interpolation = cv.INTER_CUBIC,
                            return_unit_vec = True)
    
        

        if self.config["SENSOR_ENCODING"] == "CoordConv":
             # build coord conv
            return_deflection = False
            return_unit_vec = False
            return_CameraTensor=True
        elif self.config["SENSOR_ENCODING"] == "CameraTensor":
            return_deflection = False
            return_unit_vec = False
            return_CameraTensor=True
        elif self.config["SENSOR_ENCODING"] == "UnitVec":
            return_deflection = False
            return_unit_vec = True
            return_CameraTensor=False

        elif self.config["SENSOR_ENCODING"] == "Deflection":
            return_deflection = True
            return_unit_vec = False
            return_CameraTensor=False
        else:
            return_deflection = False
            return_unit_vec = True
            return_CameraTensor=False

        range_img, encoding = equirect2Fisheye_FOV(range_img, (h,w),
                                    f=f,
                                    w_=d,
                                    angles=[angle1, angle2 ,angle3],
                                    interpolation = cv.INTER_NEAREST,
                                    return_deflection = return_deflection,
                                    return_unit_vec = return_unit_vec,
                                    return_CameraTensor=return_CameraTensor)
        
        range_img_FG, _ = equirect2Fisheye_FOV(range_img_FG, (h,w),
                                    f=f,
                                    w_=d,
                                    angles=[angle1, angle2 ,angle3],
                                    interpolation = cv.INTER_NEAREST,
                                    return_deflection = return_deflection,
                                    return_unit_vec = return_unit_vec,
                                    return_CameraTensor=return_CameraTensor)



        # chooce encoding
        if self.config["SENSOR_ENCODING"] == "CoordConv":
             # build coord conv
            endcoding_img = encoding[...,0:2]
        elif self.config["SENSOR_ENCODING"] == "CAMConv":
            # cam conv
            # [1]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Facil_CAM-Convs_Camera-Aware_Multi-Scale_Convolutions_for_Single-View_Depth_CVPR_2019_paper.pdf
            cx = w//2
            cy = h//2
            
            # following equ 1 & 2 of [1]
            cx_img = np.cumsum(np.ones((h,w)), axis=1)- 1 - cx
            cy_img = np.cumsum(np.ones((h,w)), axis=0)- 1 - cy
            
            # following equ 3 of [1]
            fovx_img = np.arctan(cx_img/f) * 180/np.pi
            fovy_img = np.arctan(cy_img/f) * 180/np.pi
            
            endcoding_img = np.stack([cx_img,cy_img, fovx_img, fovy_img], axis=-1)

        elif self.config["SENSOR_ENCODING"] == "CameraTensor":
            # OmniDet (non linear cam conv) 
            # [2]: https://arxiv.org/abs/2102.07448
            endcoding_img = encoding
        elif self.config["SENSOR_ENCODING"] == "UnitVec":
            endcoding_img = encoding

        elif self.config["SENSOR_ENCODING"] == "Deflection":
            encoding=np.nan_to_num(encoding,nan=0.0,neginf=0,posinf=0)
            endcoding_img = encoding[...,None]
        elif self.config["SENSOR_ENCODING"] == "Range":
            endcoding_img = range_img_FG[...,None]
        else:
            endcoding_img = encoding

        if not FLIP:
            rgb_img = rgb_img[:,::-1,:]
            unit_vec = unit_vec[:,::-1,:]
            range_img = range_img[:,::-1]
            range_img_FG = range_img_FG[:,::-1]
            endcoding_img = endcoding_img[:,::-1,:]

        range_img =  torch.as_tensor(range_img[...,None].transpose(2, 0, 1).astype("float32"))
        range_img_FG =  torch.as_tensor(range_img_FG[...,None].transpose(2, 0, 1).astype("float32"))
        color_img  =  torch.as_tensor(rgb_img.transpose(2, 0, 1).astype("float32"))
        unit_vec =  torch.as_tensor(unit_vec[...,0:3].transpose(2, 0, 1).astype("float32"))
        endcoding =  torch.as_tensor(endcoding_img.transpose(2, 0, 1).astype("float32"))

        return color_img, range_img, range_img_FG, unit_vec, endcoding


def main():
    import json
    set = "0007"

    image_path = "/home/appuser/data/Insignia/{}/Images/*.jpg".format(set)

    data_path_train = [(img, img.replace("Images","Range_FG").replace("jpg","png"), img.replace("Images","Range_BG").replace("jpg","png")) for img in glob.glob(image_path)]
    config = {}
    config["FOV"] = list(range(20,60,1))
    config["H"] = [512]#[int(128*i) for i in [2.0, 2.5,3.0, 3.5, 4.0]]
    config["PITCH"] = list(range(0,360,1))
    config["ROLL"] = [0]#list(range(-15,15,1))
    config["YAW"] =[0]
    config["DISTORTION"] = np.linspace(0.01, 2.0, 100).tolist() #[0.01, 0.5, 1.0, 1.5]
    config["ASPECT"] = [1.0]#[1.0, 1.5, 2.0]
    config["FLIP"] = [True, False]
    config["MAX_RANGE"] = 100.0
    config["MIN_RANGE"] = 2.5
    config["NUM_MIXTURES"] = 3
    config["SENSOR_ENCODING"] = "CameraTensor"
    depth_dataset_train = THABEquirectangular(data_path_train, config)
    dataloader_train = DataLoader(depth_dataset_train, batch_size=2, shuffle=True)
    for batch_idx, (color_img, range_img, range_img_FG, unit_vec, endcoding) in enumerate(dataloader_train):
        xyz_img_gt = unit_vec*range_img_FG
        M = (range_img_FG > 0).to(torch.float32)

        xyz_img_gt = (M*xyz_img_gt).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
        rgb_img = color_img.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
        range_img  = range_img.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
        unit_vec  = unit_vec.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
        mixtures = mixtures[0,...].cpu().detach().numpy()

     
        alpha = 0.4  # Weight for image1
        beta = 1.0 - alpha  # Weight for image2

        color_range = cv.applyColorMap(np.uint8(255*range_img/config["MAX_RANGE"]), cv.COLORMAP_JET)

        # Blend the images
        output = cv.addWeighted(np.uint8(rgb_img), alpha, color_range, beta, 0)
        output= np.uint8(np.where(range_img <=1,rgb_img, output))
       

        cv.imshow("rgb_img", np.uint8(rgb_img))
        cv.imshow("gt", cv.applyColorMap(np.uint8(255*range_img/config["MAX_RANGE"]), cv.COLORMAP_JET))

        cv.imshow("mixed", output)

        if (cv.waitKey(0) & 0xFF) == ord('q'):
            #xyz_img[...,1] = -xyz_img[...,1] +1.9
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_img_gt.reshape(-1,3))
            pcd.colors = o3d.utility.Vector3dVector(np.float32(rgb_img[...,::-1].reshape(-1,3))/255.0)

            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            o3d.visualization.draw_geometries([mesh, pcd])

                

if __name__ == "__main__":
    main()
