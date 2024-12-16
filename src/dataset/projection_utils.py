import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2 as cv
import time
import torch.distributions as dist
import open3d as o3d

def model_summary(model):
  print("model_summary")
  print()
  print("Layer_name"+"\t"*7+"Number of Parameters")
  print("="*100)
  model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
  layer_name = [child for child in model.children()]
  j = 0
  total_params = 0
  print("\t"*10)
  for i in layer_name:
    print()
    param = 0
    try:
      bias = (i.bias is not None)
    except:
      bias = False  
    if not bias:
      param =model_parameters[j].numel()+model_parameters[j+1].numel()
      j = j+2
    else:
      param =model_parameters[j].numel()
      j = j+1
    print(str(i)+"\t"*3+str(param))
    total_params+=param
  print("="*100)
  print(f"Total Params:{total_params}")       

class EquirectangularRayCasting:
    def __init__(self, height=2048, origin=(0,0,0)):
        
        self.origin = origin
        self.height = height
        width = 2*height
        self.width = width
        phi_min, phi_max = [-np.pi, np.pi]
        theta_min, theta_max = [-np.pi/2, np.pi/2]
        # assuming uniform distribution of rays
        bins_h = np.linspace(theta_min, theta_max, height)[::-1]
        bins_w = np.linspace(phi_min, phi_max, width)[::-1]
        
        theta_img = np.stack(width*[bins_h], axis=-1)
        phi_img = np.stack(height*[bins_w], axis=0)

        
        x = np.sin(theta_img+np.pi/2)*np.cos(phi_img)#+np.pi)
        y = np.sin(theta_img+np.pi/2)*np.sin(phi_img)#+np.pi)
        z = -np.cos(theta_img+np.pi/2)

        self.ray_img = np.stack([x,y,z],axis=-1)
        self.origin_img_ = np.ones_like(self.ray_img)

        self.merged_mesh = o3d.geometry.TriangleMesh()
        self.createScene()

        

    def createScene(self):
        self.scene = o3d.t.geometry.RaycastingScene()
        #self.scene = o3d.geometry.RaycastingScene()
    
    def get_colors(self, mesh, primitive_ids):
        valid_mask = np.where(primitive_ids != self.scene.INVALID_ID, 1, 0)
        primitive_ids_masked = valid_mask*primitive_ids
        vertex_indices = np.asarray(mesh.triangles)[primitive_ids_masked.flatten()][...,0]
        colors = np.asarray(mesh.vertex_colors)[vertex_indices]
        colors = colors.reshape(primitive_ids.shape + (3,))
        colors = np.where(valid_mask[...,None], colors, [0,0,0])
        return colors


    def addMesh(self, mesh):
        self.merged_mesh += mesh # TODO: add up triangle meshes (does not work for t.geometry)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        _id = self.scene.add_triangles(mesh)
        return _id

    def rayCast(self, T=np.eye(4)):
        R = T[0:3,0:3]
        origin = T[0:3,3]
        self.ray_img
        ray_img = np.einsum("ik,...k->...i", R, self.ray_img)
        rays = np.concatenate([np.array(origin)*self.origin_img_, ray_img], axis=-1)
        rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        ans = self.scene.cast_rays(rays)
        return ans, rays
    
    def computeDistance(self, query_points):
        query_points = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
        distance = self.scene.compute_signed_distance(query_points)
        return distance

pseudo_ground_plane = o3d.geometry.TriangleMesh.create_box(width=100.0, height=100.0, depth=0.1)
pseudo_ground_plane.translate((-50.0,-50.0,-1.9))
ray_caster = EquirectangularRayCasting()

ray_caster.createScene()
ray_caster.addMesh(pseudo_ground_plane)

rc, rays = ray_caster.rayCast()
hit = rc['t_hit'].isfinite()
points = rays[hit][:,:3] + rays[hit][:,3:]*rc['t_hit'][hit].reshape((-1,1))
ground_prior_img = np.nan_to_num(rc['t_hit'].numpy(),nan=0.0,posinf=0.0,neginf=0.0)

def rmat(roll,
         pitch,
         yaw):


    #R = cv.Rodrigues(np.deg2rad([alpha, beta, gamma]))[0]
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    yawMatrix = np.array([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw), np.cos(yaw), 0],
    [0, 0, 1]
    ])

    pitchMatrix = np.array([
    [np.cos(pitch), 0, np.sin(pitch)],
    [0, 1, 0],
    [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    rollMatrix = np.array([
    [1, 0, 0],
    [0, np.cos(roll), -np.sin(roll)],
    [0, np.sin(roll), np.cos(roll)]
    ])

    R = yawMatrix@pitchMatrix@rollMatrix
    return R


def RPY_to_Axis(roll,pitch,yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    yawMatrix = np.matrix([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw), np.cos(yaw), 0],
    [0, 0, 1]
    ])

    pitchMatrix = np.matrix([
    [np.cos(pitch), 0, np.sin(pitch)],
    [0, 1, 0],
    [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    rollMatrix = np.matrix([
    [1, 0, 0],
    [0, np.cos(roll), -np.sin(roll)],
    [0, np.sin(roll), np.cos(roll)]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix

    theta = np.arccos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
    multi = 1 / (2 * np.sin(theta))

    rx = multi * (R[2, 1] - R[1, 2]) * theta
    ry = multi * (R[0, 2] - R[2, 0]) * theta
    rz = multi * (R[1, 0] - R[0, 1]) * theta
    rx = np.rad2deg(rx)
    ry = np.rad2deg(ry)
    rz = np.rad2deg(rz)
    return rx, ry, rz

def to_deflection_coordinates(x,y,z):
    # To cylindrical
    p = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    # To spherical   
    theta = np.arctan2(p, z) #+ np.pi/4
    return phi, theta
  
 
def equirect2Fisheye_FOV(img,
                             outShape,
                             f=50,
                             w_=0.5,
                             angles=[0, 0, 0],
                             interpolation = cv.INTER_CUBIC,
                             max_FOV = np.deg2rad(180)/2,
                             return_deflection = False,
                             return_unit_vec = False,
                             return_CameraTensor = False):

        Hd = outShape[0]
        Wd = outShape[1]
        f = f
        w_ = w_

        Hs, Ws = img.shape[:2]

        Cx = Wd / 2.0
        Cy = Hd / 2.0

        x = np.linspace(0, Wd - 1, num=Wd, dtype=np.float32)
        y = np.linspace(0, Hd - 1, num=Hd, dtype=np.float32)

        x, y = np.meshgrid(range(Wd), range(Hd))
        

        mx = (x - Cx) / f
        my = (y - Cy) / f

        rd = np.sqrt(mx ** 2 + my ** 2)

        Ps_x = mx * np.sin(rd * w_) / (2 * rd * np.tan(w_ / 2) + 1e-10)
        Ps_y = my * np.sin(rd * w_) / (2 * rd * np.tan(w_ / 2) + 1e-10)
        Ps_z = np.cos(rd * w_)
        
        # assume no rotation for deflection metric
        Ps = np.stack((Ps_x, Ps_y, Ps_z), -1)
        
        # build normals on the sensor
        Ps_ = Ps /np.linalg.norm(Ps,axis=-1, keepdims=True)
        unit_vec = Ps_#Ps
        Ps_x_, Ps_y_, Ps_z_ = np.split(Ps_, 3, axis=-1)
        Ps_x_ = Ps_x_[:, :, 0]
        Ps_y_ = Ps_y_[:, :, 0]
        Ps_z_ = Ps_z_[:, :, 0]
        _, theta_ = to_deflection_coordinates(Ps_x_, Ps_y_, Ps_z_)

        # For camera tensor from: (https://arxiv.org/pdf/2102.07448)
        inc_angle_w = np.arctan2(Ps_z_, Ps_x_)-np.pi/2
        inc_angle_h = np.arctan2(Ps_z_,Ps_y_)-np.pi/2
        cx, cy = x-Cx, y-Cy
        normed_x = cx/(Wd / 2.0)
        normed_y = cy/(Hd / 2.0)

        camera_tensor = np.stack([cx,cy,inc_angle_w, inc_angle_h, normed_x,normed_y], axis=-1)

        deflection = theta_
        
        alpha = angles[0]
        beta = angles[1]
        gamma = angles[2]

        R = rmat(alpha, beta, gamma)
        R = np.matmul(
            np.matmul(rmat(-90, 0, 0), rmat(0, 90, 0)),R
        )

        Ps = np.stack((Ps_x, Ps_y, Ps_z), -1)
        
        Ps = np.matmul(Ps, R.T)


        Ps_x, Ps_y, Ps_z = np.split(Ps, 3, axis=-1)
        Ps_x = Ps_x[:, :, 0]
        Ps_y = Ps_y[:, :, 0]
        Ps_z = Ps_z[:, :, 0]

        theta = np.arctan2(Ps_y, Ps_x)
        phi = np.arctan2(Ps_z, np.sqrt(Ps_x ** 2 + Ps_y ** 2))
        

        a = 2 * np.pi / (Ws - 1)
        b = np.pi - a * (Ws - 1)
        map_x = (1.0 / a) * (theta - b)

        a = -np.pi / (Hs - 1)
        b = np.pi / 2
        map_y = (1.0 / a) * (phi - b)

        output = cv.remap(
            img,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation,
            borderMode=cv.BORDER_WRAP,
        )
        

        if len(output.shape)==3:
            output = np.where(deflection[...,None]>max_FOV, 0, output)
        else:
            output = np.where(deflection>max_FOV, 0, output)

        deflection = np.where(deflection>max_FOV, 0, deflection)
        camera_tensor = np.where(deflection[...,None]>max_FOV, 0, camera_tensor)
        unit_vec = np.where(deflection[...,None]>max_FOV, 0, unit_vec)
        if return_deflection:
            return output, deflection
        if return_unit_vec:
            return output, unit_vec
        if return_CameraTensor:
            return output, camera_tensor
        return output
 




