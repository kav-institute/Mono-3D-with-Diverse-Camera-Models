from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2 as cv
import glob, os

try:
    from dataset.projection_utils import equirect2Fisheye_FOV
except:
    from projection_utils import equirect2Fisheye_FOV

try:
    from dataset.utils import build_normal_xyz
except:
    from utils import build_normal_xyz

from struct import unpack 

def read_dpt(dpt_file_path):
    """read depth map from *.dpt file.
    :param dpt_file_path: the dpt file path
    :type dpt_file_path: str
    :return: depth map data
    :rtype: numpy
    """
    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(dpt_file_path)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % dpt_file_path)
    assert ext == '.dpt', exit('readFlowFile: fname %s should have extension ''.flo''' % dpt_file_path)

    fid = None
    try:
        fid = open(dpt_file_path, 'rb')
    except IOError:
        print('readFlowFile: could not open %s', dpt_file_path)

    tag = unpack('f', fid.read(4))[0]
    width = unpack('i', fid.read(4))[0]
    height = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % dpt_file_path)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (dpt_file_path, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (dpt_file_path, height))

    # arrange into matrix form
    depth_data = np.fromfile(fid, np.float32)
    depth_data = depth_data.reshape(height, width)

    fid.close()

    return depth_data

def sample_from_cfg(param_list):
    param = np.random.choice(param_list)
    return param



class MatterportEquirectangular(Dataset):
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
        
        frame_path, range_path = self.data_path[idx]

        rgb_img = cv.imread(frame_path)


        # convert range to meter
        range_img =read_dpt(range_path)
        # clip range
        range_img = np.where(range_img>self.config["MAX_RANGE"], 0, range_img)
        range_img = np.where(range_img<self.config["MIN_RANGE"], 0, range_img)

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
        else:
            endcoding_img = encoding

        if not FLIP:
            rgb_img = rgb_img[:,::-1,:]
            unit_vec = unit_vec[:,::-1,:]
            range_img = range_img[:,::-1]
            endcoding_img = endcoding_img[:,::-1,:]

        xyzi_img = unit_vec*range_img[...,None]

        normals = build_normal_xyz(xyzi_img)

        range_img =  torch.as_tensor(range_img[...,None].transpose(2, 0, 1).astype("float32"))
        color_img  =  torch.as_tensor(rgb_img.transpose(2, 0, 1).astype("float32"))
        unit_vec =  torch.as_tensor(unit_vec[...,0:3].transpose(2, 0, 1).astype("float32"))
        endcoding =  torch.as_tensor(endcoding_img.transpose(2, 0, 1).astype("float32"))
        normals =  torch.as_tensor(normals.transpose(2, 0, 1).astype("float32"))

        return color_img, range_img, unit_vec, normals, endcoding
    

def main():
    data_path_train = [(bin_path, bin_path.replace("_rgb.png", "_depth.dpt"))  for bin_path in glob.glob(f"/home/appuser/data/Matterport3D/data/*/*_rgb.png")]
    config = {}
    config["FOV"] = list(range(20,70,1))
    config["H"] = [256]#[int(128*i) for i in [2.0, 2.5,3.0, 3.5, 4.0]]
    config["PITCH"] = list(range(0,360,1))
    config["ROLL"] = list(range(-15,15,1))
    config["YAW"] =[0]
    config["DISTORTION"] = np.linspace(0.01, 1.0, 100).tolist() #[0.01, 0.5, 1.0, 1.5]
    config["ASPECT"] = [1.0]#[1.0, 1.5, 2.0]
    config["FLIP"] = [True, False]
    config["MAX_RANGE"] = 10.0
    config["MIN_RANGE"] = 0.0
    config["SENSOR_ENCODING"] = "CameraTensor"
    
    depth_dataset_train = MatterportEquirectangular(data_path_train, config)
    dataloader_train = DataLoader(depth_dataset_train, batch_size=1, shuffle=True, num_workers=1)

    for batch_idx, (color_img, range_img, xyz, normals, encoding) in enumerate(dataloader_train):
        rgb_img = color_img.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
        range_img  = range_img.permute(0, 2, 3, 1)[0,...,0].cpu().detach().numpy()
        encoding  = encoding.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
        cv.imshow("rgb_img", np.uint8(rgb_img))
        cv.imshow("gt", cv.applyColorMap(np.uint8(255*range_img/config["MAX_RANGE"]), cv.COLORMAP_JET))
        cv.waitKey(0)
                

if __name__ == "__main__":
    main()
