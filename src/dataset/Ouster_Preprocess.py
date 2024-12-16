import torch
import numpy as np
import open3d as o3d
import cv2 
import copy
from ouster.sdk import client, pcap
#from ouster import pcap
from contextlib import closing
import time
import os
from ouster.sdk.client._utils import AutoExposure, BeamUniformityCorrector


def clip_extremes(image, lower_percentile=0.5, upper_percentile=99.5):
    """
    Clip the extremes of the image based on percentiles.
    """
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)
    clipped_image = np.clip(image, lower_bound, upper_bound)
    return clipped_image

def normalize_image(image):
    """
    Normalize the image data to the 0-255 range.
    """
    image_min = np.min(image)
    image_max = np.max(image)
    #normalized_image = np.minimum(image/image_max,image_max)*255
    normalized_image = 2*(image - image_min) / (image_max - image_min) * 255
    return normalized_image

def auto_exposure(intensity_data, target_mean=255):
    """
    Adjust the exposure settings based on the mean intensity.
    """
    current_mean = np.mean(intensity_data)
    adjustment_factor = target_mean / current_mean
    return adjustment_factor


pcap_path = "/home/appuser/data/Ouster/0003/OS-2-128-992317000331-2048x10.pcap"
metadata_path = "/home/appuser/data/Ouster/0003/OS-2-128-992317000331-2048x10.json"

with open(metadata_path, 'r') as f:
    file = f.read()
    metadata = client.SensorInfo(file)

source = pcap.Pcap(pcap_path, metadata)
load_scan = lambda:  client.Scans(source)

NearIR_path = "/home/appuser/data/Ouster/0003/NearIR/"
depth_path = "/home/appuser/data/Ouster/0003/Range/"

ae = AutoExposure()
buc = BeamUniformityCorrector()


xyzlut = client.XYZLut(metadata)
with closing(load_scan()) as stream:
    for i, scan in enumerate(stream):

        xyz = xyzlut(scan)
        xyz = client.destagger(stream.metadata, xyz)
        
        raydrop = 1/np.linalg.norm(xyz,axis=-1)
        raydrop = np.uint8(255*np.where(np.isnan(raydrop) | np.isinf(raydrop), 0, 1))
        h,w,c = xyz.shape
        scalar_Id = np.array(range(0,h*w)).astype(np.int32)
        reflectivity_field = scan.field(client.ChanField.REFLECTIVITY)
        reflectivity_img = client.destagger(stream.metadata, reflectivity_field)
        NearIR_field = scan.field(client.ChanField.NEAR_IR)
        NearIR_img = client.destagger(stream.metadata, NearIR_field).astype(np.float32)

        ae(NearIR_img)
        buc(NearIR_img, update_state=True)

        NearIR_img_normed = np.uint8(np.clip(np.rint(NearIR_img*255), 0, 255))

        #NearIR_img_normed = normalize_image(NearIR_img)
        #adjustmetn_factor = 128/np.median(NearIR_img_normed)
        #NearIR_img_normed = np.uint8(np.minimum(adjustmetn_factor*NearIR_img_normed,255))
        range_field = scan.field(client.ChanField.RANGE)
        range_img = client.destagger(stream.metadata, range_field)
        color_img = cv2.applyColorMap(np.uint8(reflectivity_img), cv2.COLORMAP_PARULA)
        range_img = np.linalg.norm(xyz,axis=-1)

        # Define the maximum depth range
        max_depth = 250.0  # Maximum depth in meters

        # Scale the depth image to the range 0-65535
        img_range_scaled = (range_img / max_depth * 65535).clip(0, 65535).astype(np.uint16)

        #normals = build_normal_xyz(xyz)
        cv2.imshow("NearIR_img_normed",NearIR_img_normed)
        if scan.timestamp[0] != 0:
            # Create black rows with the same width and channels as the image
            black_rows = np.zeros((int(((w/2)-128)/2), w), dtype=np.uint16)

            # Add the black rows to the bottom of the image
            img_range_scaled = np.vstack((black_rows, img_range_scaled, black_rows))
            black_rows = np.uint8(black_rows)
            NearIR_img_normed = np.vstack((black_rows, NearIR_img_normed, black_rows))
            filename = "{}.png".format(scan.timestamp[0])
            cv2.imwrite(os.path.join(depth_path, filename), img_range_scaled)
            cv2.imwrite(os.path.join(NearIR_path, filename), np.uint8(NearIR_img_normed))


        cv2.waitKey(1)