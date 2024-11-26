import numpy as np
import cv2
import open3d as o3d

def to_deflection_coordinates(x,y,z):
    # To cylindrical
    p = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(-y, -x)
    # To spherical   
    theta = -np.arctan2(p, z) + np.pi/2
    return phi, theta
    
def spherical_projection(pc, height=64, width=2048, theta_range=None, th=0.1, only_first_return=True):
    '''spherical projection 
    Args:
        pc: point cloud, dim: N*C, first 3 channels should be x, y, z
    Returns:
        pj_img: projected spherical iamges, shape: h*w*C
    '''


    # sort range by distance to sensor     
    r = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
    arr1inds = r.argsort()
    pc = pc[arr1inds[::-1]]
    #pc = pc[arr1inds]
    r = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
    indices = np.where(r > th)
    pc = pc[indices]
    # first 3 indices should be x, y, z     
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        
    phi, theta = to_deflection_coordinates(x,y,z)

    #indices = np.where(r > th)
    if isinstance(theta_range, type(None)):
        theta_min, theta_max = [theta.min(), theta.max()]
    else: 
        theta_min, theta_max = theta_range
        
    #phi_min, phi_max = [phi.min(), phi.max()]
    phi_min, phi_max = [-np.pi, np.pi]
    
    # assuming uniform distribution of rays
    bins_h = np.linspace(theta_min, theta_max, height)[::-1]
    bins_w = np.linspace(phi_min, phi_max, width)[::-1]
    
    theta_img = np.stack(width*[bins_h], axis=-1)
    phi_img = np.stack(height*[bins_w], axis=0)

    idx_h = np.digitize(theta, bins_h)-1
    idx_w = np.digitize(phi, bins_w)-1
    
    pj_img = np.zeros((height, width, pc.shape[1])).astype(np.float32)
    # since some indices are double, only the last assignmets are used. Thats why we sort befor
    pj_img[idx_h, idx_w, :] = pc
   
    return pj_img, (theta_min, theta_max), (phi_min, phi_max) 





def load_ptx(file_path):
    with open(file_path, 'r') as file:
        # Parse header
        num_columns = int(file.readline().strip())  # Number of columns (W)
        num_rows = int(file.readline().strip())  # Number of rows (H)

        # Scanner registered position and axes
        scanner_position = np.array(list(map(float, file.readline().strip().split())))
        scanner_axis_x = np.array(list(map(float, file.readline().strip().split())))
        scanner_axis_y = np.array(list(map(float, file.readline().strip().split())))
        scanner_axis_z = np.array(list(map(float, file.readline().strip().split())))

        # Transformation matrix
        transformation_matrix = np.zeros((4, 4))
        for i in range(4):
            transformation_matrix[i, :] = list(map(float, file.readline().strip().split()))

        # Parse body (point cloud data)
        data = []
        for line in file:
            if line.strip():  # Skip empty lines
                data.append(list(map(float, line.strip().split())))

        data = np.array(data)

        # Validate dimensions
        if data.shape[0] != num_columns * num_rows:
            raise ValueError("Mismatch between header dimensions and number of points in the body.")

        # Reshape into [H, W, 7] (X, Y, Z, intensity, R, G, B)
        data

        return data


# Usage
import glob, os
ptx_files = []
ptx_files += list(glob.glob("/home/appuser/data/JPN_Dataset/Coast/*.*.ptx"))
ptx_files += list(glob.glob("/home/appuser/data/JPN_Dataset/ParkingIn/*.*.ptx"))
ptx_files += list(glob.glob("/home/appuser/data/JPN_Dataset/ParkingOut/*.*.ptx"))
ptx_files += list(glob.glob("/home/appuser/data/JPN_Dataset/Forest/*.*.ptx"))
ptx_files += list(glob.glob("/home/appuser/data/JPN_Dataset/Residential/*.*.ptx"))
ptx_files += list(glob.glob("/home/appuser/data/JPN_Dataset/Urban/*.*.ptx"))
save_path = "/home/appuser/data/JPN_Dataset/refined/"
#os.makedirs(os.path.join(save_path, "intensity"), exist_ok=True)
os.makedirs(os.path.join(save_path, "rgb"), exist_ok=True)
os.makedirs(os.path.join(save_path, "range"), exist_ok=True)
os.makedirs(os.path.join(save_path, "reflectivity"), exist_ok=True)
#os.makedirs(os.path.join(save_path, "ssim_map"), exist_ok=True)
#os.makedirs(os.path.join(save_path, "NearIR"), exist_ok=True)
#os.makedirs(os.path.join(save_path, "pcd"), exist_ok=True)
for scan_path in ptx_files:
    try:
        file_path = scan_path

        base= os.path.basename(file_path)

        save_file = os.path.join(save_path, base)
        points_array = load_ptx(file_path)

        img_rgb = cv2.imread(scan_path.replace(".ptx",".RGB.png"))
        ref_img = cv2.imread(scan_path.replace(".ptx",".Ref.png"), cv2.IMREAD_UNCHANGED)
        
        

        h, w, c = img_rgb.shape
        h_ = w//2

        # Create black rows with the same width and channels as the image
        black_rows = np.zeros((h_ - h, w, 3), dtype=np.uint8)

        # Add the black rows to the bottom of the image
        img_rgb = np.vstack((img_rgb, black_rows))
        ref_img = np.vstack((ref_img, black_rows[...,0]))

        w_ = 2*2048#int(w-0.05*w)
        xyz_img, _, _ = spherical_projection(points_array,height=w_//2, width=w_, theta_range=[-np.pi/2,np.pi/2])
        xyz_img = xyz_img
        #normals = build_normal_xyz(xyz_img)
        h, w, c = xyz_img.shape
        img_rgb_ = cv2.resize(img_rgb, (w, h))
        ref_img_ = cv2.resize(ref_img, (w, h))

        img_range = np.linalg.norm(xyz_img[...,0:3],axis=-1)

        # Define the maximum depth range
        max_depth = 150.0  # Maximum depth in meters

        # Scale the depth image to the range 0-65535
        img_range_scaled = (img_range / max_depth * 65535).clip(0, 65535).astype(np.uint16)
        
        # invert scale
        img_range_original = (img_range_scaled / 65535) * max_depth

        # build NearIR
        nearIR = ref_img_-np.uint8(xyz_img[...,4]) 
    except:
        continue

    # build SSIM 
    #img_gray = cv2.cvtColor(img_rgb_, cv2.COLOR_BGR2GRAY)
    #ssim_map = compute_ssim_per_pixel(img_gray, np.uint8(xyz_img[...,4]))
    #ssim_map = np.uint8(255*(ssim_map+1)/2)

    # cv2.imshow("range_img", cv2.applyColorMap(np.uint8(255*np.minimum(img_range[::4,::4],50)/50), cv2.COLORMAP_MAGMA))
    # cv2.imshow("intensity", np.uint8(xyz_img[...,4:5][::4,::4,:]))
    # cv2.imshow("normals", ((normals+1)/2)[::4,::4,:])
    # cv2.imshow("img_rgb", img_rgb_[::4,::4,:])
    cv2.imwrite(os.path.join(save_path, "rgb", base).replace(".ptx", ".png"), img_rgb_)
    cv2.imwrite(os.path.join(save_path, "reflectivity", base).replace(".ptx", ".png"), ref_img_)
    #cv2.imwrite(os.path.join(save_path, "intensity", base).replace(".ptx", ".png"), np.uint8(xyz_img[...,4:5]))
    cv2.imwrite(os.path.join(save_path, "range", base).replace(".ptx", ".png"), img_range_scaled)
    #cv2.imwrite(os.path.join(save_path, "ssim_map", base).replace(".ptx", ".png"), ssim_map)
    #cv2.imwrite(os.path.join(save_path, "NearIR", base).replace(".ptx", ".png"), nearIR)

    #cv2.waitKey(1)
    
    # pcd_dynamic = o3d.geometry.PointCloud()
    # pcd_dynamic.points = o3d.utility.Vector3dVector(xyz_img[...,0:3].reshape(-1,3))
    # #pcd_dynamic.normals = o3d.utility.Vector3dVector(normals[...,0:3].reshape(-1,3))
    # pcd_dynamic.colors = o3d.utility.Vector3dVector(img_rgb_[...,::-1].reshape(-1,3)/255.0)

    # o3d.io.write_point_cloud(os.path.join(save_path, "pcd", base).replace(".ptx", ".pcd"), pcd_dynamic)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     o3d.visualization.draw_geometries([pcd_dynamic])

