import glob
import argparse
import torch
from torch.utils.data import DataLoader
from dataset.dataloader_CARLA import CarlaEquirectangular
from models.semanticFCN import RangeNetWithFPN
from models.losses import loss_3D_gaussian, metric_3D_loss, metric_3D_loss_RMSE, cosine_similarity_loss, torch_build_normal_xyz, gradient_loss
import torch.optim as optim
import tqdm
import time
import numpy as np
import cv2
import os
import open3d as o3d
from dataset.definitions import color_map, class_names, meta_channel_dict
from torch.utils.tensorboard import SummaryWriter
from models.evaluator import DepthEvaluator


def calculate_intersection_union(outputs, targets, num_classes):
    # Initialize IoU per class
    iou_per_class = torch.zeros(num_classes)
    intersection_per_class = torch.zeros(num_classes)
    union_per_class = torch.zeros(num_classes)

    for cls in range(num_classes):
        # Get predictions and targets for the current class
        pred_cls = (outputs == cls).float()
        target_cls = (targets == cls).float()
        
        # Calculate intersection and union
        intersection_per_class[cls] = (pred_cls * target_cls).sum()
        union_per_class[cls] = (pred_cls + target_cls).sum() - intersection_per_class[cls]
        
    return intersection_per_class, union_per_class

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
    config["FOV"] = list(range(20,70,1))
    config["H"] = [256]#[32*i for i in range(2,32,4)]
    config["PITCH"] = [0] #list(range(0,360,1))
    config["ROLL"] = [0]
    config["YAW"] =[0]
    config["DISTORTION"] = np.linspace(0.01, 1.0, 100).tolist() 
    config["ASPECT"] = [1.0]
    config["FLIP"] = [True, False]
    config["MAX_RANGE"] = 100.0
    config["MIN_RANGE"] = 2.5
    config["SENSOR_ENCODING"] = args.encoding #"UnitVec"
    # DataLoader
    data_path_train = [(bin_path, bin_path.replace("rgb", "range"), bin_path.replace("rgb", "labels"))  for bin_path in glob.glob(f"/home/appuser/mnt/CARLA360/dataset/train/rgb/*.png")]
    data_path_test = [(bin_path, bin_path.replace("rgb", "range"), bin_path.replace("rgb", "labels"))  for bin_path in glob.glob(f"/home/appuser/mnt/CARLA360/dataset/val/rgb/*.png")]
    
    depth_dataset_train = CarlaEquirectangular(data_path_train, config=config)
    dataloader_train = DataLoader(depth_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    depth_dataset_test = CarlaEquirectangular(data_path_test, config=config)
    dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=False, num_workers=4)
    
    # Depth Estimation Network
    nocs_model = RangeNetWithFPN(backbone=args.model_type, meta_channel_dim=meta_channel_dict[config["SENSOR_ENCODING"]])

    num_params = count_parameters(nocs_model)
    print("num_params", count_parameters(nocs_model))
    
    # Define optimizer
    optimizer = optim.Adam(nocs_model.parameters(), lr=args.learning_rate)
    
    depth_eval = DepthEvaluator()
    
    # TensorBoard
    save_path ='/home/appuser/data/train_depth/{}_{}/'.format(args.model_type, args.encoding)
    writer = SummaryWriter(save_path)
    
    # Timer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Training loop
    num_epochs = args.num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nocs_model.to(device)
    time.sleep(3)
    for epoch in range(num_epochs):
        nocs_model.train()
        total_loss = 0.0
        # train one epoch
        for batch_idx, (color_img, range_img, unit_vec, normals, semantics, encoding) in enumerate(dataloader_train): #enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            color_img, range_img, unit_vec, normals, semantics, encoding = color_img.to(device), range_img.to(device), unit_vec.to(device), normals.to(device), semantics.to(device), encoding.to(device) 
            # run forward path
            start_time = time.time()
            start.record()
            #outputs_prob, outputs_semantic = nocs_model(color_img, encoding)
            outputs_prob = nocs_model(color_img, encoding)
            mu = outputs_prob
            #outputs_range =torch.nn.functional.relu(outputs_range)
            end.record()
            curr_time = (time.time()-start_time)*1000
    
            # Waits for everything to finish running
            torch.cuda.synchronize()
            


            # get losses
            M = (range_img > 0).to(torch.float32)
            
            xyz_img = unit_vec*mu#unit_vec*mu
            xyz_img_gt = unit_vec*range_img
            outputs_range = (torch.linalg.norm(xyz_img, axis=1, keepdims=True))

            loss_RMSE = metric_3D_loss_RMSE(range_img, outputs_range, M)
            loss_pc = metric_3D_loss_RMSE(xyz_img_gt, xyz_img, M)

            normal_img = torch_build_normal_xyz(xyz_img)
            normal_gt = torch_build_normal_xyz(xyz_img_gt)
            loss_normals = cosine_similarity_loss(normal_gt, normal_img, M)
            #loss_SSIL = gradient_loss(mu,range_img, M)
            loss = loss_pc+loss_normals+loss_RMSE
            

            xyz_img = xyz_img*M
            normal_img = normal_img*M
            
            print("inference took time: cpu: {} ms., cuda: {} ms. loss: {}".format(curr_time,start.elapsed_time(end), loss_RMSE.item()))
            

            
            if True:#args.visualization:
                # visualize first sample in batch
                rgb_img = color_img.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                normals  = normal_gt.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                normal_img  = normal_img.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                range_img  = range_img.permute(0, 2, 3, 1)[0,...,0].cpu().detach().numpy()
                outputs_img = outputs_range.permute(0, 2, 3, 1)[0,...,0].cpu().detach().numpy()
                xyz_img = xyz_img.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                semantics_pred = (M).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                #semantics_pred = (semseg_img).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                cv2.imshow("semantics", np.uint8(255*semantics_pred))
                cv2.imshow("rgb_img", np.uint8(rgb_img))
                cv2.imshow("normal_img", np.uint8(255*0.5*(normal_img+1)))
                cv2.imshow("normal_gt", np.uint8(255*0.5*(normals+1)))
                cv2.imshow("gt", cv2.applyColorMap(np.uint8(255*range_img/config["MAX_RANGE"]), cv2.COLORMAP_JET))
                cv2.imshow("pred", cv2.applyColorMap(np.uint8(255*semantics_pred[...,0]*outputs_img/config["MAX_RANGE"]), cv2.COLORMAP_JET))

                #cv2.waitKey(0)
                #cv2.imshow("inf", np.vstack((reflectivity_img,np.uint8(255*normal_img),prev_sem_pred,prev_sem_gt)))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    # grid_size = 50
                    # step_size = 1
         
                    # # Create vertices for the grid lines
                    # lines = []
                    # for i in range(-grid_size, grid_size + step_size, step_size):
                    #     lines.append([[i, -grid_size, 0], [i, grid_size, 0]])
                    #     lines.append([[-grid_size, i, 0], [grid_size, i, 0]])
        
                    # # Create an Open3D LineSet
                    # line_set = o3d.geometry.LineSet()
                    # line_set.points = o3d.utility.Vector3dVector(np.array(lines).reshape(-1, 3))
                    # line_set.lines = o3d.utility.Vector2iVector(np.arange(len(lines) * 2).reshape(-1, 2))
                    # line_set.translate((0,0,-1.7))
                    #time.sleep(10)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_img.reshape(-1,3))
                    pcd.colors = o3d.utility.Vector3dVector(np.float32(rgb_img[...,::-1].reshape(-1,3))/255.0)
        
                    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
                    o3d.visualization.draw_geometries([mesh, pcd])
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            
        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader_train)
        writer.add_scalar('Loss_EPOCH', avg_loss, epoch)
        print(f"Train Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")
        nocs_model.eval()
        depth_eval.reset()
        # test one epoch
        for batch_idx, (color_img, range_img, unit_vec, normals, semantics, encoding) in enumerate(dataloader_test): #enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            color_img, range_img, unit_vec, normals, semantics, encoding = color_img.to(device), range_img.to(device), unit_vec.to(device), normals.to(device), semantics.to(device), encoding.to(device) 
            # run forward path
            start_time = time.time()
            start.record()
            #outputs_prob, outputs_semantic = nocs_model(color_img, encoding)
            outputs_prob = nocs_model(color_img, encoding)
            mu = outputs_prob
            #outputs_range =torch.nn.functional.relu(outputs_range)
            end.record()
            curr_time = (time.time()-start_time)*1000
    
            # Waits for everything to finish running
            torch.cuda.synchronize()

            M = (range_img > 0).to(torch.float32)

            xyz_img = unit_vec*mu#unit_vec*mu
            xyz_img_gt = unit_vec*range_img

            outputs_range = (torch.linalg.norm(xyz_img, axis=1, keepdims=True))
            depth_eval.update(range_img, outputs_range, M)

            loss_RMSE = metric_3D_loss_RMSE(range_img, outputs_range, M)
            # loss_pc = metric_3D_loss(xyz_img_gt, xyz_img, M)
            loss_range = metric_3D_loss_RMSE(range_img, outputs_range, M)

            normal_img = torch_build_normal_xyz(xyz_img)
            normal_gt = torch_build_normal_xyz(xyz_img_gt)
            loss_normals = metric_3D_loss(normal_gt, normal_img, M)
            loss_SSIL = gradient_loss(outputs_range,range_img, M)
            loss = loss_range+0.5*loss_normals+0.25*loss_SSIL
            

            xyz_img = xyz_img*M#*semseg_img#[:,None,...]
            normal_img = normal_img*M#*semseg_img
            
            print("inference took time: cpu: {} ms., cuda: {} ms. loss: {}".format(curr_time,start.elapsed_time(end), loss_RMSE.item()))
            
            if True:#args.visualization:
                # visualize first sample in batch
                rgb_img = color_img.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                normals  = normal_gt.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                normal_img  = normal_img.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                range_img  = range_img.permute(0, 2, 3, 1)[0,...,0].cpu().detach().numpy()
                outputs_img = outputs_range.permute(0, 2, 3, 1)[0,...,0].cpu().detach().numpy()
                xyz_img = xyz_img.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                semantics_pred = (M).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                #semantics_pred = (semseg_img).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                cv2.imshow("semantics", np.uint8(255*semantics_pred))
                cv2.imshow("rgb_img", np.uint8(rgb_img))
                cv2.imshow("normal_img", np.uint8(255*0.5*(normal_img+1)))
                cv2.imshow("normal_gt", np.uint8(255*0.5*(normals+1)))
                cv2.imshow("gt", cv2.applyColorMap(np.uint8(255*range_img/config["MAX_RANGE"]), cv2.COLORMAP_JET))
                cv2.imshow("pred", cv2.applyColorMap(np.uint8(255*semantics_pred[...,0]*outputs_img/config["MAX_RANGE"]), cv2.COLORMAP_JET))

                #cv2.waitKey(0)
                #cv2.imshow("inf", np.vstack((reflectivity_img,np.uint8(255*normal_img),prev_sem_pred,prev_sem_gt)))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    # grid_size = 50
                    # step_size = 1
         
                    # # Create vertices for the grid lines
                    # lines = []
                    # for i in range(-grid_size, grid_size + step_size, step_size):
                    #     lines.append([[i, -grid_size, 0], [i, grid_size, 0]])
                    #     lines.append([[-grid_size, i, 0], [grid_size, i, 0]])
        
                    # # Create an Open3D LineSet
                    # line_set = o3d.geometry.LineSet()
                    # line_set.points = o3d.utility.Vector3dVector(np.array(lines).reshape(-1, 3))
                    # line_set.lines = o3d.utility.Vector2iVector(np.arange(len(lines) * 2).reshape(-1, 2))
                    # line_set.translate((0,0,-1.7))
                    #time.sleep(10)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_img.reshape(-1,3))
                    pcd.colors = o3d.utility.Vector3dVector(np.float32(rgb_img[...,::-1].reshape(-1,3))/255.0)
        
                    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
                    o3d.visualization.draw_geometries([mesh, pcd])
                

            

            
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = depth_eval.compute_final_metrics()
        print("metrics: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3", abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
        writer.add_scalar('abs_rel', abs_rel, epoch)
        writer.add_scalar('sq_rel', sq_rel, epoch)
        writer.add_scalar('rmse', rmse, epoch)
        writer.add_scalar('rmse_log', rmse_log, epoch)
        writer.add_scalar('a1', a1, epoch)
        writer.add_scalar('a2', a2, epoch)
        writer.add_scalar('a3', a3, epoch)
        print("D")


    # # Save the trained model if needed
    torch.save(nocs_model.state_dict(), os.path.join(save_path, "model_final.pth"))
    
    # Close the TensorBoard writer
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train script for SemanticKitti')
    parser.add_argument('--model_type', type=str, default='resnet18',
                        help='Type of the model to be used (default: resnet50)')
    parser.add_argument('--encoding', type=str, default="Deflection",
                        help='Type of the model to be used (default: CameraTensor)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the model (default: 0.001)')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of epochs for training (default: 50)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers (default: 1)')
    parser.add_argument('--rotate', action='store_true',
                        help='Whether to apply rotation augmentation (default: False)')
    parser.add_argument('--flip', action='store_true',
                        help='Whether to apply flip augmentation (default: False)')
    parser.add_argument('--visualization', action='store_true',
                        help='Toggle visualization during training (default: False)')
    args = parser.parse_args()

    main(args)

