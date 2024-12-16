
import torch

from models.losses import metric_3D_loss, edge_loss

import time
import numpy as np
import cv2
import os
import open3d as o3d

from torch.utils.tensorboard import SummaryWriter
from models.evaluator import DepthEvaluator
import torch.distributions as D


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer:
    def __init__(self, model, optimizer, save_path, scheduler= None, visualize = False, max_range_vis=80.0):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        time.sleep(3)

        self.visualize = visualize
        self.max_range_vis = max_range_vis
        # TensorBoard
        #save_path ='/home/appuser/data/train_depth/{}_{}/'.format(args.model_type, args.encoding)
        self.save_path = save_path
        self.writer = SummaryWriter(save_path)

        # Timer
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        
        # Evaluator
        self.depth_eval = DepthEvaluator()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_one_epoch(self, dataloder, epoch):
        self.model.train()
        
        total_loss = 0.0
        # train one epoch
        for batch_idx, (color_img, range_img, unit_vec, normals, encoding) in enumerate(dataloder): #enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            color_img, range_img, unit_vec, normals, encoding = color_img.to(self.device), range_img.to(self.device), unit_vec.to(self.device), normals.to(self.device), encoding.to(self.device) 
            # run forward path
            start_time = time.time()
            self.start.record()
            #outputs_prob, outputs_semantic = nocs_model(color_img, encoding)
            outputs_mu, outputs_skymask  = self.model(color_img, encoding)
            self.end.record()
            curr_time = (time.time()-start_time)*1000
    
            # Waits for everything to finish running
            torch.cuda.synchronize()
            
            M = (range_img > 0).to(torch.float32)
            loss_skymask = 0.0
            if isinstance(outputs_skymask,type(None)):
                outputs_skymask = M
            loss_skymask = torch.nn.functional.binary_cross_entropy(outputs_skymask,M)
            outputs_skymask = (outputs_skymask > 0.5).to(torch.float32)

            xyz_img = outputs_mu*M
            xyz_img_gt = unit_vec*range_img
            
            loss_pc = metric_3D_loss(xyz_img_gt, xyz_img, M)

            # build depth images
            depth = (outputs_skymask*outputs_mu)[:,2:3,...]
            depth_gt = xyz_img_gt[:,2:3,...]
            

            #normal_img = torch_build_normal_xyz(xyz_img)
            #normal_gt = torch_build_normal_xyz(xyz_img_gt)

            loss = loss_skymask + loss_pc  + edge_loss(xyz_img, xyz_img_gt, M, ord=2, dim=0)+ edge_loss(xyz_img, xyz_img_gt, M, ord=2, dim=1) + edge_loss(xyz_img, xyz_img_gt, M, ord=2, dim=2)# + normal_loss#+ normal_loss_fn(normal_img, normal_gt, M)
    
            if self.visualize:
                # visualize first sample in batch
                rgb_img = color_img.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                range_img  = depth_gt.permute(0, 2, 3, 1)[0,...,0].cpu().detach().numpy()
                outputs_img = depth.permute(0, 2, 3, 1)[0,...,0].cpu().detach().numpy()
                xyz_img = (outputs_skymask*xyz_img).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                xyz_img_gt = xyz_img_gt.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                semantics_pred = (outputs_skymask).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                cv2.imshow("semantics", np.uint8(255*semantics_pred)[::2,::2])
                cv2.imshow("rgb_img", np.uint8(rgb_img)[::2,::2])

                cv2.imshow("gt", cv2.applyColorMap(np.uint8(255*range_img/self.max_range_vis), cv2.COLORMAP_JET)[::2,::2])
                cv2.imshow("pred", cv2.applyColorMap(np.uint8(255*outputs_img/self.max_range_vis), cv2.COLORMAP_JET)[::2,::2])

                if (cv2.waitKey(1) & 0xFF) == ord('q'):

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_img.reshape(-1,3))
                    pcd.colors = o3d.utility.Vector3dVector(np.float32(rgb_img[...,::-1].reshape(-1,3))/255.0)

                    pcdgt = o3d.geometry.PointCloud()
                    pcdgt.points = o3d.utility.Vector3dVector(xyz_img_gt[::4,::4].reshape(-1,3))


                    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
                    o3d.visualization.draw_geometries([mesh, pcd, pcdgt])
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            total_loss += loss.item()
            #total_unc += (det_img.sum()/M.sum()).item()
        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloder)
        self.writer.add_scalar('Loss_EPOCH', avg_loss, epoch)#

    def test_one_epoch(self, dataloder, epoch):
        self.model.eval()
        self.depth_eval.reset()
        for batch_idx, (color_img, range_img, unit_vec, normals, encoding) in enumerate(dataloder): #enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            color_img, range_img, unit_vec, normals, encoding = color_img.to(self.device), range_img.to(self.device), unit_vec.to(self.device), normals.to(self.device), encoding.to(self.device) 
            # run forward path
            start_time = time.time()
            self.start.record()
            #outputs_prob, outputs_semantic = nocs_model(color_img, encoding)
            outputs_mu, outputs_skymask = self.model(color_img, encoding)


            self.end.record()
            curr_time = (time.time()-start_time)*1000
    
            # Waits for everything to finish running
            torch.cuda.synchronize()

            M = (range_img > 0).to(torch.float32)

            if isinstance(outputs_skymask,type(None)):
                outputs_skymask = M
            outputs_skymask = (outputs_skymask > 0.25).to(torch.float32)
            xyz_img = outputs_mu*M
            xyz_img_gt = unit_vec*range_img*M

            # build depth images
            depth = xyz_img[:,2:3,...]
            depth_gt = xyz_img_gt[:,2:3,...]

            outputs_range = torch.linalg.norm(xyz_img, axis=1, keepdims=True)
            self.depth_eval.update(depth, depth_gt, M)

            xyz_img = xyz_img*M
            #normal_img = normal_img*M
            
            
            if self.visualize:
                # visualize first sample in batch
                rgb_img = color_img.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()

                range_img  = range_img.permute(0, 2, 3, 1)[0,...,0].cpu().detach().numpy()
                outputs_img = outputs_range.permute(0, 2, 3, 1)[0,...,0].cpu().detach().numpy()
                xyz_img = (outputs_skymask*xyz_img).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                semantics_pred = (outputs_skymask).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()

                cv2.imshow("semantics", np.uint8(255*semantics_pred)[::4,::4])
                cv2.imshow("rgb_img", np.uint8(rgb_img)[::4,::4])
                # cv2.imshow("normal_img", np.uint8(255*0.5*(normal_img+1))[::4,::4])
                # cv2.imshow("normal_gt", np.uint8(255*0.5*(normals+1))[::4,::4])
                cv2.imshow("gt", cv2.applyColorMap(np.uint8(255*range_img/self.max_range_vis), cv2.COLORMAP_JET)[::4,::4])
                cv2.imshow("pred", cv2.applyColorMap(np.uint8(255*semantics_pred[...,0]*outputs_img/self.max_range_vis), cv2.COLORMAP_JET)[::4,::4])

                if (cv2.waitKey(1) & 0xFF) == ord('q'):

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_img.reshape(-1,3))
                    pcd.colors = o3d.utility.Vector3dVector(np.float32(rgb_img[...,::-1].reshape(-1,3))/255.0)
        
                    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
                    o3d.visualization.draw_geometries([mesh, pcd])

        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = self.depth_eval.compute_final_metrics()
        print("metrics: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3", abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
        self.writer.add_scalar('abs_rel', abs_rel, epoch)
        self.writer.add_scalar('sq_rel', sq_rel, epoch)
        self.writer.add_scalar('rmse', rmse, epoch)
        self.writer.add_scalar('rmse_log', rmse_log, epoch)
        self.writer.add_scalar('a1', a1, epoch)
        self.writer.add_scalar('a2', a2, epoch)
        self.writer.add_scalar('a3', a3, epoch)
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    
    def __call__(self, dataloder_train, dataloder_test, num_epochs=50, test_every_nth_epoch=1, save_every_nth_epoch=-1):
        for epoch in range(num_epochs):
            # train one epoch
            self.train_one_epoch(dataloder_train, epoch)
            # test
            if epoch > 0 and epoch % test_every_nth_epoch == 0:
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = self.test_one_epoch(dataloder_test, epoch)
                # update scheduler based on rmse
                if not isinstance(self.scheduler, type(None)):
                    self.scheduler.step(rmse)
            # save
            if save_every_nth_epoch >= 1 and epoch % save_every_nth_epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_{}.pth".format(str(epoch).zfill(6))))

            
        # run final test
        self.test_one_epoch(dataloder_test, epoch)
        # save last epoch
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_final.pth"))

