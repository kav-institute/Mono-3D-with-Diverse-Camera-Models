import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()

    def forward(self, predicted_features, target_features, mask):
        
        predicted_features_sm = F.softmax(predicted_features, dim=1)
        target_features_sm = F.softmax(target_features, dim=1)

        # Cross Entropy
        loss = target_features_sm * torch.log(predicted_features_sm) * mask

        # KL loss
        #loss += target_features_sm * torch.log(target_features_sm) * mask

        denom = torch.sum(mask)
        # Calculate the total loss
        loss = torch.sum(-loss)/denom
        return loss
    

class BrierLoss(nn.Module):
    def __init__(self):
        super(BrierLoss, self).__init__()

    def forward(self, predicted_features, target_features, mask):
        
        predicted_features_sm = F.softmax(predicted_features, dim=1)
        target_features_sm = F.softmax(target_features, dim=1)

        # Cross Entropy
        loss = torch.square(target_features_sm-predicted_features_sm)

        loss = torch.sum(loss, dim=1) * mask

        denom = torch.sum(mask)
        # Calculate the total loss
        loss = torch.sum(loss)/denom
        return loss

class SemanticSegmentationLoss(nn.Module):
    def __init__(self):
        super(SemanticSegmentationLoss, self).__init__()

        # Assuming three classes
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predicted_logits, target, num_classes=20):
        # Flatten the predictions and the target
        predicted_logits_flat = predicted_logits.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target_flat = target.view(-1)

        # Calculate the Cross-Entropy Loss
        loss = self.criterion(predicted_logits_flat, target_flat)

        return loss

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.9, beta=0.1, num_classes=20):
        targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).transpose(1, 4).squeeze(-1)   
        inputs = F.softmax(inputs, dim=1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky

class DepthEstimationLoss(nn.Module):
    def __init__(self, alpha=0.85, beta=0.15, max_depth=None):
        super(DepthEstimationLoss, self).__init__()
        self.alpha = alpha  # Weight for L2 loss
        self.beta = beta    # Weight for L1 loss
        self.max_depth = max_depth  # Optional, to clip depth values if needed

    def forward(self, predicted_depth, ground_truth_depth, mask):
        # Optionally clip depth values to avoid large gradients
        if self.max_depth is not None:
            predicted_depth = torch.clamp(predicted_depth, 0, self.max_depth)
            ground_truth_depth = torch.clamp(ground_truth_depth, 0, self.max_depth)

        denom = torch.sum(mask)

        # L2 loss (Mean Squared Error)
        l2_loss = torch.sum(mask*(predicted_depth - ground_truth_depth) ** 2)/denom
        
        # L1 loss (Mean Absolute Error)
        l1_loss = torch.sum(mask*torch.abs(predicted_depth - ground_truth_depth))

        # Combined loss with weighted terms
        loss = self.alpha * l2_loss + self.beta * l1_loss
        
        return loss

class ScaleInvariantDepthLoss(nn.Module):
    def __init__(self):
        super(ScaleInvariantDepthLoss, self).__init__()

    def forward(self, predicted_depth, ground_truth_depth, mask):
        # Ensure both depths are non-zero to avoid division by zero
        epsilon = 1e-6
        pred_mean = torch.mean(predicted_depth)
        gt_mean = torch.mean(ground_truth_depth)

        denom = torch.sum(mask)

        # Normalize predicted depth by scaling it to ground truth mean
        scaled_pred = predicted_depth * (gt_mean / (pred_mean + epsilon))

        # Compute L2 loss between scaled predicted depth and ground truth
        loss = torch.sum(mask* ((scaled_pred - ground_truth_depth) ** 2))/denom

        return loss
    
class NormalLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(NormalLoss, self).__init__()
        self.weight = weight  # Weight to scale the normal loss

    def forward(self, predicted_normals, ground_truth_normals, mask):

        denom = torch.sum(mask)
        # Compute the cosine similarity between predicted and ground truth normals
        normal_loss = torch.sum(mask*(1 - torch.cosine_similarity(predicted_normals, ground_truth_normals, dim=1)))/denom

        # Scale the normal loss by the specified weight
        return self.weight * normal_loss


def metric_3D_loss(pc_img_gt, pc_img_pred, mask):
    
    #loss = torch.square(mask*pc_img_gt-mask*pc_img_pred)
    loss = torch.nn.functional.huber_loss(mask*pc_img_pred, mask*pc_img_gt, reduction='sum', delta=1.0)
    #loss = torch.nn.functional.mse_loss(mask*pc_img_pred, mask*pc_img_gt, size_average=None, reduce=None, reduction='sum') 
    denom = torch.sum(mask)
    # Calculate the total variation loss
    loss = torch.sum(loss)/denom
    
    
    return loss

def weighted_metric_3D_loss(pc_img_gt, pc_img_pred, weights, mask):
    
    #loss = torch.square(mask*pc_img_gt-mask*pc_img_pred)
    loss = torch.nn.functional.huber_loss(weights*mask*pc_img_pred, weights*mask*pc_img_gt, reduction='sum', delta=1.0)
    #loss = torch.nn.functional.mse_loss(mask*pc_img_pred, mask*pc_img_gt, size_average=None, reduce=None, reduction='sum') 
    denom = torch.sum(mask)
    # Calculate the total variation loss
    loss = torch.sum(loss)/denom
    
    
    return loss

def metric_3D_loss_RMSE(pc_img_gt, pc_img_pred, mask):
    
    #loss = torch.square(mask*pc_img_gt-mask*pc_img_pred)
    #loss = torch.nn.functional.huber_loss(mask*pc_img_pred, mask*pc_img_gt, reduction='sum', delta=1.0)
    loss = torch.nn.functional.mse_loss(mask*pc_img_pred, mask*pc_img_gt, size_average=None, reduce=None, reduction='sum') 
    denom = torch.sum(mask)
    # Calculate the total variation loss
    loss = torch.sum(loss)/denom
    # build sqrt
    loss = loss

    return loss

def difference_of_normals(normals_gt, normals_pred, mask):
    loss = torch.linalg.norm(mask*torch.square(normals_gt - normals_pred)/2, dim = 1)
    denom = torch.sum(mask)
    # Calculate the total variation loss
    loss = torch.sum(loss)/denom
    return loss

# def torch_build_normal_xyz(xyz):
#     '''
#     @param xyz: ndarray with shape (h,w,3) containing a staggered point cloud
#     '''
#     scharr_x_weight = torch.tensor([[-3, 0, 3],
#                                     [-10, 0, 10],
#                                     [-3, 0, 3]], dtype=torch.float32, device=xyz.device, requires_grad=False).unsqueeze(0).unsqueeze(0)

#     scharr_y_weight = torch.tensor([[3, 10, 3],
#                                     [0, 0, 0],
#                                     [-3, -10, -3]], dtype=torch.float32, device=xyz.device, requires_grad=False).unsqueeze(0).unsqueeze(0)

#     x = xyz[:,0:1,...]
#     y = xyz[:,1:2,...]
#     z = xyz[:,2:3,...]

#     Sxx = F.conv2d(x, weight=scharr_x_weight, padding=1)
#     Sxy = F.conv2d(x, weight=scharr_y_weight, padding=1)

#     Syx = F.conv2d(y, weight=scharr_x_weight, padding=1)
#     Syy = F.conv2d(y, weight=scharr_y_weight, padding=1)

#     Szx = F.conv2d(z, weight=scharr_x_weight, padding=1)
#     Szy = F.conv2d(z, weight=scharr_y_weight, padding=1)

#     # Build cross product
#     normal = torch.cat((Syx*Szy - Szx*Syy,
#                         Szx*Sxy - Szy*Sxx,
#                         Sxx*Syy - Syx*Sxy), dim=1)

#     # Normalize the normal vectors
#     n = torch.norm(normal, dim=1, keepdim=True) + torch.tensor(1e-10, dtype=torch.float32)
#     normal = normal / n

#     #normal = normal * torch.sgn(z)
#     return normal

def build_scharr_filter(size):
    """ 
    Build a generalized square Scharr filter for both x and y directions of size (size x size).
    
    Args:
        size (int): The size of the filter. Must be an odd number.
    
    Returns:
        scharr_x_weight, scharr_y_weight (tensors): The generalized Scharr filter for x and y gradients.
    """
    assert size % 2 == 1, "Filter size must be an odd number."
    
    # Create an index grid to generate the filter based on distance from the center
    center = size // 2
    
    # Generate Scharr-like filter based on the distance from the center
    scharr_x_weight = torch.zeros((size, size), dtype=torch.float32)
    scharr_y_weight = torch.zeros((size, size), dtype=torch.float32)
    
    for i in range(size):
        for j in range(size):
            dx = j - center
            dy = i - center
            # Following Scharr's principle: more weight in the middle of each row/column
            scharr_x_weight[i, j] = dx * (3 if abs(dx) == 1 else 10 if dx == 0 else 1)
            scharr_y_weight[i, j] = dy * (3 if abs(dy) == 1 else 10 if dy == 0 else 1)
    
    return scharr_x_weight.unsqueeze(0).unsqueeze(0), scharr_y_weight.unsqueeze(0).unsqueeze(0)


def torch_build_normal_xyz(xyz, filter_size=3):
    '''
    @param xyz: tensor of shape (batch_size, 3, h, w) containing a staggered point cloud
    @param filter_size: The size of the filter for computing normals (must be an odd number).
    '''

    # Build Scharr filters dynamically based on the filter_size
    scharr_x_weight, scharr_y_weight = build_scharr_filter(filter_size)
    scharr_x_weight = scharr_x_weight.to(xyz.device)
    scharr_y_weight = scharr_y_weight.to(xyz.device)

    # Separate the x, y, z components of the input point cloud
    x = xyz[:, 0:1, ...]  # x component
    y = xyz[:, 1:2, ...]  # y component
    z = xyz[:, 2:3, ...]  # z component

    # Compute the gradients in both x and y directions for each component
    Sxx = F.conv2d(x, weight=scharr_x_weight, padding=filter_size // 2)
    Sxy = F.conv2d(x, weight=scharr_y_weight, padding=filter_size // 2)

    Syx = F.conv2d(y, weight=scharr_x_weight, padding=filter_size // 2)
    Syy = F.conv2d(y, weight=scharr_y_weight, padding=filter_size // 2)

    Szx = F.conv2d(z, weight=scharr_x_weight, padding=filter_size // 2)
    Szy = F.conv2d(z, weight=scharr_y_weight, padding=filter_size // 2)

    # Build the cross product to calculate surface normals
    normal = torch.cat((Syx * Szy - Szx * Syy,  # Normal component in the x-direction
                        Szx * Sxy - Szy * Sxx,  # Normal component in the y-direction
                        Sxx * Syy - Syx * Sxy), dim=1)  # Normal component in the z-direction

    # Normalize the normal vectors
    n = torch.norm(normal, dim=1, keepdim=True) + torch.tensor(1e-10, dtype=torch.float32, device=xyz.device)
    normal = normal / n

    return normal

def edge_loss(xyz, xyz_gt, mask, ord=1, dim=0, filter_size=3):
    '''
    @param xyz: tensor of shape (batch_size, 3, h, w) containing a staggered point cloud
    @param filter_size: The size of the filter for computing normals (must be an odd number).
    '''

    # Build Scharr filters dynamically based on the filter_size
    scharr_x_weight, scharr_y_weight = build_scharr_filter(filter_size)
    scharr_x_weight = scharr_x_weight.to(xyz.device)
    scharr_y_weight = scharr_y_weight.to(xyz.device)

    # Separate the x, y, z components of the input point cloud
    x = xyz[:, dim:dim+1, ...]  # x component
    x_gt = xyz_gt[:, dim:dim+1, ...]  # x component

    # Compute the gradients in both x and y directions for each component
    Sx = F.conv2d(x, weight=scharr_x_weight, padding=filter_size // 2)
    Sy = F.conv2d(x, weight=scharr_y_weight, padding=filter_size // 2)

    Sx_gt = F.conv2d(x_gt, weight=scharr_x_weight, padding=filter_size // 2)
    Sy_gt = F.conv2d(x_gt, weight=scharr_y_weight, padding=filter_size // 2)



    
    grad = torch.cat((Sx,Sy), dim=1)  

    grad_gt = torch.cat((Sx_gt,Sy_gt), dim=1)  

    loss = torch.linalg.vector_norm((grad-grad_gt), dim=1, ord=ord)

    denom = torch.sum(mask)
    # Calculate the total variation loss
    loss = torch.sum(loss)/denom

    return loss

class PairWiseNormalLoss(nn.Module):
    def __init__(self, size_small=3, size_big=9):
        super(PairWiseNormalLoss, self).__init__()
        self.size_small, self.size_big = size_small, size_big
        self.scharr_x_weight_small, self.scharr_y_weight_small = self.build_scharr_filter(size_small)
        self.scharr_x_weight_big, self.scharr_y_weight_big = self.build_scharr_filter(size_big)

    def build_scharr_filter(self, size):
        """ 
        Build a generalized square Scharr filter for both x and y directions of size (size x size).
        
        Args:
            size (int): The size of the filter. Must be an odd number.
        
        Returns:
            scharr_x_weight, scharr_y_weight (tensors): The generalized Scharr filter for x and y gradients.
        """
        assert size % 2 == 1, "Filter size must be an odd number."
        
        # Create an index grid to generate the filter based on distance from the center
        center = size // 2
        
        # Generate Scharr-like filter based on the distance from the center
        scharr_x_weight = torch.zeros((size, size), dtype=torch.float32)
        scharr_y_weight = torch.zeros((size, size), dtype=torch.float32)
        
        for i in range(size):
            for j in range(size):
                dx = j - center
                dy = i - center
                # Following Scharr's principle: more weight in the middle of each row/column
                scharr_x_weight[i, j] = dx * (3 if abs(dx) == 1 else 10 if dx == 0 else 1)
                scharr_y_weight[i, j] = dy * (3 if abs(dy) == 1 else 10 if dy == 0 else 1)
        
        return scharr_x_weight.unsqueeze(0).unsqueeze(0), scharr_y_weight.unsqueeze(0).unsqueeze(0)

    
    def build_normals(self, xyz, scharr_x_weight, scharr_y_weight, filter_size):
        '''
        @param xyz: tensor of shape (batch_size, 3, h, w) containing a staggered point cloud
        @param filter_size: The size of the filter for computing normals (must be an odd number).
        '''

        # Build Scharr filters dynamically based on the filter_size
        scharr_x_weight, scharr_y_weight = build_scharr_filter(filter_size)
        scharr_x_weight = scharr_x_weight.to(xyz.device)
        scharr_y_weight = scharr_y_weight.to(xyz.device)

        # Separate the x, y, z components of the input point cloud
        x = xyz[:, 0:1, ...]  # x component
        y = xyz[:, 1:2, ...]  # y component
        z = xyz[:, 2:3, ...]  # z component

        # Compute the gradients in both x and y directions for each component
        Sxx = F.conv2d(x, weight=scharr_x_weight, padding=filter_size // 2)
        Sxy = F.conv2d(x, weight=scharr_y_weight, padding=filter_size // 2)

        Syx = F.conv2d(y, weight=scharr_x_weight, padding=filter_size // 2)
        Syy = F.conv2d(y, weight=scharr_y_weight, padding=filter_size // 2)

        Szx = F.conv2d(z, weight=scharr_x_weight, padding=filter_size // 2)
        Szy = F.conv2d(z, weight=scharr_y_weight, padding=filter_size // 2)

        # Build the cross product to calculate surface normals
        normal = torch.cat((Syx * Szy - Szx * Syy,  # Normal component in the x-direction
                            Szx * Sxy - Szy * Sxx,  # Normal component in the y-direction
                            Sxx * Syy - Syx * Sxy), dim=1)  # Normal component in the z-direction

        # Normalize the normal vectors
        n = torch.norm(normal, dim=1, keepdim=True) + torch.tensor(1e-10, dtype=torch.float32, device=xyz.device)
        normal = normal / n

        return normal

    def forward(self, xyz_pred, xyz_gt, mask):
        normals_pred_small = self.build_normals(xyz_pred, self.scharr_x_weight_small, self.scharr_y_weight_small, self.size_small)
        normals_gt_small = self.build_normals(xyz_gt, self.scharr_x_weight_small, self.scharr_y_weight_small, self.size_small)

        normals_pred_big = self.build_normals(xyz_pred, self.scharr_x_weight_big, self.scharr_y_weight_big, self.size_big)
        normals_gt_big = self.build_normals(xyz_gt, self.scharr_x_weight_big, self.scharr_y_weight_big, self.size_big)

        # build pairwise normal
        pairwise_normal_big = normals_gt_big*normals_pred_big
        pairwise_normal_small = normals_gt_small*normals_pred_small
        # L1 loss
        loss = torch.sum(mask*torch.abs(pairwise_normal_big-pairwise_normal_small))

        denom = torch.sum(mask)
        # Calculate the total variation loss
        loss = torch.sum(loss)/denom
        return loss

class NormalLoss(nn.Module):
    def __init__(self, size=3):
        super(NormalLoss, self).__init__()
        self.size=size
        self.scharr_x_weight, self.scharr_y_weight = self.build_scharr_filter(size)


    def build_scharr_filter(self, size):
        """ 
        Build a generalized square Scharr filter for both x and y directions of size (size x size).
        
        Args:
            size (int): The size of the filter. Must be an odd number.
        
        Returns:
            scharr_x_weight, scharr_y_weight (tensors): The generalized Scharr filter for x and y gradients.
        """
        assert size % 2 == 1, "Filter size must be an odd number."
        
        # Create an index grid to generate the filter based on distance from the center
        center = size // 2
        
        # Generate Scharr-like filter based on the distance from the center
        scharr_x_weight = torch.zeros((size, size), dtype=torch.float32)
        scharr_y_weight = torch.zeros((size, size), dtype=torch.float32)
        
        for i in range(size):
            for j in range(size):
                dx = j - center
                dy = i - center
                # Following Scharr's principle: more weight in the middle of each row/column
                scharr_x_weight[i, j] = dx * (3 if abs(dx) == 1 else 10 if dx == 0 else 1)
                scharr_y_weight[i, j] = dy * (3 if abs(dy) == 1 else 10 if dy == 0 else 1)
        
        return scharr_x_weight.unsqueeze(0).unsqueeze(0), scharr_y_weight.unsqueeze(0).unsqueeze(0)

    
    def build_normals(self, xyz, scharr_x_weight, scharr_y_weight, filter_size):
        '''
        @param xyz: tensor of shape (batch_size, 3, h, w) containing a staggered point cloud
        @param filter_size: The size of the filter for computing normals (must be an odd number).
        '''

        # Build Scharr filters dynamically based on the filter_size
        scharr_x_weight, scharr_y_weight = build_scharr_filter(filter_size)
        scharr_x_weight = scharr_x_weight.to(xyz.device)
        scharr_y_weight = scharr_y_weight.to(xyz.device)

        # Separate the x, y, z components of the input point cloud
        x = xyz[:, 0:1, ...]  # x component
        y = xyz[:, 1:2, ...]  # y component
        z = xyz[:, 2:3, ...]  # z component

        # Compute the gradients in both x and y directions for each component
        Sxx = F.conv2d(x, weight=scharr_x_weight, padding=filter_size // 2)
        Sxy = F.conv2d(x, weight=scharr_y_weight, padding=filter_size // 2)

        Syx = F.conv2d(y, weight=scharr_x_weight, padding=filter_size // 2)
        Syy = F.conv2d(y, weight=scharr_y_weight, padding=filter_size // 2)

        Szx = F.conv2d(z, weight=scharr_x_weight, padding=filter_size // 2)
        Szy = F.conv2d(z, weight=scharr_y_weight, padding=filter_size // 2)

        # Build the cross product to calculate surface normals
        normal = torch.cat((Syx * Szy - Szx * Syy,  # Normal component in the x-direction
                            Szx * Sxy - Szy * Sxx,  # Normal component in the y-direction
                            Sxx * Syy - Syx * Sxy), dim=1)  # Normal component in the z-direction

        # Normalize the normal vectors
        n = torch.norm(normal, dim=1, keepdim=True) + torch.tensor(1e-10, dtype=torch.float32, device=xyz.device)
        normal = normal / n

        return normal

    def forward(self, xyz_pred, xyz_gt, mask):
        normals_pred = self.build_normals(xyz_pred, self.scharr_x_weight, self.scharr_y_weight, self.size)
        normals_gt = self.build_normals(xyz_gt, self.scharr_x_weight, self.scharr_y_weight, self.size)

        # L1 loss
        loss = torch.sum(mask*torch.abs(normals_pred-normals_gt))

        denom = torch.sum(mask)
        # Calculate the total variation loss
        loss = torch.sum(loss)/denom
        return loss
    
class SilogLoss(nn.Module):
    """
    Compute SILog loss. See https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf for
    more information about scale-invariant loss.
    """
    def __init__(self, variance_focus=0.5, loss_weight=1, data_type=['stereo', 'lidar'], **kwargs):
        super(SilogLoss, self).__init__()
        self.variance_focus = variance_focus
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
    
    def silog_loss(self, prediction, target, mask):
        denom = torch.sum(mask)
        d = mask * (torch.log(mask * prediction + self.eps) - torch.log(mask * target + self.eps))
        d_square_mean = torch.sum(d ** 2) / (denom + self.eps)
        d_mean = torch.sum(d) / (denom + self.eps)
        loss = d_square_mean - self.variance_focus * (d_mean ** 2)

        #denom = torch.sum(mask)
        # Calculate the total variation loss
        #loss = torch.sum(loss)/denom

        return loss

    def forward(self, prediction, target, mask=None, **kwargs):
        
        loss = self.silog_loss(prediction, target, mask)

        return loss * self.loss_weight
    
class SkyLoss(nn.Module):
    def __init__(self, variance_focus=0.5, loss_weight=1, data_type=['stereo', 'lidar'], **kwargs):
        super(SkyLoss, self).__init__()
        self.variance_focus = variance_focus
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
    


    def forward(self, prediction, mask=None, **kwargs):
        
        sky_gt = torch.zeros_like(prediction)
        mask = (~mask.to(torch.bool)).to(torch.float32)
        loss = torch.nn.functional.mse_loss(mask*prediction, mask*sky_gt, size_average=None, reduce=None, reduction='sum') 
        denom = torch.sum(mask)
        # Calculate the total variation loss
        loss = torch.sum(loss)/denom


        return loss * self.loss_weight

import numpy as np

class HDSNLoss(nn.Module):
    """
    Hieratical depth spatial normalization loss.
    loss = MAE((d-median(d)/s - (d'-median(d'))/s'), s = mean(d- median(d))
    """
    def __init__(self, loss_weight=1.0, grid=3, data_type=['sfm', 'stereo', 'lidar'], **kwargs):
        super(HDSNLoss, self).__init__()
        self.loss_weight = loss_weight
        self.grid = grid
        self.data_type = data_type

    def get_hierachy_masks(self, batch, image_size, mask):
        mask = mask.to(torch.bool)
        height, width = image_size
        anchor_power = [(1 / 2) ** (i) for i in range(self.grid)]
        anchor_power.reverse()

        map_grid_list = []
        for anchor in anchor_power:  # e.g. 1/8
            for h in range(int(1 / anchor)):
                for w in range(int(1 / anchor)):
                    mask_new = torch.zeros((batch,  1, height, width), dtype=torch.bool).cuda()
                    mask_new[:, :, int(h * anchor * height):int((h + 1) * anchor * height),
                        int(w * anchor * width):int((w + 1) * anchor * width)] = True
                    mask_new = mask & mask_new
                    map_grid_list.append(mask_new)
        batch_map_grid=torch.stack(map_grid_list,dim=0) # [N, B, 1, H, W]

        return batch_map_grid
    
    def ssi_mae(self, prediction, target, mask_valid):
        B, C, H, W = target.shape
        prediction_nan = prediction.clone()
        target_nan = target.clone()
        prediction_nan[~mask_valid] = float('nan')
        target_nan[~mask_valid] = float('nan')

        valid_pixs = mask_valid.reshape((B, C,-1)).sum(dim=2, keepdims=True) + 1e-10
        valid_pixs = valid_pixs[:, :, :, None]

        gt_median = target_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        gt_median[torch.isnan(gt_median)] = 0
        gt_diff = (torch.abs(target - gt_median) * mask_valid).reshape((B, C, -1))
        gt_s = gt_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        gt_trans = (target - gt_median) / (gt_s + 1e-8)

        pred_median = prediction_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        pred_median[torch.isnan(pred_median)] = 0
        pred_diff = (torch.abs(prediction - pred_median) * mask_valid).reshape((B, C, -1))
        pred_s = pred_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        pred_trans = (prediction - pred_median) / (pred_s + 1e-8)

        loss = torch.sum(torch.abs(gt_trans - pred_trans)*mask_valid) / (torch.sum(mask_valid) + 1e-8)
        return pred_trans, gt_trans, loss

    def forward(self, prediction, target, mask=None, **kwargs):
        """
        Calculate loss.
        """
        B, C, H, W = target.shape
        hierachy_masks = self.get_hierachy_masks(B, (H, W), mask) # [N, B, 1, H, W]
        hierachy_masks_shape = hierachy_masks.reshape(-1, C, H, W)    
        prediction_hie = prediction.unsqueeze(0).repeat(hierachy_masks.shape[0], 1, 1, 1, 1).reshape(-1, C, H, W)     

        target_hie = target.unsqueeze(0).repeat(hierachy_masks.shape[0], 1, 1, 1, 1).reshape(-1, C, H, W)

        #_, _, loss = self.ssi_mae(prediction, target, mask)
        _, _, loss = self.ssi_mae(prediction_hie, target_hie, hierachy_masks_shape)
        return loss * self.loss_weight

def gradient_loss(prediction, target, mask):

    
    divisor = torch.sum(mask)

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    return torch.sum(torch.sum(grad_x, (2, 3)) + torch.sum(grad_y, (2, 3))) / divisor


def build_covariance_matrix(std):
        # Function to construct the covariance matrix from std and corr
    B, C, H, W = std.size()
    covariance_matrix = torch.zeros(B, H, W, C, C)
    covariance_matrix = covariance_matrix.to(std.device)
    for i in range(C):
        for j in range(C):
            if i == j:
                covariance_matrix[:, :, :,  i, j] = 1

                
    return covariance_matrix

def rotate_covariance_matrix(covariance_matrix, unit_vectors, M):
    """
    Rotate the covariance matrix using unit vectors.
    
    Parameters:
    covariance_matrix : torch.Tensor
        Covariance matrix with shape (B, H, W, C, C)
    unit_vectors : torch.Tensor
        Unit vectors with shape (B, 3, H, W)
    
    Returns:
    torch.Tensor
        Rotated covariance matrix with shape (B, H, W, C, C)
    """
    B, H, W, C, _ = covariance_matrix.size()
    
    # Ensure unit vectors are normalized
    unit_vectors = unit_vectors / torch.norm(unit_vectors, dim=1, keepdim=True)
    
    # Reshape unit_vectors to (B, H, W, C, 1)
    unit_vectors = unit_vectors.permute(0, 2, 3, 1).unsqueeze(-1)
    
    # Compute H matrix
    H_matrix = torch.eye(C, device=covariance_matrix.device).reshape(1, 1, 1, C, C) - 2 * torch.matmul(unit_vectors, unit_vectors.transpose(-1, -2))
    H_matrix = M.permute(0, 2, 3, 1).unsqueeze(-1)*H_matrix
    # Rotate the covariance matrix
    rotated_covariance_matrix = torch.matmul(torch.matmul(H_matrix, covariance_matrix), H_matrix.transpose(-1, -2))
    
    return rotated_covariance_matrix

def ensure_psd(matrix):
    """
    Ensure that the input matrix is positive semidefinite.
    
    Parameters:
    matrix : torch.Tensor
        Input matrix with shape (..., C, C)
    
    Returns:
    torch.Tensor
        PSD matrix with shape (..., C, C)
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = F.relu(eigenvalues)+0.1
    psd_matrix = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)
    return psd_matrix

import torch.distributions as dist

def loss_3D_gaussian(mu,sigma,y,mask):
    sigma = torch.nn.functional.elu(sigma)+1+0.01

    # Create the covariance matrix
    cov_matrix = build_covariance_matrix(sigma)
    #cov_matrix = rotate_covariance_matrix(cov_matrix, unit_vec, mask)
    #cov_matrix = ensure_psd(cov_matrix)
    # Reshape tensors to fit the requirements of MultivariateNormal
    mean = mu.permute(0, 2, 3, 1)
    y = y.permute(0, 2, 3, 1)

    # Define the multivariate normal distribution
    mvn = dist.MultivariateNormal(mean, covariance_matrix=cov_matrix)

    nll = -mvn.log_prob(y)*mask

    denom = torch.sum(mask)
    # Calculate the total variation loss
    loss = torch.sum(nll)/denom

    return loss

def cosine_similarity_loss(predicted, target, mask):
    # Normalize the normal vectors
    predicted_norm = F.normalize(predicted, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    
    # Compute the cosine similarity
    cosine_similarity = F.cosine_similarity(predicted_norm, target_norm, dim=1)
    # mask out invalid points
    cosine_similarity = mask * (1-cosine_similarity)
    denom = torch.sum(mask)
    # Calculate the total variation loss
    loss = torch.sum(cosine_similarity)/denom
    
    return loss

class Hist(torch.autograd.Function):
  
  @staticmethod
  def forward(ctx, sim, n_bins, w):

    # compute the step size in the histogram
    step = 1. / n_bins
    idx = sim / step

    lower = idx.floor()
    upper = idx.ceil()

    delta_u = idx - lower
    delta_l = upper - idx

    lower = lower.long()
    upper = upper.long()

    hist = torch.bincount(upper, delta_u * w, n_bins + 1) + torch.bincount( lower, delta_l * w, n_bins + 1)
    w_sum = w.sum()
    hist = hist / w_sum

    ctx.save_for_backward(upper, lower, w, w_sum)

    return hist
    

  @staticmethod
  def backward(ctx, grad_hist):
    upper, lower, w, w_sum = ctx.saved_tensors
    grad_sim = None
  
    grad_hist = grad_hist / w_sum
    grad_sim = (grad_hist[upper] - grad_hist[lower]) * w

    return grad_sim, None, None


class HistogramLoss(nn.Module):
  # from https://github.com/desa-lab/HistLoss
  def __init__(self):
    super(HistogramLoss, self).__init__()
    
    self.hist = Hist.apply

  def forward(self, sim_pos, sim_neg, n_bins, w_pos=None, w_neg=None):  
 
    sim_pos = sim_pos.flatten()
    sim_neg = sim_neg.flatten()

    # linearly transform similarity values to the range between 0 and 1
    max_pos, min_pos = torch.max(sim_pos.data), torch.min(sim_pos.data)
    max_neg, min_neg = torch.max(sim_neg.data), torch.min(sim_neg.data)

    max_ = max_pos if max_pos >= max_neg else max_neg
    min_ = min_pos if min_pos <= min_neg else min_neg

    sim_pos = (sim_pos - min_) / (max_ - min_)
    sim_neg = (sim_neg - min_) / (max_ - min_)

    # if w_pos is not None:
    #   w_pos = w_pos.data.flatten()
    #   assert sim.size() == w.size(), "Please make sure the size of the similarity tensor matches that of the weight tensor."
    # else:
    w_pos = torch.ones_like(sim_pos)

    # if w_neg is not None:
    #   w_neg = w_neg.data.flatten()
    #   assert sim.size() == w.size(), "Please make sure the size of the similarity tensor matches that of the weight tensor."
    # else:
    w_neg = torch.ones_like(sim_neg)
 
    pdf_pos = self.hist(sim_pos, n_bins, w_pos)
    pdf_neg = self.hist(sim_neg, n_bins, w_neg)

    cdf_pos = torch.cumsum(pdf_pos, dim=0)
    loss = (cdf_pos * pdf_neg).sum()

    return loss
  
import torch.distributions as D
  
def build_gmm_cdf(gmm_tensor, x_vals_image):
    """
    gmm_tensor: torch.Tensor of shape [B, 3, N], where:
      - gmm_tensor[:, 0, :] are the means of the Gaussians
      - gmm_tensor[:, 1, :] are the variances (we'll take sqrt to get stddev)
      - gmm_tensor[:, 2, :] are the weights of the Gaussians (these should sum to 1 for each batch)
      
    x_vals_image: torch.Tensor of shape [B, H, W, 1], where B is the batch size, H is the height, W is the width, and 1 is the single channel.
                 These are the points where we want to evaluate the CDF of the GMM.
    
    Returns: torch.Tensor of shape [B, H, W, 1], the CDF of the GMM evaluated at the image points.
    """
    
    means = gmm_tensor[:, 1, :]  # [B, N] - means for each Gaussian in the batch
    stddevs = gmm_tensor[:, 2, :]  # [B, N] - standard deviations for each Gaussian
    weights = gmm_tensor[:, 0, :]  # [B, N] - weights for each Gaussian component
    
    B, N = means.shape  # B = batch size, N = number of components
    H, W = x_vals_image.shape[2:]  # H = height, W = width
    

    # Create normal distributions
    normal_dist = D.Normal(means, stddevs)
    
    # Compute the CDF for each component at the image points
    component_cdfs = normal_dist.cdf(x_vals_image.reshape((-1,B,1)))  # [B, H, W, N]
    
    # Weight the component CDFs by the corresponding weights
    weighted_cdfs = component_cdfs * weights  # [B, H, W, N]
    
    # Sum across the components (axis=-1, the N dimension)
    gmm_cdf = weighted_cdfs.sum(dim=-1)  # [B, H, W, 1] - final GMM CDF
    gmm_cdf = gmm_cdf.reshape((B,1,H,W))
    return gmm_cdf


class CDFNornLoss(nn.Module):
  # from https://github.com/desa-lab/HistLoss
  def __init__(self):
    super(CDFNornLoss, self).__init__()
    
    self.hist = Hist.apply

  def forward(self, range_gt, range_pred, mixtures, mask):  
 
    range_gt_normed = build_gmm_cdf(mixtures, range_gt)
    range_pred_normed = build_gmm_cdf(mixtures, range_pred)
    loss = torch.linalg.norm(mask*torch.square(range_gt_normed - range_pred_normed)/2, dim = 1)
    denom = torch.sum(mask)
    # Calculate the total variation loss
    loss = torch.sum(loss)/denom
    return loss
  
import torch
import torch.nn.functional as F

# Median filter for batch images (B, C, H, W)
def median_filter(input, kernel_size=3):
    # Padding to maintain the original size
    padding = kernel_size // 2
    unfolded = F.unfold(input, kernel_size=kernel_size, padding=padding)
    unfolded = unfolded.view(input.size(0), input.size(1), kernel_size * kernel_size, -1)  # B, C, K*K, H*W
    median = unfolded.median(dim=2)[0]  # Take median along the kernel dimension
    output = median.view(input.size())  # Reshape back to the original shape (B, C, H, W)
    return output

class RPNLoss(torch.nn.Module):
    def __init__(self, kernel_size=3):
        super(RPNLoss, self).__init__()
        self.kernel_size = kernel_size


    def forward(self, pred_depth, target_depth, mask):

        # Apply median filtering to both predicted and target depth maps
        pred_median = median_filter(pred_depth, self.kernel_size)
        target_median = median_filter(target_depth, self.kernel_size)
        
        # Get Contrast
        pred_contrast = pred_depth - pred_median
        target_contrast = target_depth - target_median
        denom = torch.sum(mask, dim=(-2,-1), keepdim=True)
        mean_pred_contrast = torch.sum(mask*torch.abs(pred_contrast), dim=(-2,-1), keepdim=True)/denom
        mean_target_contrast = torch.sum(mask*torch.abs(target_contrast), dim=(-2,-1), keepdim=True)/denom

        loss= torch.abs((pred_contrast/mean_pred_contrast) - (target_contrast/mean_target_contrast))


        # Calculate the total variation loss
        denom = torch.sum(mask)
        loss = torch.sum(loss)/denom

        return loss
