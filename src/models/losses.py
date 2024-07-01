import torch
import torch.nn as nn
import torch.nn.functional as F

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

def metric_3D_loss(pc_img_gt, pc_img_pred, mask):
    
    #loss = torch.square(mask*pc_img_gt-mask*pc_img_pred)
    loss = torch.nn.functional.huber_loss(mask*pc_img_pred, mask*pc_img_gt, reduction='sum', delta=1.0)
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

def torch_build_normal_xyz(xyz):
    '''
    @param xyz: ndarray with shape (h,w,3) containing a stagged point cloud
    @param norm_factor: int for the smoothing in Schaar filter
    '''
    scharr_x_weight = torch.tensor([[-3, 0, 3],
                                            [-10, 0, 10],
                                            [-3, 0, 3]], dtype=torch.float32, device=xyz.device, requires_grad=False).unsqueeze(0).unsqueeze(0)

    scharr_y_weight = torch.tensor([[3, 10, 3],
                                            [0, 0, 0],
                                            [-3, -10, -3]], dtype=torch.float32, device=xyz.device, requires_grad=False).unsqueeze(0).unsqueeze(0)

    x = xyz[:,0:1,...]
    y = xyz[:,1:2,...]
    z = xyz[:,2:3,...]

    Sxx = F.conv2d(x, weight=scharr_x_weight, padding=1)
    Sxy = F.conv2d(x, weight=scharr_y_weight, padding=1)

    Syx = F.conv2d(y, weight=scharr_x_weight, padding=1)
    Syy = F.conv2d(y, weight=scharr_y_weight, padding=1)

    Szx = F.conv2d(z, weight=scharr_x_weight, padding=1)
    Szy = F.conv2d(z, weight=scharr_y_weight, padding=1)


    #build cross product
    normal = -torch.cat((Syx*Szy - Szx*Syy,
                        Szx*Sxy - Szy*Sxx,
                        Sxx*Syy - Syx*Sxy), dim=1)


    n = torch.norm(normal, dim=1,  keepdim=True) + torch.tensor(1e-10, dtype=torch.float64)
    normal = normal/n 

    return normal

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