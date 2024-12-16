import torch
import numpy as np

class DepthEvaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.abs_rel_errors = []
        self.sq_rel_errors = []
        self.rmse_errors = []
        self.rmse_log_errors = []
        self.a1_acc = []
        self.a2_acc = []
        self.a3_acc = []

    def update(self, gt, pred, mask):

        """Update metrics with a new batch of data."""
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = self.compute_errors(gt, pred, mask)
        self.abs_rel_errors.append(abs_rel)
        self.sq_rel_errors.append(sq_rel)
        self.rmse_errors.append(rmse)
        self.rmse_log_errors.append(rmse_log)
        self.a1_acc.append(a1)
        self.a2_acc.append(a2)
        self.a3_acc.append(a3)

    def compute_errors(self, gt, pred, mask):
        
        """Computation of error metrics between predicted and ground truth depths."""
        divisor = torch.sum(mask)
        # Avoid division by zero
        eps = 1e-6
        pred = torch.clamp(pred, min=eps)
        gt = torch.clamp(gt, min=eps)
        
        # Calculate the ratio thresholds
        thresh = torch.max(gt / pred, pred / gt)
        
        # Proportion of ratios below thresholds 1.25, 1.25^2, and 1.25^3
        a1 = ((mask*(thresh < 1.25).float()).sum()/divisor).item()
        a2 = ((mask*(thresh < 1.25**2).float()).sum()/divisor).item()
        a3 = ((mask*(thresh < 1.25**3).float()).sum()/divisor).item()

        # Root Mean Squared Error
        rmse = torch.sqrt((mask*(gt - pred) ** 2).sum()/divisor).item()

        # Logarithmic Root Mean Squared Error
        rmse_log = torch.sqrt((mask*((torch.log(gt) - torch.log(pred))) ** 2).sum()/divisor).item()

        # Absolute Relative Difference
        abs_rel = ((mask*torch.abs(gt - pred) / gt).sum()/divisor).item()

        # Squared Relative Difference
        sq_rel = (((mask*(gt - pred) ** 2) / gt).sum()/divisor).item()

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

    def compute_final_metrics(self):
        """Compute final metrics after processing all batches."""
        abs_rel = np.mean(self.abs_rel_errors)
        sq_rel = np.mean(self.sq_rel_errors)
        rmse = np.mean(self.rmse_errors)
        rmse_log = np.mean(self.rmse_log_errors)
        a1 = np.mean(self.a1_acc)
        a2 = np.mean(self.a2_acc)
        a3 = np.mean(self.a3_acc)
        
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3