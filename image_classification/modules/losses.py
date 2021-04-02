from torch.nn import functional as F
import torch


def get_loss_function(loss_function_str: str):

    if loss_function_str == 'nll':

        return F.nll_loss

    elif loss_function_str == 'focal':

        return FocalLoss


## 예시
class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha, gamma):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss