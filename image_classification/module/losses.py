from torch.nn import functional as F

def get_loss_function(loss_function_str: str):

    if loss_function_str == 'nll':

        return F.nll_loss