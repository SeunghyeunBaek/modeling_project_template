
from sklearn.metrics import accuracy_score, confusion_matrix

def get_metric_function(metric_function_str):
    
    metric_funcion = None
    
    if metric_function_str == 'accuracy':
        metric_funcion = accuracy_score
    
    elif metric_function_str == 'confusion_matrix':
        metric_funcion = confusion_matrix

    return metric_funcion
