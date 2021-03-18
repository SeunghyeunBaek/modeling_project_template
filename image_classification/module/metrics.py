
from sklearn.metrics import accuracy_score

def get_metric_function(metric_function_str):
    
    metric_funcion = None
    
    if metric_function_str == 'accuracy':
        metric_funcion = accuracy_score

    return metric_funcion