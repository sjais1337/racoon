import torch
import copy
from django.conf import settings
from .models_pytorch import LogisticRegressionModel, SimpleNN 

def get_pytorch_model_instance(model_config_key):
    if model_config_key not in settings.FL_MODEL_CONFIGS:
        raise ValueError(f"Model configuration '{model_config_key}' not found in settings.FL_MODEL_CONFIGS.")

    config = settings.FL_MODEL_CONFIGS[model_config_key]

    model_class_name = config.get('model_class_name')
    input_dim = config.get('input_dim')
    output_dim = config.get('output_dim')

    if model_class_name == 'LogisticRegressionModel':
        model = LogisticRegressionModel(input_dim, output_dim)
    elif model_class_name == 'SimpleNN':
        hidden_size = config.get('hidden_size') 
        model = SimpleNN(input_size=input_dim, hidden_size=hidden_size, num_classes=output_dim)
    else: 
        raise ValueError(f"Unknown model_class_name {model_class_name}. ")

    return model

def server_aggregate_weights_util(base_model_state_dict_structure, client_model_state_dicts):
    if not client_model_state_dicts:
        return None 

    aggregated_weights = copy.deepcopy(base_model_state_dict_structure)

    for key in aggregated_weights:
        aggregated_weights[key] = torch.zeros_like(aggregated_weights[key], dtype=aggregated_weights[key].dtype)

    num_clients_updated = len(client_model_state_dicts)
    for client_weights in client_model_state_dicts:
        for key in aggregated_weights:
            if key in client_weights: 
                aggregated_weights[key] += client_weights[key]
            else:
                print(f"Warning: Key {key} not found in a client's weights during aggregation.")

    for key in aggregated_weights:
        aggregated_weights[key] = torch.div(aggregated_weights[key], float(num_clients_updated)) 

    return aggregated_weights
