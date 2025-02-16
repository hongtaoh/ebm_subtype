import json
import pandas as pd
import os
import logging
from typing import List, Dict
from scipy.stats import kendalltau
import re 
import utils
import hard_kmeans 
import numpy as np 

# Configure logging to display INFO messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def convert_numpy(obj):
    """ Recursively converts NumPy types to Python native types """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj  # Return original if not a NumPy type


if __name__=="__main__":
    folder_name = 'hard_kmeans'
    os.makedirs(folder_name, exist_ok = True)
    data_file = 'data/data.csv'
    data_we_have = pd.read_csv(data_file)
    all_orders, log_likelihoods, participant_order_assignments = hard_kmeans.metropolis_hastings_subtype_hard_kmeans(
        data_we_have = data_we_have,
        iterations = 2000,
        n_shuffle = 2
    )
    all_orders_serializable = convert_numpy(all_orders)
    with open(f"{folder_name}/all_orders.json", "w") as f:
        json.dump(all_orders_serializable, f, indent = 4)
    
    log_likelihoods_serializable = convert_numpy(log_likelihoods)
    with open(f"{folder_name}/log_likelihoods.json", "w") as f:
        json.dump(log_likelihoods_serializable, f, indent = 4)
    
    participant_order_assignments_serializable = convert_numpy(participant_order_assignments)
    with open(f"{folder_name}/participant_order_assignments.json", "w") as f:
        json.dump(participant_order_assignments_serializable, f, indent = 4)


