import json
import pandas as pd
import os
import logging
from typing import List, Dict, Tuple
from scipy.stats import kendalltau
import re 
import utils
import conjugate_priors 
import numpy as np 
from alabEBM import get_params_path

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


biomarker_order_V1 = {
    'AB': 1, 'ADAS': 2, 'AVLT-Sum': 3, 'FUS-FCI': 4, 'FUS-GMI': 5, 
    'HIP-FCI': 6, 'HIP-GMI': 7, 'MMSE': 8, 'P-Tau': 9, 'PCC-FCI': 10
}

biomarker_order_V2 = {
    'AB': 10, 'ADAS': 9, 'AVLT-Sum': 8, 'FUS-FCI': 7, 'FUS-GMI': 6, 
    'HIP-FCI': 5, 'HIP-GMI': 4, 'MMSE': 3, 'P-Tau': 2, 'PCC-FCI': 1
}

def process_participant_data(
    data: pd.DataFrame,
    biomarker_order_V1: Dict[str, int],
    biomarker_order_V2: Dict[str, int]
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Preprocess participant data into NumPy arrays for efficient computation.

    Args:
        data (pd.DataFrame): Raw participant data.
        biomarker_order_V1 (Dict): first biomarker order dict
        biomarker_order_V2 (Dict): second biomarker order dict

    Returns:
        Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]: A dictionary where keys are participant IDs,
            and values are tuples of (measurements, S_n, biomarkers).
    """
    def add_sn(row):
        if row.participant <= 49:
            return biomarker_order_V1[row.biomarker]
        else:
            return biomarker_order_V2[row.biomarker]
    data['S_n'] = data.apply(add_sn, axis = 1)
    participant_data = {}
    for participant, pdata in data.groupby('participant'):
        measurements = pdata['measurement'].values 
        S_n = pdata['S_n'].values 
        biomarkers = pdata['biomarker'].values  
        participant_data[participant] = (measurements, S_n, biomarkers)
    return participant_data

def calculate_upper_limit(
    participant_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    non_diseased_ids: np.ndarray,
    theta_phi: Dict[str, Dict[str, float]],
    diseased_stages: np.ndarray,
) -> float:
    """
    Calculate the total log likelihood across all participants and update their disease stages.

    Args:
        participant_data (Dict): Dictionary containing participant data. Keys are participant IDs, and values
            are tuples of (measurements, S_n, biomarkers).
        non_diseased_ids (np.ndarray): Array of participant IDs who are non-diseased.
        theta_phi (Dict): Theta and phi parameters for each biomarker.
        diseased_stages (np.ndarray): Array of possible disease stages.

    Returns:
        float: Total log likelihood across all participants.
    """
    total_ln_likelihood = 0.0 
    for participant, (measurements, S_n, biomarkers) in participant_data.items():
        if participant in non_diseased_ids:
            ln_likelihood = utils.compute_ln_likelihood(
                measurements, S_n, biomarkers, k_j = 0, theta_phi = theta_phi)
        else:
            ln_stage_likelihoods = np.array([
                utils.compute_ln_likelihood(
                    measurements, S_n, biomarkers, k_j = k_j, theta_phi=theta_phi
                ) for k_j in diseased_stages
            ])
            # Use log-sum-exp trick for numerical stability
            max_ln_likelihood = np.max(ln_stage_likelihoods)
            stage_likelihoods = np.exp(ln_stage_likelihoods - max_ln_likelihood)
            likelihood_sum = np.sum(stage_likelihoods)
            ln_likelihood = max_ln_likelihood + np.log(likelihood_sum)
        total_ln_likelihood += ln_likelihood
    return total_ln_likelihood


if __name__=="__main__":
    folder_name = 'conjugate_priors'
    os.makedirs(folder_name, exist_ok = True)
    data_file = 'data/data.csv'
    data_we_have = pd.read_csv(data_file)

    biomarkers = data_we_have.biomarker.unique()
    n_stages = len(biomarkers) + 1
    diseased_stages = np.arange(start=1, stop=n_stages, step=1)
    non_diseased_ids = data_we_have.loc[data_we_have.diseased == False].participant.unique()

    real_theta_phi_file = get_params_path()
    # Load theta and phi values from the JSON file
    try:
        with open(real_theta_phi_file) as f:
            real_theta_phi = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {real_theta_phi} not fount")
    except json.JSONDecodeError:
        raise ValueError(
            f"File {real_theta_phi_file} is not a valid JSON file.")

    participant_data = process_participant_data(
        data_we_have,
        biomarker_order_V1,
        biomarker_order_V2
    )

    upper_limit = calculate_upper_limit(
        participant_data,
        non_diseased_ids,
        real_theta_phi,
        diseased_stages,
    )

    all_orders, log_likelihoods, participant_order_assignments = conjugate_priors.metropolis_hastings_subtype_conjugate_priors(
        data_we_have = data_we_have,
        iterations = 2000,
        n_shuffle = 2,
        upper_limit = upper_limit
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


