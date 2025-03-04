import json
import pandas as pd
import numpy as np 
import os
import logging
from typing import List, Dict, Optional, Tuple
from scipy.stats import kendalltau
import re 

# Import utility functions
from alabEBM.utils.visualization import save_heatmap, save_traceplot 
from alabEBM.utils.logging_utils import setup_logging 
from alabEBM.utils.data_processing import get_theta_phi_estimates, obtain_most_likely_order_dic
from alabEBM.utils.runners import extract_fname, cleanup_old_files
from alabEBM.utils import data_processing as data_utils

# Import algorithms
import conjugate_priors 
from alabEBM import get_params_path

"""
`process_participant_data` adds S_n to the original data based on the two biomarker orderings. 
`calculate_upper_limit` calculates the upper limit of probability (when the ordering is correct, 
    the prob is the highest and any guesses cannot exceed that)
"""

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
            ln_likelihood = data_utils.compute_ln_likelihood(
                measurements, S_n, biomarkers, k_j = 0, theta_phi = theta_phi)
        else:
            ln_stage_likelihoods = np.array([
                data_utils.compute_ln_likelihood(
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

def run_ebm_subtype(
    data_file: str,
    algorithm: str, 
    n_iter: int = 2000,
    n_shuffle: int = 2,
    burn_in: int = 1000,
    thinning: int = 50,
) -> Dict:
    """
    Run the metropolis hastings algorithm and save results 

    Args:
        data_file (str): Path to the input CSV file with biomarker data.
        algorithm (str): Choose from 'hard_kmeans', 'soft_kmeans', and 'conjugate_priors'.
        n_iter (int): Number of iterations for the Metropolis-Hastings algorithm.
        n_shuffle (int): Number of shuffles per iteration.
        burn_in (int): Burn-in period for the MCMC chain.
        thinning (int): Thinning interval for the MCMC chain.
        correct_ordering (Optional[Dict[str, int]]): biomarker name: the initial correct order of it (if known)

    Returns:
        Dict: Results
    """
    # Extract target tau
    target_tau = float(data_file.split("_")[-1].split(".csv")[0])
    # Folder to save all outputs
    output_dir = algorithm
    fname = extract_fname(data_file)

    # First do cleanup
    logging.info(f"Starting cleanup for {algorithm.replace('_', ' ')}...")
    cleanup_old_files(output_dir, fname)

    # Then create directories
    os.makedirs(output_dir, exist_ok=True)
    heatmap_folder = f"{output_dir}/heatmaps"
    traceplot_folder = f"{output_dir}/traceplots"
    results_folder = f"{output_dir}/results"
    logs_folder = f"{output_dir}/records"

    os.makedirs(heatmap_folder, exist_ok=True)
    os.makedirs(traceplot_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    # Finally set up logging
    log_file = f"{logs_folder}/{fname}.log"
    setup_logging(log_file)

    # Log the start of the run
    logging.info(f"Running {algorithm.replace('_', ' ')} for file: {fname}")
    logging.getLogger().handlers[0].flush()  # Flush logs immediately

    # Load data
    try:
        data = pd.read_csv(data_file)
    except Exception as e:
        logging.error(f"Error reading data file: {e}")
        raise

    # Determine the number of biomarkers
    n_biomarkers = len(data.biomarker.unique())
    logging.info(f"Number of biomarkers: {n_biomarkers}")

    """
    Calculate upper_limit
    """
    biomarkers = data.biomarker.unique()
    n_stages = len(biomarkers) + 1
    diseased_stages = np.arange(start=1, stop=n_stages, step=1)
    non_diseased_ids = data.loc[data.diseased == False].participant.unique()

    real_theta_phi_file = get_params_path()
    # Load theta and phi values from the JSON file
    try:
        with open(real_theta_phi_file) as f:
            real_theta_phi = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {real_theta_phi_file} not found")
    except json.JSONDecodeError:
        raise ValueError(
            f"File {real_theta_phi_file} is not a valid JSON file.")
    
    orders_file = 'data/orders.json'
    with open(orders_file) as f:
        orders = json.load(f)
    
    participant_data = process_participant_data(
        data,
        orders['1.0']['order'],
        orders[f"{target_tau}"]['order'],
    )

    upper_limit = calculate_upper_limit(
        participant_data,
        non_diseased_ids,
        real_theta_phi,
        diseased_stages,
    )
    """
    """

    # Run the EBM-subtype algorithm
    try:
        if algorithm == 'soft_kmeans':
            pass
            # accepted_order_dicts, log_likelihoods = metropolis_hastings_soft_kmeans(
            #     data, n_iter, n_shuffle
            # )
        elif algorithm == 'hard_kmeans':
            pass
            # accepted_order_dicts, log_likelihoods = metropolis_hastings_hard_kmeans(
            #     data, n_iter, n_shuffle
            # )
        elif algorithm == 'conjugate_priors':
            accepted_order_dicts, log_likelihoods, participant_order_assignments = conjugate_priors.metropolis_hastings_subtype_conjugate_priors(
                data_we_have = data,
                iterations = n_iter,
                n_shuffle = n_shuffle,
                upper_limit = upper_limit
            )
        else:
            raise ValueError("You must choose from 'hard_kmeans', 'soft_kmeans', and 'conjugate_priors'!")
    except Exception as e:
        logging.error(f"Error in Metropolis-Hastings algorithm: {e}")
        raise

    # Get most likely order 
    most_likely_order1_dic = obtain_most_likely_order_dic(
        accepted_order_dicts['order1'], burn_in, thinning
    )
    most_likely_order2_dic = obtain_most_likely_order_dic(
        accepted_order_dicts['order2'], burn_in, thinning
    )

    def best_kendall_tau_pairing(
        real_order1_dict:Dict[str, int], 
        real_order2_dict:Dict[str, int], 
        guessed_order1_dict:Dict[str, int], 
        guessed_order2_dict:Dict[str, int], 
        ) -> Tuple[Dict, Dict, Dict, Dict, float, float, str]:
        #sort keys for each dict, so we can compare the values safely
        real_order1_dict = {k: real_order1_dict[k] for k in sorted(real_order1_dict.keys())}
        real_order2_dict = {k: real_order2_dict[k] for k in sorted(real_order2_dict.keys())}
        guessed_order1_dict = {k: guessed_order1_dict[k] for k in sorted(guessed_order1_dict.keys())}
        guessed_order2_dict = {k: guessed_order2_dict[k] for k in sorted(guessed_order2_dict.keys())}

        real_order1 = list(real_order1_dict.values())
        real_order2 = list(real_order2_dict.values())
        guessed_order1 = list(guessed_order1_dict.values())
        guessed_order2 = list(guessed_order2_dict.values())

        tau1, _ = kendalltau(real_order1, guessed_order1)
        tau2, _ = kendalltau(real_order2, guessed_order2)
        tau3, _ = kendalltau(real_order1, guessed_order2)
        tau4, _ = kendalltau(real_order2, guessed_order1)

        direct_match = tau1 + tau2
        swapped_match = tau3 + tau4

        # We want the pairing that maximize the tau sum
        if swapped_match > direct_match:
            return (real_order1_dict, guessed_order2_dict, real_order2_dict, guessed_order1_dict, tau3, tau4, 'Swapped')
        else:
            return (real_order1_dict, guessed_order1_dict, real_order2_dict, guessed_order2_dict, tau1, tau2, 'Direct')

    real_order1_dict, guessed_order1_dict, real_order2_dict, guessed_order2_dict, tau1, tau2, paring_result = best_kendall_tau_pairing(
        orders['1.0']['order'],
        orders[f"{target_tau}"]['order'],
        most_likely_order1_dic,
        most_likely_order2_dic
    )

    if paring_result == 'Swapped':
        correct_order1 = orders[f"{target_tau}"]['order']
        correct_order2 = orders['1.0']['order']
    else:
        correct_order1 = orders['1.0']['order']
        correct_order2 = orders[f"{target_tau}"]['order']

    # Save heatmap
    try:
        save_heatmap(
            accepted_order_dicts['order1'],
            burn_in,
            thinning,
            folder_name=heatmap_folder,
            file_name=f"{fname}_heatmap_{algorithm}_order1",
            title=f"Heatmap of {fname} using {algorithm}_order1",
            correct_ordering = correct_order1
        )
    except Exception as e:
        logging.error(f"Error generating heatmap: {e}")
        raise

    # Save heatmap
    try:
        save_heatmap(
            accepted_order_dicts['order2'],
            burn_in,
            thinning,
            folder_name=heatmap_folder,
            file_name=f"{fname}_heatmap_{algorithm}_order2",
            title=f"Heatmap of {fname} using {algorithm}_order2",
            correct_ordering = correct_order2
        )
    except Exception as e:
        logging.error(f"Error generating heatmap: {e}")
        raise

    # Save trace plot
    try:
        save_traceplot(log_likelihoods['order1'], traceplot_folder, f"{fname}_traceplot_{algorithm}_order1")
    except Exception as e:
        logging.error(f"Error generating trace plot: {e}")
        raise 

    # Save trace plot
    try:
        save_traceplot(log_likelihoods['order2'], traceplot_folder, f"{fname}_traceplot_{algorithm}_order2")
    except Exception as e:
        logging.error(f"Error generating trace plot: {e}")
        raise 

    # Save results 
    results = {
        'total_tau': tau1+tau2,
        "n_iter": n_iter,
        "most_likely_order1": guessed_order1_dict,
        "kendalls_tau1": tau1, 
        "original_order1": real_order1_dict,
        "most_likely_order2": guessed_order2_dict,
        "kendalls_tau2": tau2, 
        "original_order2": real_order2_dict,
    }
    try:
        with open(f"{results_folder}/{fname}_results.json", "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        logging.error(f"Error writing results to file: {e}")
        raise 
    logging.info(f"Results saved to {results_folder}/{fname}_results.json")

    # Clean up logging handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    return results