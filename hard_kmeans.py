import numpy as np
import numba 
import pandas as pd 
from typing import List, Dict, Tuple
import utils
import logging
from collections import defaultdict

def preprocess_participant_data(
    data_we_have: pd.DataFrame, current_order_dict: Dict
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Preprocess participant data into NumPy arrays for efficient computation.

    Args:
        data (pd.DataFrame): Raw participant data.
        current_order_dict (Dict): Mapping of biomarkers to stages.

    Returns:
        Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]: A dictionary where keys are participant IDs,
            and values are tuples of (measurements, S_n, biomarkers).
    """
    # Create a copy instead of modifying the original DataFrame
    data_copy = data_we_have.copy()
    data_copy['S_n'] = data_copy['biomarker'].map(current_order_dict)

    participant_data = {}
    for participant, pdata in data_copy.groupby('participant'):
        measurements = pdata['measurement'].values 
        S_n = pdata['S_n'].values 
        biomarkers = pdata['biomarker'].values  
        participant_data[participant] = (measurements, S_n, biomarkers)
    return participant_data

def calculate_all_participant_ln_likelihood(
    participant_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    non_diseased_ids: np.ndarray,
    theta_phi: Dict[str, Dict[str, float]],
    diseased_stages: np.ndarray
    ) -> float:
    """Calculate the total log likelihood across all participants."""
    total_ln_likelihood = 0.0 
    for participant, (measurements, S_n, biomarkers) in participant_data.items():
        if participant in non_diseased_ids:
            ln_likelihood = utils.compute_ln_likelihood(
                measurements, S_n, biomarkers, k_j = 0, theta_phi = theta_phi
            )
        else:
            ln_stage_likelihoods = [
                utils.compute_ln_likelihood(
                    measurements, S_n, biomarkers, k_j = k_j, theta_phi=theta_phi
                ) for k_j in diseased_stages
            ]
            # Use log-sum-exp trick for numerical stability
            max_ln_likelihood = np.max(ln_stage_likelihoods)
            stage_likelihoods = np.exp(ln_stage_likelihoods - max_ln_likelihood)
            likelihood_sum = np.sum(stage_likelihoods)
            ln_likelihood = max_ln_likelihood + np.log(likelihood_sum)
            
        total_ln_likelihood += ln_likelihood

    return total_ln_likelihood


def calculate_ln_likelihood_per_participant(
    participant_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    non_diseased_ids: np.ndarray,
    theta_phi: Dict[str, Dict[str, float]],
    diseased_stages: np.ndarray
) -> Dict[int, float]:
    """Calculate the log likelihood per participant."""
    ln_likelihoods = {}
    for participant, (measurements, S_n, biomarkers) in participant_data.items():
        if participant in non_diseased_ids:
            ln_likelihood = utils.compute_ln_likelihood(
                measurements, S_n, biomarkers, k_j=0, theta_phi=theta_phi
            )
        else:
            ln_stage_likelihoods = [
                utils.compute_ln_likelihood(
                    measurements, S_n, biomarkers, k_j=k_j, theta_phi=theta_phi
                )
                for k_j in diseased_stages
            ]
            # Use log-sum-exp trick for numerical stability
            max_ln_likelihood = np.max(ln_stage_likelihoods)
            stage_likelihoods = np.exp(ln_stage_likelihoods - max_ln_likelihood)
            likelihood_sum = np.sum(stage_likelihoods)
            ln_likelihood = max_ln_likelihood + np.log(likelihood_sum)

        ln_likelihoods[participant] = ln_likelihood
    return ln_likelihoods

def metropolis_hastings_subtype_hard_kmeans(
    data_we_have: pd.DataFrame,
    iterations: int, 
    n_shuffle: int
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Metropolis-Hastings clustering algorithm."""

    n_participants = len(data_we_have.participant.unique())
    biomarkers = data_we_have.biomarker.unique()
    n_stages = len(biomarkers) + 1

    diseased_stages = np.arange(1, n_stages)
    non_diseased_ids = data_we_have.loc[data_we_have.diseased == False].participant.unique()

    theta_phi_default = utils.get_theta_phi_estimates(data_we_have)

    logging.info(f"Default Theta and Phi Parameters: {theta_phi_default.items()} ")

    current_order1 = np.random.permutation(np.arange(1, n_stages))
    current_order1_dict = dict(zip(biomarkers, current_order1))
    current_ln_likelihood1 = -np.inf
    acceptance_count1 = 0

    current_order2 = np.random.permutation(np.arange(1, n_stages))
    current_order2_dict = dict(zip(biomarkers, current_order2))
    current_ln_likelihood2 = -np.inf
    acceptance_count2 = 0

    # Note that this records only the current accepted orders in each iteration
    all_orders = []
    # This records all log likelihoods
    log_likelihoods = []
    participant_order_assignments = []

    for iteration in range(iterations):
        log_likelihoods.append({'order1': current_ln_likelihood1, 'order2': current_ln_likelihood2})
        # Suffle the order 
        # Note that copy here is necessary because without it, each iteration is 
        # shuffling the order in the last iteration. 
        # With copy, we can ensure that the current state remains unchanged until
        # the proposed state is accepted.  

        new_order1 = current_order1.copy()
        utils.shuffle_order(new_order1, n_shuffle)
        new_order1_dict = dict(zip(biomarkers, new_order1))

        new_order2 = current_order2.copy()
        utils.shuffle_order(new_order2, n_shuffle)
        new_order2_dict = dict(zip(biomarkers, new_order2))

        # Update participant data with the new order dict
        participant_data1 = preprocess_participant_data(data_we_have, new_order1_dict)
        participant_data2 = preprocess_participant_data(data_we_have, new_order2_dict)

        # Calculate likelihoods
        ln_likelihoods_order1 = calculate_ln_likelihood_per_participant(
            participant_data1, non_diseased_ids, theta_phi_default, diseased_stages
        )
        ln_likelihoods_order2 = calculate_ln_likelihood_per_participant(
            participant_data2, non_diseased_ids, theta_phi_default, diseased_stages
        )
        new_assignments = {}
        for p in range(n_participants):
            if ln_likelihoods_order1[p] > ln_likelihoods_order2[p]:
                new_assignments[p] = 1
            else:
                new_assignments[p] = 2

        participant_order_assignments.append(new_assignments)
        
        new_order1_ln_likelihood = sum(
            ln_likelihoods_order1[p] for p in range(n_participants) if new_assignments[p]==1
        )
        new_order2_ln_likelihood = sum(
            ln_likelihoods_order2[p] for p in range(n_participants) if new_assignments[p]==2
        )
        
        delta1 = new_order1_ln_likelihood - current_ln_likelihood1
        delta2 = new_order2_ln_likelihood - current_ln_likelihood2

        # Compute acceptance probability safely
        if delta1 > 0:
            prob_accept1 = 1.0  # Always accept improvements
        else:
            prob_accept1 = np.exp(delta1)  # Only exponentiate negative deltas

        # Compute acceptance probability safely
        if delta2 > 0:
            prob_accept2 = 1.0  # Always accept improvements
        else:
            prob_accept2 = np.exp(delta2)  # Only exponentiate negative deltas

        # prob_accept = np.exp(ln_likelihood - current_ln_likelihood)
        # np.exp(a)/np.exp(b) = np.exp(a - b)
        # if a > b, then np.exp(a - b) > 1
        
        # Accept or reject 
        # it will definitly update at the first iteration
        if np.random.rand() < prob_accept1:
            current_order1 = new_order1
            current_ln_likelihood1 = new_order1_ln_likelihood
            current_order1_dict = new_order1_dict 
            acceptance_count1 += 1

        if np.random.rand() < prob_accept2:
            current_order2 = new_order2
            current_ln_likelihood2 = new_order2_ln_likelihood
            current_order2_dict = new_order2_dict 
            acceptance_count2 += 1
        
        all_orders.append({"order1": current_order1_dict, "order2": current_order2_dict})

        # Log progress
        if (iteration + 1) % max(10, iterations // 10) == 0:
            acceptance_ratio1 = 100 * acceptance_count1 / (iteration + 1)
            acceptance_ratio2 = 100 * acceptance_count2 / (iteration + 1)
            logging.info(
                f"Iteration {iteration + 1}/{iterations}, "
                f"Acceptance Ratio 1: {acceptance_ratio1:.2f}%, "
                f"Log Likelihood 1: {current_ln_likelihood1:.4f}, "
                f"Current Accepted Order1: {current_order1_dict}, "
                f"Acceptance Ratio 2: {acceptance_ratio2:.2f}%, "
                f"Log Likelihood 2: {current_ln_likelihood2:.4f}, "
                f"Current Accepted Order2: {current_order2_dict}"
            )

    return all_orders, log_likelihoods, participant_order_assignments