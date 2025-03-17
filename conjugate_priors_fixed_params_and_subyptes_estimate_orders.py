import numpy as np 
import pandas as pd 
from alabebm.utils import data_processing as utils
from alabebm.algorithms import soft_kmeans_algo as sk 
from alabebm.algorithms import conjugate_priors_algo as cp
from typing import List, Dict, Tuple
import logging 
from collections import defaultdict 

def per_participant_compute_ln_likelihood_and_stage_likelihoods(
    participant_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    non_diseased_ids: np.ndarray,
    theta_phi: Dict[str, Dict[str, float]],
    diseased_stages: np.ndarray,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Calculate the likelihood for each participant and update their disease stages.

    Args:
        participant_data (Dict): Dictionary containing participant data. Keys are participant IDs, and values
            are tuples of (measurements, S_n, biomarkers).
        non_diseased_ids (np.ndarray): Array of participant IDs who are non-diseased.
        theta_phi (Dict): Theta and phi parameters for each biomarker.
        diseased_stages (np.ndarray): Array of possible disease stages.

    Returns:
        ln_likelihoods (Dict[int, float]): For each participant, calculate ln(a1+a2+a3...). That is,
            the log likelihood of this sequence of biomarker measurements given S for this participant
        stage_likelihoods_posteriors (Dict[int, float]): probability of this participant in different disease stages;
            Normalized, thus sum to 1
    """
    ln_likelihoods = {}
    stage_likelihoods_posteriors = {} # only for diseased participants 
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
            # Proof: https://hongtaoh.com/en/2024/12/14/log-sum-exp/
            ln_likelihood = max_ln_likelihood + np.log(likelihood_sum)
            stage_likelihoods_posteriors[participant] = stage_likelihoods/likelihood_sum
        ln_likelihoods[participant] = ln_likelihood
    return ln_likelihoods, stage_likelihoods_posteriors

def split_data_by_subtype(
    data: pd.DataFrame,
    assignments: Dict[int, int]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into subtype-specific datasets"""
    subtype1_ids = [p for p, s in assignments.items() if s == 1]
    subtype2_ids = [p for p, s in assignments.items() if s == 2]

    data_subtype1 = data[data.participant.isin(subtype1_ids)].copy()
    data_subtype2 = data[data.participant.isin(subtype2_ids)].copy()
    return data_subtype1, data_subtype2

# fixed theta_phi + two orders -> subtype
# fixed theta_phi + subtype assignment --> two orders
# fixed theta_phi -> two orders + subtype assignments

def metropolis_hastings_subtype_conjugate_priors(
    data_we_have: pd.DataFrame,
    iterations: int,
    n_shuffle: int,
    real_theta_phi: Dict[str, Dict[str, float]],
) -> Tuple[List[Dict[str, Dict[str, int]]], List[Dict[str, float]], List[Dict[str, int]]]:
    """
    Perform Metropolis-Hastings sampling with conjugate priors to estimate biomarker orderings.

    Args:
        data_we_have (pd.DataFrame): Raw participant data.
        iterations (int): Number of iterations for the algorithm.
        n_shuffle (int): Number of swaps to perform when shuffling the order.
        upper_limit (float): the total likelihood assuming we know real_theta_phi, two orders (and associated participants), and S_n

    Returns:
        Tuple[List[Dict], List[float]]: 
            - List of accepted biomarker orderings at each iteration.
            - List of log likelihoods at each iteration.
    """
    n_participants = len(data_we_have.participant.unique())
    biomarkers = data_we_have.biomarker.unique()
    n_stages = len(biomarkers) + 1
    diseased_stages = np.arange(start=1, stop=n_stages, step=1)
    non_diseased_ids = data_we_have.loc[data_we_have.diseased == False].participant.unique()
    theta_phi_estimates = real_theta_phi

    # initialize an ordering and likelihood
    order1 = np.random.permutation(np.arange(1, n_stages))
    # order1 = np.arange(1,11)
    order1_dict = dict(zip(biomarkers, order1))
    ln_likelihood1 = -np.inf
    acceptance_count1 = 0

    # initialize an ordering and likelihood
    order2 = np.random.permutation(np.arange(1, n_stages))
    # order2 = np.array([1,2,3,5,4,6,7,8,10,9])
    order2_dict = dict(zip(biomarkers, order2))
    ln_likelihood2 = -np.inf
    acceptance_count2 = 0

    # Note that this records only the current accepted orders in each iteration
    all_orders = defaultdict(list)
    # This records all log likelihoods
    log_likelihoods = defaultdict(list)
    participant_subtype_assignment_history = []
    participant_subtype_assignment = {}
    for p in range(n_participants):
        if p < n_participants//2:
            participant_subtype_assignment[p] = 1
        else:
            participant_subtype_assignment[p] = 2

    data_subtype1, data_subtype2 = split_data_by_subtype(data_we_have, participant_subtype_assignment)
    
    for iteration in range(iterations):
        log_likelihoods['order1'].append(ln_likelihood1)
        log_likelihoods['order2'].append(ln_likelihood2)

        new_order1 = order1.copy()
        utils.shuffle_order(new_order1, n_shuffle)
        new_order1_dict = dict(zip(biomarkers, new_order1))

        new_order2 = order2.copy()
        utils.shuffle_order(new_order2, n_shuffle)
        new_order2_dict = dict(zip(biomarkers, new_order2))

        participant_data1 = sk.preprocess_participant_data(data_subtype1, new_order1_dict)
        participant_data2 = sk.preprocess_participant_data(data_subtype2, new_order2_dict)
        
        new_ln_likelihood1, _ = sk.compute_total_ln_likelihood_and_stage_likelihoods(
            participant_data1,
            non_diseased_ids,
            theta_phi_estimates,
            diseased_stages
        )

        new_ln_likelihood2, _ = sk.compute_total_ln_likelihood_and_stage_likelihoods(
            participant_data2,
            non_diseased_ids,
            theta_phi_estimates,
            diseased_stages
        )

        # Compute delta using the already-calculated new_ln_likelihood1/2
        delta1 = new_ln_likelihood1 - ln_likelihood1
        delta2 = new_ln_likelihood2 - ln_likelihood2

        # Handle overflow-safe acceptance probabilities
        prob_accept1 = 1.0 if delta1 > 0 else np.exp(delta1)
        prob_accept2 = 1.0 if delta2 > 0 else np.exp(delta2)

        if np.random.rand() < prob_accept1:
            order1 = new_order1
            ln_likelihood1 = new_ln_likelihood1
            order1_dict = new_order1_dict 
            acceptance_count1 += 1

        if np.random.rand() < prob_accept2:
            order2 = new_order2
            ln_likelihood2 = new_ln_likelihood2
            order2_dict = new_order2_dict 
            acceptance_count2 += 1

        all_orders['order1'].append(order1_dict.copy())
        all_orders['order2'].append(order2_dict.copy())
        participant_subtype_assignment_history.append(participant_subtype_assignment.copy())
        
        # Log progress
        if (iteration + 1) % max(10, iterations // 10) == 0:
            acceptance_ratio1 = 100 * acceptance_count1 / (iteration + 1)
            acceptance_ratio2 = 100 * acceptance_count2 / (iteration + 1)
            logging.info(
                f"Iteration {iteration + 1}/{iterations}, "
                f"Acceptance Ratio 1: {acceptance_ratio1:.2f}%, "
                f"Log Likelihood 1: {ln_likelihood1:.4f}, "
                f"Current Accepted Order1: {order1_dict}, "
                f"Acceptance Ratio 2: {acceptance_ratio2:.2f}%, "
                f"Log Likelihood 2: {ln_likelihood2:.4f}, "
                f"Current Accepted Order2: {order2_dict}"
            )

    return all_orders, log_likelihoods, None