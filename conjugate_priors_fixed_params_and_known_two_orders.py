import numpy as np 
import pandas as pd 
from alabebm.utils import data_processing as utils
from alabebm.algorithms import soft_kmeans_algo as sk 
from alabebm.algorithms import conjugate_priors_algo as cp
from typing import List, Dict, Tuple
import logging 
from collections import defaultdict 
import run 

def flip_random_keys(d: Dict[int, int], k:int) -> None:
    """Randomly select k keys from the dictionary and flip their values"""
    if k > len(d):
        raise ValueError("k cannot be greater than the number of keys!")
    
    keys = np.array(list(d.keys()))
    selected_keys = np.random.choice(keys, k, replace=False)

    for key in selected_keys:
        d[key] = 3 - d[key]

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

def compute_total_ln_likelihood(
    participant_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    non_diseased_ids: np.ndarray,
    theta_phi: Dict[str, Dict[str, float]],
    diseased_stages: np.ndarray
    ) -> float:
    total_ln_likelihood = 0.0 
    for participant, (measurements, S_n, biomarkers) in participant_data.items():
        if participant in non_diseased_ids:
            raise ValueError("Participants are not expected to be non diseased participants!")
        else:
            # Diseased participant (sum over possible stages)
            ln_stage_likelihoods = np.array([
                utils.compute_ln_likelihood(
                    measurements, S_n, biomarkers, k_j = k_j, theta_phi=theta_phi
                ) for k_j in diseased_stages
            ])
            max_ln_likelihood = np.max(ln_stage_likelihoods)
            stage_likelihoods = np.exp(ln_stage_likelihoods - max_ln_likelihood)
            likelihood_sum = np.sum(stage_likelihoods)
            # Proof: https://hongtaoh.com/en/2024/12/14/log-sum-exp/
            ln_likelihood = max_ln_likelihood + np.log(likelihood_sum)
        total_ln_likelihood += ln_likelihood
    return total_ln_likelihood

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

def flip_uncertain_keys(assignments: Dict[int, int], ll1: Dict[int, float], ll2: Dict[int, float], k: int) -> None:
    """Flip assignments for participants with the smallest likelihood differences."""
    confidence = {p: abs(ll1[p] - ll2[p]) for p in assignments}
    uncertain = sorted(confidence.keys(), key=lambda x: confidence[x])[:k]
    for p in uncertain:
        assignments[p] = 3 - assignments[p]

def metropolis_hastings_subtype_conjugate_priors(
    data_we_have: pd.DataFrame,
    iterations: int,
    n_shuffle: int,
    real_theta_phi: Dict[str, Dict[str, float]],
    upper_limit: float, 
    flip_proportion: int,
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
    diseased_participants = [x for x in range(n_participants) if x not in non_diseased_ids]
    theta_phi_estimates = real_theta_phi
    subtype_ground_truth = {i: 1 if i < n_participants//2 else 2 for i in range(n_participants)}
    flip_k = int(flip_proportion * len(diseased_participants))

    acceptance_count = 0 

    """Two known subtype orders"""
    # initialize an ordering and likelihood
    order1 = np.arange(1,11)
    order1_dict = dict(zip(biomarkers, order1))

    # initialize an ordering and likelihood
    order2 = np.arange(10, 0)
    # order2 = np.array([9,7,3,10,8,6,1,5,4,2])
    order2_dict = dict(zip(biomarkers, order2))

    subtype_assignment_history = []

    full_data1 = sk.preprocess_participant_data(data_we_have, order1_dict)
    full_data2 = sk.preprocess_participant_data(data_we_have, order2_dict)

    ln_likelihoods1, _ = per_participant_compute_ln_likelihood_and_stage_likelihoods(
        full_data1,
        non_diseased_ids,
        theta_phi_estimates,
        diseased_stages,
    )

    ln_likelihoods2,_ = per_participant_compute_ln_likelihood_and_stage_likelihoods(
        full_data2,
        non_diseased_ids,
        theta_phi_estimates,
        diseased_stages,
    )

    # Only for diseased participants
    current_subtype_assignment = {} # participant (int): assignment (int)
    current_ln_likelihood = 0
    # # Calculate confidence scores for each assignment
    # confidence_scores = {}
    for p in range(n_participants):
        if p in non_diseased_ids:
            continue
        ll1 = ln_likelihoods1[p]
        ll2 = ln_likelihoods2[p]
        # Numerically stable softmax (CORRECT for Gibbs sampling)
        max_ll = max(ll1, ll2)
        prob_subtype1 = np.exp(ll1 - max_ll) / (np.exp(ll1 - max_ll) + np.exp(ll2 - max_ll))
        # Sample from distribution
        if np.random.rand() < prob_subtype1:
            current_subtype_assignment[p] = 1
            current_ln_likelihood += ll1
        else:
            current_subtype_assignment[p] = 2
            current_ln_likelihood += ll2 

    subtype_accuracy = run.compute_subtype_accuracy(current_subtype_assignment, subtype_ground_truth)
    logging.info(f"Initial Subtype Accuracy: {subtype_accuracy}, ")

    for iteration in range(iterations):
        new_assignment = current_subtype_assignment.copy()
        flip_random_keys(new_assignment, flip_k)
        # flip_uncertain_keys(new_assignment, ln_likelihoods1, ln_likelihoods2, flip_k)
        # flip_uncertain_and_random(new_assignment, ln_likelihoods1, ln_likelihoods2, flip_k)
        # Compute new likelihood using precomputed values
        new_ln_likelihood = sum(
            ln_likelihoods1[p] if s == 1 else ln_likelihoods2[p]
            for p, s in new_assignment.items()
        )
        delta = new_ln_likelihood - current_ln_likelihood
        prob_accept = 1.0 if delta > 0 else np.exp(delta)
        
        if np.random.rand() < prob_accept:
            current_subtype_assignment = new_assignment
            current_ln_likelihood = new_ln_likelihood
            acceptance_count += 1
            
        subtype_assignment_history.append(current_subtype_assignment.copy())
        subtype_accuracy = run.compute_subtype_accuracy(current_subtype_assignment, subtype_ground_truth)

        # Log progress
        if (iteration + 1) % max(10, iterations // 10) == 0:
            acceptance_ratio = 100 * acceptance_count / (iteration + 1)
            logging.info(
                f"Iteration {iteration + 1}/{iterations}, "
                f"Acceptance Ratio: {acceptance_ratio:.2f}%, "
                f"Log Likelihood: {current_ln_likelihood:.4f}, "
                f"Current Subtype Accuracy: {subtype_accuracy}, "
            )
    return None, None, subtype_assignment_history