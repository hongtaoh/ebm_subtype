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

def metropolis_hastings_subtype_conjugate_priors(
    data_we_have: pd.DataFrame,
    iterations: int,
    n_shuffle: int,
    real_theta_phi: Dict[str, Dict[str, float]],
    upper_limit: float, 
    flip_proportion: int,
) -> Tuple[List[Dict[str, Dict[str, int]]], List[Dict[str, float]], List[Dict[str, int]]]:
    """
    Perform Gibbs sampling to estimate subtype assignments by iteratively updating each participant's assignment.
    """
    # Initialization (same as before)
    n_participants = len(data_we_have.participant.unique())
    biomarkers = data_we_have.biomarker.unique()
    n_stages = len(biomarkers) + 1
    diseased_stages = np.arange(1, n_stages)
    non_diseased_ids = data_we_have.loc[data_we_have.diseased == False].participant.unique()
    diseased_participants = [p for p in range(n_participants) if p not in non_diseased_ids]
    theta_phi_estimates = real_theta_phi
    subtype_ground_truth = {i: 1 if i < n_participants//2 else 2 for i in range(n_participants)}

    # Precompute likelihoods for all participants under both orders
    order1 = np.arange(1, 11)
    order1_dict = dict(zip(biomarkers, order1))
    order2 = np.array([9,7,3,10,8,6,1,5,4,2])
    order2_dict = dict(zip(biomarkers, order2))

    # Preprocess all data once under each order
    full_data1 = sk.preprocess_participant_data(data_we_have, order1_dict)
    full_data2 = sk.preprocess_participant_data(data_we_have, order2_dict)

    # Compute log-likelihoods for each participant under both orders
    ln_likelihoods1, _ = per_participant_compute_ln_likelihood_and_stage_likelihoods(
        full_data1, non_diseased_ids, theta_phi_estimates, diseased_stages
    )
    ln_likelihoods2, _ = per_participant_compute_ln_likelihood_and_stage_likelihoods(
        full_data2, non_diseased_ids, theta_phi_estimates, diseased_stages
    )

    # Initialize assignments based on softmax probabilities
    current_subtype_assignment = {}
    for p in range(n_participants):
        if p in non_diseased_ids:
            continue  # Non-diseased participants are irrelevant
        ll1 = ln_likelihoods1[p]
        ll2 = ln_likelihoods2[p]
        max_ll = max(ll1, ll2)
        prob_subtype1 = np.exp(ll1 - max_ll) / (np.exp(ll1 - max_ll) + np.exp(ll2 - max_ll))
        current_subtype_assignment[p] = 1 if np.random.rand() < prob_subtype1 else 2

    subtype_assignment_history = [current_subtype_assignment.copy()]
    accuracies = [run.compute_subtype_accuracy(current_subtype_assignment, subtype_ground_truth)]

    for iteration in range(iterations):
        # Shuffle participants to update in random order
        participants = list(current_subtype_assignment.keys())
        np.random.shuffle(participants)

        for p in participants:
            # Compute likelihoods for the current participant
            ll1 = ln_likelihoods1[p]
            ll2 = ln_likelihoods2[p]

            # Compute probabilities using softmax
            max_ll = max(ll1, ll2)
            prob_subtype1 = np.exp(ll1 - max_ll) / (np.exp(ll1 - max_ll) + np.exp(ll2 - max_ll))

            # Sample new assignment
            new_assignment = 1 if np.random.rand() < prob_subtype1 else 2
            current_subtype_assignment[p] = new_assignment

        # Record accuracy and history
        current_accuracy = run.compute_subtype_accuracy(current_subtype_assignment, subtype_ground_truth)
        accuracies.append(current_accuracy)
        subtype_assignment_history.append(current_subtype_assignment.copy())

        # Log progress
        if (iteration + 1) % max(10, iterations // 10) == 0:
            logging.info(
                f"Iteration {iteration + 1}/{iterations}, "
                f"Current Accuracy: {current_accuracy:.2f}"
            )
    return None, None, subtype_assignment_history