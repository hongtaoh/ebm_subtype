import numpy as np 
import pandas as pd 
from alabebm.utils import data_processing as utils
from alabebm.algorithms import soft_kmeans_algo as sk 
from alabebm.algorithms import conjugate_priors_algo as cp
from typing import List, Dict, Tuple
import logging 
from collections import defaultdict 
import run 

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

def metropolis_hastings_subtype_conjugate_priors(
    data_we_have: pd.DataFrame,
    iterations: int,
    n_shuffle: int,
    real_theta_phi: Dict[str, Dict[str, float]],
) -> Tuple[List[Dict[str, Dict[str, int]]], List[Dict[str, float]], List[Dict[int, int]]]:
    """
    Perform Metropolis-Hastings sampling to jointly estimate biomarker orderings and subtype assignments.
    
    Uses a proper Gibbs sampling approach with alternating updates.
    """
    n_participants = len(data_we_have.participant.unique())
    biomarkers = data_we_have.biomarker.unique()
    n_stages = len(biomarkers) + 1
    diseased_stages = np.arange(start=1, stop=n_stages, step=1)
    non_diseased_ids = data_we_have.loc[data_we_have.diseased == False].participant.unique()
    theta_phi_estimates = real_theta_phi
    subtype_ground_truth = {i: 1 if i < n_participants//2 else 2 for i in range(n_participants)}

    # Initialize orderings
    order1 = np.random.permutation(np.arange(1, n_stages))
    order1_dict = dict(zip(biomarkers, order1))
    
    order2 = np.random.permutation(np.arange(1, n_stages))
    order2_dict = dict(zip(biomarkers, order2))
    
    # Initialize subtype assignments
    subtype_assignment = {}
    ln_likelihood1 = float('-inf')
    ln_likelihood2 = float('-inf')
    # for p in range(n_participants):
    #     if p in non_diseased_ids:
    #         continue
    #     # Random initial assignment
    #     subtype_assignment[p] = np.random.choice([1, 2])
    
    # # Split data based on initial assignments
    # data_subtype1, data_subtype2 = split_data_by_subtype(data_we_have, subtype_assignment)
    
    # # Calculate initial likelihoods
    # participant_data1 = sk.preprocess_participant_data(data_subtype1, order1_dict)
    # participant_data2 = sk.preprocess_participant_data(data_subtype2, order2_dict)
    
    # ln_likelihood1, _ = sk.compute_total_ln_likelihood_and_stage_likelihoods(
    #     participant_data1,
    #     non_diseased_ids,
    #     theta_phi_estimates,
    #     diseased_stages
    # )
    
    # ln_likelihood2, _ = sk.compute_total_ln_likelihood_and_stage_likelihoods(
    #     participant_data2,
    #     non_diseased_ids,
    #     theta_phi_estimates,
    #     diseased_stages
    # )
    
    # Track history
    all_orders = defaultdict(list)
    log_likelihoods = defaultdict(list)
    subtype_assignments_history = []
    
    acceptance_count1 = 0
    acceptance_count2 = 0
    
    for iteration in range(iterations):
        # Record current state
        log_likelihoods['order1'].append(ln_likelihood1)
        log_likelihoods['order2'].append(ln_likelihood2)
        all_orders['order1'].append(order1_dict.copy())
        all_orders['order2'].append(order2_dict.copy())
        subtype_assignments_history.append(subtype_assignment.copy())
        
        ###############################################
        # STEP 1: Update subtype assignments (Gibbs sampling)
        ###############################################
        
        # Calculate likelihoods for all participants under both models
        full_data1 = sk.preprocess_participant_data(data_we_have, order1_dict)
        full_data2 = sk.preprocess_participant_data(data_we_have, order2_dict)
        
        ln_likelihoods1, _ = per_participant_compute_ln_likelihood_and_stage_likelihoods(
            full_data1,
            non_diseased_ids,
            theta_phi_estimates,
            diseased_stages,
        )
        
        ln_likelihoods2, _ = per_participant_compute_ln_likelihood_and_stage_likelihoods(
            full_data2,
            non_diseased_ids,
            theta_phi_estimates,
            diseased_stages,
        )
        
        # Sample new subtype assignments
        new_subtype_assignment = {}
        for p in range(n_participants):
            if p in non_diseased_ids:
                continue
            
            ll1 = ln_likelihoods1[p]
            ll2 = ln_likelihoods2[p]
            
            # Numerically stable softmax
            max_ll = max(ll1, ll2)
            prob_subtype1 = np.exp(ll1 - max_ll) / (np.exp(ll1 - max_ll) + np.exp(ll2 - max_ll))
            
            # Sample from distribution
            if np.random.rand() < prob_subtype1:
                new_subtype_assignment[p] = 1
            else:
                new_subtype_assignment[p] = 2
        
        # Update subtype assignment
        subtype_assignment = new_subtype_assignment
        
        # Re-split data with new assignments
        data_subtype1, data_subtype2 = split_data_by_subtype(data_we_have, subtype_assignment)
        
        # Recalculate likelihoods with new assignments
        participant_data1 = sk.preprocess_participant_data(data_subtype1, order1_dict)
        participant_data2 = sk.preprocess_participant_data(data_subtype2, order2_dict)
        
        ln_likelihood1, _ = sk.compute_total_ln_likelihood_and_stage_likelihoods(
            participant_data1,
            non_diseased_ids,
            theta_phi_estimates,
            diseased_stages
        )
        
        ln_likelihood2, _ = sk.compute_total_ln_likelihood_and_stage_likelihoods(
            participant_data2,
            non_diseased_ids,
            theta_phi_estimates,
            diseased_stages
        )
        
        ###############################################
        # STEP 2: Update order1 (Metropolis-Hastings)
        ###############################################
        
        # Propose new order1
        new_order1 = order1.copy()
        utils.shuffle_order(new_order1, n_shuffle)
        new_order1_dict = dict(zip(biomarkers, new_order1))
        
        # Calculate likelihood with new order1 but current assignments
        new_participant_data1 = sk.preprocess_participant_data(data_subtype1, new_order1_dict)
        
        new_ln_likelihood1, _ = sk.compute_total_ln_likelihood_and_stage_likelihoods(
            new_participant_data1,
            non_diseased_ids,
            theta_phi_estimates,
            diseased_stages
        )
        
        # Calculate acceptance probability
        delta1 = new_ln_likelihood1 - ln_likelihood1
        prob_accept1 = 1.0 if delta1 > 0 else np.exp(delta1)
        
        # Accept or reject new order1
        if np.random.rand() < prob_accept1:
            order1 = new_order1
            order1_dict = new_order1_dict
            ln_likelihood1 = new_ln_likelihood1
            acceptance_count1 += 1
        
        ###############################################
        # STEP 3: Update order2 (Metropolis-Hastings)
        ###############################################
        
        # Propose new order2
        new_order2 = order2.copy()
        utils.shuffle_order(new_order2, n_shuffle)
        new_order2_dict = dict(zip(biomarkers, new_order2))
        
        # Calculate likelihood with new order2 but current assignments
        new_participant_data2 = sk.preprocess_participant_data(data_subtype2, new_order2_dict)
        
        new_ln_likelihood2, _ = sk.compute_total_ln_likelihood_and_stage_likelihoods(
            new_participant_data2,
            non_diseased_ids,
            theta_phi_estimates,
            diseased_stages
        )
        
        # Calculate acceptance probability
        delta2 = new_ln_likelihood2 - ln_likelihood2
        prob_accept2 = 1.0 if delta2 > 0 else np.exp(delta2)
        
        # Accept or reject new order2
        if np.random.rand() < prob_accept2:
            order2 = new_order2
            order2_dict = new_order2_dict
            ln_likelihood2 = new_ln_likelihood2
            acceptance_count2 += 1
        
        # Log progress
        if (iteration + 1) % max(10, iterations // 10) == 0:
            acceptance_ratio1 = 100 * acceptance_count1 / (iteration + 1)
            acceptance_ratio2 = 100 * acceptance_count2 / (iteration + 1)
            subtype_accuracy = run.compute_subtype_accuracy(subtype_assignment, subtype_ground_truth)
            
            logging.info(
                f"Iteration {iteration + 1}/{iterations}, "
                f"Acceptance Ratio 1: {acceptance_ratio1:.2f}%, "
                f"Log Likelihood 1: {ln_likelihood1:.4f}, "
                f"Acceptance Ratio 2: {acceptance_ratio2:.2f}%, "
                f"Log Likelihood 2: {ln_likelihood2:.4f}, "
                f"Subtype Accuracy: {subtype_accuracy}"
            )
    
    return all_orders, log_likelihoods, subtype_assignments_history