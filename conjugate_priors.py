import numpy as np 
import pandas as pd 
from alabEBM.utils import data_processing as utils
from typing import List, Dict, Tuple
import logging 
from collections import defaultdict 

# - participant data, depends on S_n, thus should be separate. 
# - theta_phi_estimates, depends on `affected`, which depends on `S_n`, thus should be separete.
# - participant_stages gets updated when looping through participant_data, 
#     which is dependent on S_n, and thus participant stages should be separate as well. 

def estimate_params_exact(
    m0: float, 
    n0: float, 
    s0_sq: float, 
    v0: float, 
    data: np.ndarray
) -> Tuple[float, float]:
    """
    Estimate posterior mean and standard deviation using conjugate priors for a Normal-Inverse Gamma model.

    Args:
        m0 (float): Prior estimate of the mean (μ).
        n0 (float): Strength of the prior belief in m0.
        s0_sq (float): Prior estimate of the variance (σ²).
        v0 (float): Prior degrees of freedom, influencing the certainty of s0_sq.
        data (np.ndarray): Observed data (measurements).

    Returns:
        Tuple[float, float]: Posterior mean (μ) and standard deviation (σ).
    """
    # Data summary
    sample_mean = np.mean(data)
    sample_size = len(data)
    sample_var = np.var(data, ddof=1)  # ddof=1 for unbiased estimator

    # Update hyperparameters for the Normal-Inverse Gamma posterior
    updated_m0 = (n0 * m0 + sample_size * sample_mean) / (n0 + sample_size)
    updated_n0 = n0 + sample_size
    updated_v0 = v0 + sample_size
    updated_s0_sq = (1 / updated_v0) * ((sample_size - 1) * sample_var + v0 * s0_sq +
                                        (n0 * sample_size / updated_n0) * (sample_mean - m0)**2)
    updated_alpha = updated_v0/2
    updated_beta = updated_v0*updated_s0_sq/2

    # Posterior estimates
    mu_posterior_mean = updated_m0
    sigma_squared_posterior_mean = updated_beta/updated_alpha

    mu_estimation = mu_posterior_mean
    std_estimation = np.sqrt(sigma_squared_posterior_mean)

    return mu_estimation, std_estimation

def update_theta_phi_estimates(
    biomarker_data: Dict[str, Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], 
    theta_phi_default: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Update theta (θ) and phi (φ) parameters for all biomarkers using conjugate priors.

    Args:
        biomarker_data (Dict): Dictionary containing biomarker data. Keys are biomarker names, and values
            are tuples of (curr_order, measurements, participants, diseased, affected).
        theta_phi_default (Dict): Default values for theta and phi parameters for each biomarker.

    Returns:
        Dict[str, Dict[str, float]]: Updated theta and phi parameters for each biomarker.

    Notes:
        - If there is only one observation or no observations at all, the function resorts to the default
          values provided in `theta_phi_default`.
        - This situation can occur if, for example, a biomarker indicates a stage of (num_biomarkers),
          but all participants' stages are smaller than that stage. In such cases, the biomarker is not
          affected for any participant, and default values are used.
    """
    updated_params = defaultdict(dict)
    for biomarker, (
        curr_order, measurements, participants, diseased, affected) in biomarker_data.items():
        theta_mean = theta_phi_default[biomarker]['theta_mean']
        theta_std = theta_phi_default[biomarker]['theta_std']
        phi_mean = theta_phi_default[biomarker]['phi_mean']
        phi_std = theta_phi_default[biomarker]['phi_std']

        for affected_bool in [True, False]:
            measurements_of_affected_bool = measurements[affected == affected_bool]
            if len(measurements_of_affected_bool) > 1:
                s0_sq = np.var(measurements_of_affected_bool, ddof=1)
                m0 = np.mean(measurements_of_affected_bool)
                mu_estimate, std_estimate = estimate_params_exact(
                    m0=m0, n0=1, s0_sq=s0_sq, v0=1, data=measurements_of_affected_bool)
                if affected_bool:
                    theta_mean = mu_estimate
                    theta_std = std_estimate
                else:
                    phi_mean = mu_estimate
                    phi_std = std_estimate
            
            updated_params[biomarker] = {
                'theta_mean': theta_mean,
                'theta_std': theta_std,
                'phi_mean': phi_mean,
                'phi_std': phi_std,
            }
    return updated_params

def preprocess_biomarker_data(
    data_we_have: pd.DataFrame, 
    current_order_dict: Dict,
    participant_stages: np.ndarray
) -> Dict[str, Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Preprocess raw participant data into a structured format for biomarker analysis.

    Args:
        data_we_have (pd.DataFrame): Raw participant data.
        current_order_dict (Dict): Mapping of biomarkers to their current order (stages).
        participant_stages (np.ndarray): Array of disease stages for each participant.

    Returns:
        Dict[str, Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]: Preprocessed biomarker data.
            Keys are biomarker names, and values are tuples of (curr_order, measurements, participants, diseased, affected).
    """
    # This modifies data source in-place
    data_we_have['S_n'] = data_we_have['biomarker'].map(current_order_dict)
    participant_stage_dic = dict(
        zip(np.arange(0, len(participant_stages)), participant_stages))
    data_we_have['k_j'] = data_we_have['participant'].map(participant_stage_dic)
    data_we_have['affected'] = data_we_have['k_j'] >= data_we_have['S_n']

    biomarker_data = {}
    for biomarker, bdata in data_we_have.groupby('biomarker'):
        curr_order = current_order_dict[biomarker]
        measurements = bdata['measurement'].values 
        participants = bdata['participant'].values  
        diseased = bdata['diseased'].values
        affected = bdata['affected'].values
        biomarker_data[biomarker] = (curr_order, measurements, participants, diseased, affected)
    return biomarker_data

def calculate_ln_likelihood_per_participant_and_update_participant_stages(
    participant_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    non_diseased_ids: np.ndarray,
    theta_phi: Dict[str, Dict[str, float]],
    diseased_stages: np.ndarray,
    participant_stages: np.ndarray
) -> Dict[int, float]:
    """
    Calculate the likelihood for each participant and update their disease stages.

    Args:
        participant_data (Dict): Dictionary containing participant data. Keys are participant IDs, and values
            are tuples of (measurements, S_n, biomarkers).
        non_diseased_ids (np.ndarray): Array of participant IDs who are non-diseased.
        theta_phi (Dict): Theta and phi parameters for each biomarker.
        diseased_stages (np.ndarray): Array of possible disease stages.
        participant_stages (np.ndarray): Array of current disease stages for each participant.

    Returns:
        float: Total log likelihood across all participants.
    """
    ln_likelihoods = {}
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

            normalized_probs = stage_likelihoods/likelihood_sum
            participant_stages[participant] = np.random.choice(diseased_stages, p=normalized_probs)
            
            ln_likelihood = max_ln_likelihood + np.log(likelihood_sum)
        ln_likelihoods[participant] = ln_likelihood
    return ln_likelihoods

def preprocess_participant_data(
    data_we_have: pd.DataFrame, 
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Preprocess participant data into NumPy arrays for efficient computation.

    Args:
        data_we_have (pd.DataFrame): Raw participant data.

    Returns:
        Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]: A dictionary where keys are participant IDs,
            and values are tuples of (measurements, S_n, biomarkers).
    """

    participant_data = {}
    for participant, pdata in data_we_have.groupby('participant'):
        measurements = pdata['measurement'].values 
        S_n = pdata['S_n'].values 
        biomarkers = pdata['biomarker'].values  
        participant_data[participant] = (measurements, S_n, biomarkers)
    return participant_data

def metropolis_hastings_subtype_conjugate_priors(
    data_we_have: pd.DataFrame,
    iterations: int,
    n_shuffle: int,
    upper_limit: float
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

    theta_phi_default = utils.get_theta_phi_estimates(data_we_have)
    theta_phi_estimates = theta_phi_default.copy()

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
    participantg_order_assignments_history = []

    #initialize participant assignments
    participant_order_assignments = {}
    for p in range(n_participants):
        participant_order_assignments[p] = np.random.choice([1,2])

    #initialize stages based on assignments
    participant_stages1 = np.zeros(n_participants)
    participant_stages2 = np.zeros(n_participants)
    for idx in range(n_participants):
        if idx not in non_diseased_ids:
            if participant_order_assignments[idx] == 1:
                participant_stages1[idx] = np.random.randint(1, len(diseased_stages) + 1)
            else:
                participant_stages2[idx] = np.random.randint(1, len(diseased_stages) + 1)

    for iteration in range(iterations):
        log_likelihoods['order1'].append(ln_likelihood1)
        log_likelihoods['order2'].append(ln_likelihood2)

        new_order1 = order1.copy()
        utils.shuffle_order(new_order1, n_shuffle)
        new_order1_dict = dict(zip(biomarkers, new_order1))

        new_order2 = order2.copy()
        utils.shuffle_order(new_order2, n_shuffle)
        new_order2_dict = dict(zip(biomarkers, new_order2))

        # Shallow copy is enough because no cols in data_we_have contain mutable objects like lists and dicts
        data_we_have1 = data_we_have.copy()
        data_we_have2 = data_we_have.copy()

        #This will add S_n, k_j, and affected cols
        biomarker_data1 = preprocess_biomarker_data(
            data_we_have1, new_order1_dict, participant_stages1)
        biomarker_data2 = preprocess_biomarker_data(
            data_we_have2, new_order2_dict, participant_stages2)

        # Update participant data based on the new order
        # Update data_we_have based on the new order and the updated participant_stages
        participant_data1 = preprocess_participant_data(data_we_have1)
        participant_data2 = preprocess_participant_data(data_we_have2)

        # Update theta and phi parameters for all biomarkers
        # We basically need the original raw data and the updated affected col 
        theta_phi_estimates1 = update_theta_phi_estimates(
            biomarker_data1, 
            theta_phi_default
        ) 
        theta_phi_estimates2 = update_theta_phi_estimates(
            biomarker_data2, 
            theta_phi_default
        ) 

        ln_likelihoods_order1 = calculate_ln_likelihood_per_participant_and_update_participant_stages(
            participant_data1,
            non_diseased_ids,
            theta_phi_estimates1,
            diseased_stages,
            participant_stages1
        )

        ln_likelihoods_order2 = calculate_ln_likelihood_per_participant_and_update_participant_stages(
            participant_data2,
            non_diseased_ids,
            theta_phi_estimates2,
            diseased_stages,
            participant_stages2
        )

        new_assignments = {}
        for p in range(n_participants):
            if ln_likelihoods_order1[p] > ln_likelihoods_order2[p]:
                new_assignments[p] = 1
            else:
                new_assignments[p] = 2

        participantg_order_assignments_history.append(new_assignments)

        new_ln_likelihood1 = sum(ln_likelihoods_order1[p] for p in new_assignments if new_assignments[p] == 1)
        new_ln_likelihood2 = sum(ln_likelihoods_order2[p] for p in new_assignments if new_assignments[p] == 2)

        # if new_ln_likelihood1 + new_ln_likelihood2 > upper_limit:
        #     logging.error('TOTAL LN LIKELIHOOD EXCEEDS THE UPPER LIMIT! SOMETHING MUST BE WRONG!')
        #     raise ValueError('Total log-likelihood exceeds the upper limit! Check for errors in inference or likelihood computation.')

        delta1 = new_ln_likelihood1 - ln_likelihood1
        delta2 = new_ln_likelihood2 - ln_likelihood2

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
            order1 = new_order1
            ln_likelihood1 = new_ln_likelihood1
            order1_dict = new_order1_dict 
            acceptance_count1 += 1

        if np.random.rand() < prob_accept2:
            order2 = new_order2
            ln_likelihood2 = new_ln_likelihood2
            order2_dict = new_order2_dict 
            acceptance_count2 += 1

        all_orders['order1'].append(order1_dict)
        all_orders['order2'].append(order2_dict)
        
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

    return all_orders, log_likelihoods, participantg_order_assignments_history