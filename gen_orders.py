"""
To get orders with different target tau values. 
"""

import numpy as np 
from scipy.stats import kendalltau
from typing import List, Dict, Tuple
import json 

def calculate_kendall_tau(order1:List[int], order2:List[int]) -> float:
    tau, _ = kendalltau(order1, order2)
    return tau 

def shuffle_order(arr: np.ndarray, n_shuffle: int) -> None:

    """
    Randomly shuffle a specified number of elements in an array.

    Args:
    arr (np.ndarray): The array to shuffle elements in.
    n_shuffle (int): The number of elements to shuffle within the array.
    """
    # Validate input 
    if n_shuffle <= 1:
        raise ValueError("n_shuffle must be >= 2 or =0")
    if n_shuffle > len(arr):
        raise ValueError("n_shuffle cannot exceed array length")
    if n_shuffle == 0:
        return 

    # Select indices and extract elements
    indices = np.random.choice(len(arr), size=n_shuffle, replace=False)
    original_indices = indices.copy()
    
    while True:
        shuffled_indices = np.random.permutation(original_indices)
        # Full derangement: make sure no indice stays in its original place
        if not np.any(shuffled_indices == original_indices):
            break 
    arr[indices] = arr[shuffled_indices]

def generate_sequence_with_tau(
    target_tau:float, 
    original_order:List[int],
    max_attempts: int = 20000,
    tolerance: float = 0.025,
    n_shuffle: int = 5
    ) -> Tuple[List[int], float]:
    if target_tau == -1:
        return original_order[::-1], -1
    elif target_tau == 1:
        return original_order.copy(), 1
    else:
        # np.random.permutation won't change origional_order inplace
        current_order = np.random.permutation(original_order).astype(int)
    current_tau = calculate_kendall_tau(original_order, current_order)
    for _ in range(max_attempts):
        new_order = current_order.copy()
        shuffle_order(new_order, n_shuffle)
        new_tau = calculate_kendall_tau(original_order, new_order)
        # if the newly proposed order is closer to the target
        if abs(new_tau - target_tau) < abs(current_tau - target_tau):
            current_order = new_order 
            current_tau = new_tau 
        elif np.random.rand() < 0.01: #add randomness to the loop
            current_order = new_order 
            current_tau = new_tau
        if abs(current_tau - target_tau) < tolerance:
            return current_order, current_tau
    print("No exact match found within tolerance!")
    return current_order, current_tau

if __name__ == "__main__":
    biomarker_order = {
        'AB': 1, 'ADAS': 2, 'AVLT-Sum': 3, 'FUS-FCI': 4, 'FUS-GMI': 5,
        'HIP-FCI': 6, 'HIP-GMI': 7, 'MMSE': 8, 'P-Tau': 9, 'PCC-FCI': 10
    }
    biomarkers = list(biomarker_order.keys())
    original_order = list(biomarker_order.values())
    target_taus = [-1.0, -0.5, 0.0, 0.5, 0.9, 1.0]
    result = {}
    for target_tau in target_taus:
        order, tau = generate_sequence_with_tau(target_tau, original_order)
        result[float(target_tau)] = {
            'order': {key: int(value) for key, value in zip(biomarkers, order)},  # Convert to standard Python int
            'tau': float(tau)  
        }
        # results.append({
        #     'target_tau': float(target_tau), 
        #     'order': {key: int(value) for key, value in zip(biomarkers, order)},  # Convert to standard Python int
        #     'tau': float(tau)  
        # })
    with open('data/orders.json', "w") as f:
        json.dump(result, f, indent=4)