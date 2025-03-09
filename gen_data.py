"""Generate raw data before combining
"""

from alabebm import generate, get_params_path
from scipy.stats import kendalltau
import json 

# Get path to default parameters
params_file = get_params_path()

original_biomarker_order = {
    'AB': 1, 'ADAS': 2, 'AVLT-Sum': 3, 'FUS-FCI': 4, 'FUS-GMI': 5, 
    'HIP-FCI': 6, 'HIP-GMI': 7, 'MMSE': 8, 'P-Tau': 9, 'PCC-FCI': 10
}

# generate(
#         biomarker_order = original_biomarker_order,
#         real_theta_phi_file=params_file,  # Use default parameters
#         js = [50, 200, 500], # Number of participants
#         rs = [0.1, 0.25, 0.5, 0.75, 0.9], # Percentage of non-diseased participants
#         num_of_datasets_per_combination=50,
#         output_dir='data/raw',
#         seed = None,
#         prefix = None,
#         suffix = 'original'
# )

with open('data/orders.json', 'r') as f:
    results = json.load(f)

for result in results:
    generate(
        biomarker_order = result['order'],
        real_theta_phi_file=params_file,  # Use default parameters
        js = [50, 200, 500], # Number of participants
        rs = [0.1, 0.25, 0.5, 0.75, 0.9], # Percentage of non-diseased participants
        num_of_datasets_per_combination=50,
        output_dir='data/raw',
        seed = None,
        prefix = None,
        suffix = f"tau_{result['target_tau']}"
    )

