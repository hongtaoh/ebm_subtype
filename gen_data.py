from alabEBM import generate, get_params_path
from scipy.stats import kendalltau


# Get path to default parameters
params_file = get_params_path()

biomarker_order_V1 = {
    'AB': 1, 'ADAS': 2, 'AVLT-Sum': 3, 'FUS-FCI': 4, 'FUS-GMI': 5, 
    'HIP-FCI': 6, 'HIP-GMI': 7, 'MMSE': 8, 'P-Tau': 9, 'PCC-FCI': 10
}

biomarker_order_V2 = {
    'AB': 10, 'ADAS': 9, 'AVLT-Sum': 8, 'FUS-FCI': 7, 'FUS-GMI': 6, 
    'HIP-FCI': 5, 'HIP-GMI': 4, 'MMSE': 3, 'P-Tau': 2, 'PCC-FCI': 1
}

tau, p_value = kendalltau(
    list(biomarker_order_V1.values()),
    list(biomarker_order_V2.values())
)

print(f'Tau value is {tau}.')


generate(
    biomarker_order = biomarker_order_V1,
    real_theta_phi_file=params_file,  # Use default parameters
    js = [100], # Number of participants
    rs = [0.5], # Percentage of non-diseased participants
    num_of_datasets_per_combination=1,
    output_dir='data',
    seed = None,
    prefix = None,
    suffix = 'v1'
)

generate(
    biomarker_order = biomarker_order_V2,
    real_theta_phi_file=params_file,  # Use default parameters
    js = [100], # Number of participants
    rs = [0.5], # Percentage of non-diseased participants
    num_of_datasets_per_combination=1,
    output_dir='data',
    seed = None,
    prefix = None,
    suffix = 'v2'
)

