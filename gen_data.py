from alabEBM import generate, get_params_path

# Get path to default parameters
params_file = get_params_path()

biomarker_order_V1: {
    'AB': 3, 'ADAS': 6, 'AVLT-Sum': 8, 'FUS-FCI': 10, 'FUS-GMI': 9, 
    'HIP-FCI': 1, 'HIP-GMI': 7, 'MMSE': 5, 'P-Tau': 4, 'PCC-FCI': 2}
biomarker_order_V2: {
    'AB': 10, 'ADAS': 5, 'AVLT-Sum': 1, 'FUS-FCI': 8, 'FUS-GMI': 9, 
    'HIP-FCI': 4, 'HIP-GMI': 3, 'MMSE': 2, 'P-Tau': 7, 'PCC-FCI': 6}

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

