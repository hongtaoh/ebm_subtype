import run
import os
import json 

cwd = os.getcwd()
print("Current Working Directory:", cwd)
data_dir = f"{cwd}/test_data"
data_files = os.listdir(data_dir) 
print(data_files)
for data_file in data_files:
    results = run.run_ebm_subtype(
        data_file = f"{data_dir}/{data_file}",
        # data_file= f"{data_dir}/100|200_0_tau_0.0.csv",
        algorithm = 'conjugate_priors',
        n_iter = 1000,
        n_shuffle = 2,
        burn_in = 500,
        thinning = 20,
    )