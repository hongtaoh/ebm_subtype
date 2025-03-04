import run
import os
import json 

cwd = os.getcwd()
print("Current Working Directory:", cwd)
data_dir = f"{cwd}/test_data"
data_files = os.listdir(data_dir) 

# for data_file in data_files[0]:
results = run.run_ebm_subtype(
    data_file= f"{data_dir}/{data_files[0]}",
    algorithm = 'conjugate_priors',
    n_iter = 5000,
    n_shuffle = 2,
    burn_in = 2000,
    thinning = 20,
)