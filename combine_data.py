import pandas as pd 
import json 

if __name__=='__main__':
    js = [50, 200, 500]
    rs = [0.1, 0.25, 0.5, 0.75, 0.9]
    ms = range(50)
    target_taus = [-1.0, -0.5, 0.0, 0.5, 0.9]
    raw_data_dir = 'data/raw'
    processed_data_dir = 'data/processed'

    for j in js:
        for r in rs:
            comb_str = f'{int(j*r)}|{j}'
            for m in ms:
                raw_str = f"{comb_str}_{m}"
                df1_dir = f"{raw_data_dir}/{raw_str}_original.csv"
                df1 = pd.read_csv(df1_dir)
                for target_tau in target_taus:
                    df2_dir = f"{raw_data_dir}/{raw_str}_tau_{target_tau}.csv"
                    df2 = pd.read_csv(df2_dir)

                    # Adjust participant IDs in df2 to avoid overlap
                    max_participant_id = df1['participant'].max() + 1  # Get the next available ID
                    df2['participant'] = df2['participant'] + max_participant_id

                    combined_df = pd.concat([df1, df2], ignore_index = True)
                    combined_df = combined_df.sort_values(by=['participant','biomarker'])
                    result_path = f"{processed_data_dir}/{raw_str}_tau_{target_tau}.csv"
                    combined_df.to_csv(result_path, index=False)
            print(f"J={j}, R={r} is done!")