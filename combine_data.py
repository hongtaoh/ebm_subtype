import pandas as pd 

if __name__=='__main__':
    df1 = pd.read_csv('./data/50|100_0_v1.csv')
    df2 = pd.read_csv('./data/50|100_0_v2.csv')
    
    # Adjust participant IDs in df2 to avoid overlap
    max_participant_id = df1['participant'].max() + 1  # Get the next available ID
    df2['participant'] = df2['participant'] + max_participant_id

    combined_df = pd.concat([df1, df2], ignore_index = True)
    
    combined_df = combined_df.sort_values(by=['participant','biomarker'])

    combined_df.to_csv('data/data.csv', index=False)