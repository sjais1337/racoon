import pandas as pd
import numpy as np
import argparse
import os

def split_csv_into_three(name):
    df = pd.read_csv(f"base/{name}.csv")
    print(f"Successfully read '{name}' with {len(df)} rows.")

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("DataFrame shuffled.")

    df_splits = np.array_split(df_shuffled, 3)
    print(f"DataFrame split into 3 parts of sizes: {[len(part) for part in df_splits]}")

    os.mkdir(f'split/{name}')


    for i, part_df in enumerate(df_splits):
        output_filename = f"part{i+1}.csv"
        output_filepath = os.path.join('split', name, output_filename)
        try:
            part_df.to_csv(output_filepath, index=False)
            print(f"Successfully saved '{output_filepath}' with {len(part_df)} rows.")
        except Exception as e:
            print(f"Error saving file '{output_filepath}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-b", "--basename")
    
    args = parser.parse_args()

    split_csv_into_three(args.basename)