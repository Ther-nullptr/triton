import pandas as pd
import os
import re

parameters = {
        'B': 1,
        'M': 2048,
        'N': 2048,
        'K': 2048,
        'BM': 128,
        'BN': 128,
        'BK': 32,
        'GM': 8,
        'NS': 5,
        'NW': None
    }


def generate_regex(string, param_list):
    for param_index in param_list:
        param = parameters[param_index]
        if param is None:
            string += f"([0-9]+)_"
        else:
            string += f"{param}_"
    
    return string


if __name__ == '__main__':
    csv_dir = '/home/yujin/workspace/triton/python/tutorials/csv'

    # show all the *.csv files in the csv_dir
    csv_files = os.listdir(csv_dir)
    print(f'csv_files: {len(csv_files)}')
    
    # use regular expression to filter the files
    # B_M_N_K_BM_BN_BK_GM_NS_NW
    
    regex = 'nsight-compute-03-2-batched-matrix-multiplication-ncu-profiling-'

    regex = generate_regex(regex, ['B', 'M', 'N', 'K'])

    regex = regex.rstrip('_')
    regex += '-'
    
    regex = generate_regex(regex, ['BM', 'BN', 'BK', 'GM'])

    regex = regex.rstrip('_')
    regex += '-'

    regex = generate_regex(regex, ['NS', 'NW'])
    regex = regex.rstrip('_') + '.csv'

    print(regex)

    # filter the files
    matched_csv_files = sorted([file for file in csv_files if re.match(regex, file)])

    print(f'matched_csv_files: {len(matched_csv_files)}')
    print(matched_csv_files)

    # analyse the csv
    for i, csv_file in enumerate(matched_csv_files):
        # extract the parameters from the csv_file, use regular expression
        # B_M_N_K_BM_BN_BK_GM_NS_NW

        _, _, B, M, N, K, BM, BN, BK, GM, NS, NW = re.findall(r'\d+', csv_file)
        print(f'B: {B}, M: {M}, N: {N}, K: {K}, BM: {BM}, BN: {BN}, BK: {BK}, GM: {GM}, NS: {NS}, NW: {NW}')

        df = pd.read_csv(os.path.join(csv_dir, csv_file))

        if i == 0:
            print(f'Kernel Name: ', df[df.ID==3]['Kernel Name'].unique())
            print(df[df.ID==3])

        print(df[df.ID==2])
