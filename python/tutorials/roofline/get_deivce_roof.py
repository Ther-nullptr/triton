import os
import pandas as pd

if __name__ == '__main__':

    datadir = '.'
    files = [x for x in os.listdir(datadir) if x.endswith(
        '.csv') and x.startswith('output')]
    files.sort()
    files = [os.path.join(datadir, file) for file in files]
    dfs = {}
    for file in files:
        tag, ext = os.path.splitext(os.path.basename(file))
        dfs[tag] = pd.DataFrame()
        with open(file, 'r') as f:
            cnt = 0
            while True:
                ln = f.readline()
                if not ln:
                    break
                cnt += 1
                if 'Host Name' in ln:
                    break
            df = pd.read_csv(file, skiprows=cnt-1)
            df = df[df.ID==2]
            df['Metric Value'] = df['Metric Value'].apply(lambda x: float(x.replace(',', '')))
            dft = df.groupby(['Kernel Name', 'Metric Name']).sum()
            dfmetric = pd.pivot_table(
                dft, index='Kernel Name', columns='Metric Name', values='Metric Value')
            
            dfmetric['L1 peak'] = dfmetric['l1tex__t_bytes.sum.peak_sustained'] * dfmetric['l1tex__cycles_elapsed.avg.per_second']
            dfmetric['L2 peak'] = dfmetric['lts__t_bytes.sum.peak_sustained'] * dfmetric['lts__cycles_elapsed.avg.per_second']
            dfmetric['DRAM peak'] = dfmetric['dram__bytes.sum.peak_sustained'] * dfmetric['dram__cycles_elapsed.avg.per_second']
            dfmetric['Performance peak'] = dfmetric['sm__inst_executed_pipe_tensor.sum.peak_sustained'] * dfmetric['sm__cycles_elapsed.avg.per_second'] * 1024

            dfs[tag] = dfmetric


    tags = dfs.keys()
    flags = ['all']  # 'HBM','L2','L1' or 'all'
    
    for tag in tags:
        for flag in flags:
            dfm = dfs[tag]
            LABELS = dfm.index.tolist()
            L1_Peak = dfm['L1 peak'].tolist()[0]
            L2_Peak = dfm['L2 peak'].tolist()[0]
            DRAM_Peak = dfm['DRAM peak'].tolist()[0]
            Performance_Peak = dfm['Performance peak'].tolist()[0]

            print(f"L1_Peak: {L1_Peak / 10 ** 9} GB/s, L2_Peak: {L2_Peak / 10 ** 9} GB/s, DRAM_Peak: {DRAM_Peak / 10 ** 9} GB/s, Performance_Peak: {Performance_Peak / 10 ** 12} TFLOP/s")

