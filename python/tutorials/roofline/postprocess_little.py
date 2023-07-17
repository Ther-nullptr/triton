import os
import numpy as np
import pandas as pd
import argparse
from roofline_little import roofline_little

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name', type=str, default='label')
    argparser.add_argument('--dir', type=str, default='.')
    argparser.add_argument('--type', type=str, default='sm__inst_executed_pipe_tensor')
    args = argparser.parse_args()

    datadir = args.dir
    files = [x for x in os.listdir(datadir) if x.endswith('.csv')]
    files.sort()
    files = [os.path.join(datadir, file) for file in files]
    dfs = {}
    for file in files:
        tag, ext = os.path.splitext(os.path.basename(file))
        dfs[tag] = pd.DataFrame()

        df = pd.read_csv(file)
        df = df[df.ID==2]
        df['Metric Value'] = df['Metric Value'].apply(lambda x: float(x.replace(',', '')))
        dft = df.groupby(['Kernel Name', 'Metric Name']).sum()
        dfmetric = pd.pivot_table(
            dft, index='Kernel Name', columns='Metric Name', values='Metric Value')
        dfmetric['Count'] = df.groupby(['Kernel Name']).count()[
            'ID'].div(dfmetric.shape[1])

        dfmetric['Time'] = dfmetric['sm__cycles_elapsed.avg'] \
            / (dfmetric['sm__cycles_elapsed.avg.per_second'] / dfmetric['Count'])
        
        dfmetric['Peak IPC'] = dfmetric[f'{args.type}.avg.peak_sustained']
        dfmetric['Active Instructions'] = dfmetric[f'{args.type}.max.per_cycle_active']
        # TODO and latency
        dfmetric['Practical IPC'] = dfmetric['Active Instructions'] / dfmetric['Peak IPC']
        dfmetric['Latency'] = dfmetric['Time'] / dfmetric['Active Instructions']

        dfs[tag] = dfmetric


    tags = dfs.keys()
    for idx, tag in enumerate(tags):
        dfm = dfs[tag]
        filename = args.name
        PeakIPC = dfm['Peak IPC'].tolist()
        ActiveInsts = dfm['Active Instructions'].tolist()
        PracticalIPC = dfm['Practical IPC'].tolist()
        Latency = dfm['Latency'].tolist()
        roofline_little(idx, tag, filename, PeakIPC, ActiveInsts, PracticalIPC, Latency)
