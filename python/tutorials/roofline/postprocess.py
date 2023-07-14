import os
import numpy as np
import pandas as pd
import argparse
from roofline import roofline

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name', type=str, default='label')
    argparser.add_argument('--dir', type=str, default='.')
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

        dfmetric['CC FLOPs'] = 2 * dfmetric['sm__sass_thread_inst_executed_op_dfma_pred_on.sum'] \
            + dfmetric['sm__sass_thread_inst_executed_op_dmul_pred_on.sum'] \
            + dfmetric['sm__sass_thread_inst_executed_op_dadd_pred_on.sum'] \
            + 2 * dfmetric['sm__sass_thread_inst_executed_op_ffma_pred_on.sum'] \
            + dfmetric['sm__sass_thread_inst_executed_op_fmul_pred_on.sum'] \
            + dfmetric['sm__sass_thread_inst_executed_op_fadd_pred_on.sum'] \
            + 2 * dfmetric['sm__sass_thread_inst_executed_op_hfma_pred_on.sum'] \
            + dfmetric['sm__sass_thread_inst_executed_op_hmul_pred_on.sum'] \
            + dfmetric['sm__sass_thread_inst_executed_op_hadd_pred_on.sum']

        dfmetric['TC FLOPs'] = 4096 * \
            dfmetric['sm__inst_executed_pipe_tensor.sum']
        dfmetric['all FLOPs'] = dfmetric['CC FLOPs'] + dfmetric['TC FLOPs']

        dfmetric['AI HBM'] = dfmetric['all FLOPs'].div(
            dfmetric['dram__bytes.sum'])
        dfmetric['AI L2'] = dfmetric['all FLOPs'].div(
            dfmetric['lts__t_bytes.sum'])
        dfmetric['AI L1'] = dfmetric['all FLOPs'].div(
            dfmetric['l1tex__t_bytes.sum'])

        dfmetric['GFLOP/s'] = dfmetric['all FLOPs'] / \
            dfmetric['Time'] / 1024/1024/1024
        dfmetric['TC GFLOP/s'] = dfmetric['TC FLOPs'] / \
            dfmetric['Time'] / 1024/1024/1024
        
        dfmetric['L1 peak'] = dfmetric['l1tex__t_bytes.sum.peak_sustained'] * dfmetric['l1tex__cycles_elapsed.avg.per_second']
        dfmetric['L2 peak'] = dfmetric['lts__t_bytes.sum.peak_sustained'] * dfmetric['lts__cycles_elapsed.avg.per_second']
        dfmetric['DRAM peak'] = dfmetric['dram__bytes.sum.peak_sustained'] * dfmetric['dram__cycles_elapsed.avg.per_second']
        dfmetric['Performance peak'] = dfmetric['sm__inst_executed_pipe_tensor.sum.peak_sustained'] * dfmetric['sm__cycles_elapsed.avg.per_second'] * 4096

        dfs[tag] = dfmetric


    tags = dfs.keys()
    for idx, tag in enumerate(tags):
        dfm = dfs[tag]
        filename = args.name
        AIL1 = dfm['AI L1'].tolist()
        AIL2 = dfm['AI L2'].tolist()
        AIHBM = dfm['AI HBM'].tolist()
        FLOPS = dfm['GFLOP/s'].tolist()
        roofline(idx, tag, filename, FLOPS, AIHBM, AIL2, AIL1)
