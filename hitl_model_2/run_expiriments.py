import os
import argparse
# import system_run_v2 as main_module
from matplotlib import pyplot as plt
import time
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing as mp
import time
from multiprocessing import Pool
import system_run_v3 as main_module
from time import time

def getTimeStamp():
    return str(time()).split('.')[0]

def executor(_idx):
    results_with_input, results_no_input = main_module.main_executor()
    results_with_input['run_index'] = _idx
    results_no_input['run_index'] = _idx
    return (results_with_input, results_no_input)

def checkPerformance(num_runs):
    global DIR
#     with Pool(processes=num_runs) as pool:
#         process_list = [pool.apply_async(executor, (_idx,)) for _idx in range(10)]
#         results = [_process.get() for _process in process_list]
    results = []
    for n in range(num_runs):
        r = executor (n)
        results.append(r)
    
    # Join the result dataframes
    df_wI = None
    df_nI = None

    for r_pair in results:
        _df_wI = r_pair[0]
        _df_nI = r_pair[1]
        if df_wI is None:
            df_wI = _df_wI
        else:
            df_wI = df_wI.append(_df_wI,ignore_index=True)
        if df_nI is None:
            df_nI = _df_wI
        else:
            df_nI = df_nI.append(_df_nI,ignore_index=True)

    # Save files
    save_dir = 'Output'
    pathobj = Path(save_dir)
    pathobj.mkdir(exist_ok=True,parents=True)
    save_dir = os.path.join(save_dir,DIR)
    pathobj = Path(save_dir)
    pathobj.mkdir(exist_ok=True, parents=True)
    ts = getTimeStamp()
    op_fname = 'results_withInput{}.csv'.format(ts)
    op_fpath = os.path.join(save_dir, op_fname)
    df_wI.to_csv(op_fpath, index=False)
    op_fname = 'results_noInput{}.csv'.format(ts)
    op_fpath = os.path.join(save_dir, op_fname)
    df_nI.to_csv(op_fpath, index=False)
    return


parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR',
    choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5', 'us_import6'],
    default=None
)
parser.add_argument(
    '--feedback_size',
    type=int,
    default=10
)

parser.add_argument(
    '--num_runs',
    type=int,
    default=10
)

args = parser.parse_args()
DIR = args.DIR
main_module.DIR = args.DIR
feedback_batch_size = args.feedback_size
main_module.feedback_batch_size = args.feedback_size
num_runs = args.num_runs

main_module.setup_config(main_module.DIR)
figure_save_dir = 'accuracy_check'
path_obj = Path(figure_save_dir)
path_obj.mkdir(exist_ok=True,parents=True)
checkPerformance(num_runs)

