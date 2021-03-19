import os
import argparse
import system_run_v3 as main_module
import seaborn as sns
from matplotlib import pyplot as plt
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
figure_save_dir = None
from time import time


def checkPerformance(args):
    num_runs = 1
    df1 = None
   
    prec_col_name = 'precision@{}'.format(args.top_K)
    df1 = pd.DataFrame(columns =['idx',prec_col_name])
    
#     results = Parallel(n_jobs = num_runs)( delayed (main_module.execute_with_input) (args.top_K,args.feedback_size,)  for n in range(num_runs))
    
    for n in range(num_runs):
        
        precision = main_module.execute_with_input(
            check_next = args.top_K,
            batch_size= args.feedback_size
        )
        
        df1 = df1.append(
            pd.DataFrame({
                'idx': np.arange(1,len(precision)+1),
                prec_col_name : precision
            }),
            ignore_index=True
        )
        
    ts = '{}_{}'.format( args.top_K, args.feedback_size) + str(time()).split('.')[0]  
    df1.to_csv(os.path.join(figure_save_dir,'results_{}.csv'.format(ts)),index=None)

    return df1


# ---------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    '--DIR',
    choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5', 'us_import6'],
    default=None
)
parser.add_argument(
    '--feedback_size',
    type=int,
    default=25
)
parser.add_argument(
    '--top_K',
    type=int,
    default=25,
)

args = parser.parse_args()
DIR = args.DIR
main_module.DIR = args.DIR
main_module.setup_config(DIR)

# -----------------------------------
figure_save_dir = 'accuracy_check'+'/' + DIR
path_obj = Path(figure_save_dir)
path_obj.mkdir(exist_ok=True, parents=True)

checkPerformance(args)
