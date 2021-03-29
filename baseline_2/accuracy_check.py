import os
import argparse
import system_run_v1 as main_module


import seaborn as sns
from matplotlib import pyplot as plt
import time
from pathlib import Path
CLASSIFIER_TYPE =None

def plot_figure(df1):
    global DIR
    global feedback_batch_size
    global top_K_count
    global figure_save_dir
    
    ts = str(time.time()).split('.')[0]
    plt.figure(figsize=[6, 4])
    plt.title('Accuracy in next {} samples| Iteration(batch) : {} samples'.format(top_K_count, feedback_batch_size))
    plt.xlabel('Batch index', fontsize=14)
    plt.ylabel('Precision'.format(top_K_count), fontsize=14)
    sns.lineplot(data=df1, x="idx", y="precision", markers=True, label='Input provided')
    plt.legend(fontsize=14)
    plt.grid()
    plt.savefig(os.path.join(figure_save_dir, '{}_results_{}_{}.png'.format(DIR, feedback_batch_size, top_K_count)))
    try:
        plt.show()
    except:
        pass
    plt.close()


def checkPerformance():
    global figure_save_dir, DIR, CLASSIFIER_TYPE 
    num_runs = 10
    df1 = None
    global CLASSIFIER_TYPE
    for n in range(num_runs):
        results_with_input = main_module.main_executor()
        if df1 is None:
            df1 = results_with_input
        else:
            df1 = df1.append(results_with_input, ignore_index=True)
    # save output
    
    df1.to_csv(os.path.join(figure_save_dir,CLASSIFIER_TYPE , DIR,'results.csv'),index=None)
    plot_figure(df1)
    return

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR',
    choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5', 'us_import6'],
    default=None
)

parser.add_argument(
    '--clf',
    type=str,
    default='DT',
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

# ----------------------------------------
args = parser.parse_args()
DIR = args.DIR
main_module.DIR = args.DIR
CLASSIFIER_TYPE = 'SGC'
feedback_batch_size = args.feedback_size
main_module.feedback_batch_size = args.feedback_size
top_K_count = args.top_K
main_module.top_K_count = top_K_count 
main_module.setup_config(main_module.DIR)
figure_save_dir = 'classifier_op'

path_obj = Path(figure_save_dir+'/'+ CLASSIFIER_TYPE)
path_obj.mkdir(exist_ok=True,parents=True)
path_obj = Path(figure_save_dir+'/'+ CLASSIFIER_TYPE + '/' + DIR)
path_obj.mkdir(exist_ok=True,parents=True)
main_module.CLASSIFIER_TYPE=CLASSIFIER_TYPE

checkPerformance()
