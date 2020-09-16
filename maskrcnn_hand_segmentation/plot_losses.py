'''

Hand segmentation using Detectron2
Author: Farnaz Nouraei
Email: farnaz_nouraei@brown.edu

Plot Losses for Train and Eval
Dataset: EgoHands 
Model: Mask R-CNN

'''


import json
import matplotlib.pyplot as plt
import sys,getopt

''' get user args for file name index '''

try:
    opts, args = getopt.getopt(sys.argv[1:],"h",
    ["index=",])
except getopt.GetoptError as err:
    print('option not recognized!')  
    sys.exit(2)
output = None

for o, a in opts:
    if o == "-h":
        print(
        'Please follow the format: python filename --index x'
        )
        sys.exit(2)

    elif o in ("--index"):
        index = a
    else:
        assert False, "unhandled option"


output_path ='./output/'

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(output_path + 'metrics.json')
plt.figure('losses')
plt.plot(
    [x['iteration'] for x in experiment_metrics], 
    [x['total_loss'] for x in experiment_metrics],'r-')
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
    [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in
    x],'g-')
plt.legend(['total_loss', 'validation_loss'], loc='upper left')


plt.savefig('losses_{}.jpg'.format(index))
