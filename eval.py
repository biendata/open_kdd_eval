import os, argparse
import numpy as np

from Rouge import *


parser = argparse.ArgumentParser(description='Rouge-L Score Evaluate')

parser.add_argument('-hp', '--hypothesis-path', type=str, required=True, default=None, help='the hypothesis title path')
parser.add_argument('-rf', '--reference-path', type=str, required=True, default=None, help='the reference title path')
parser.add_argument('-s', '--save-dir', type=str, default='./', help='the result save path')
args = parser.parse_args()

hypo_path = args.hypothesis_path
refer_path = args.reference_path

if not os.path.isdir(hypo_path) or not os.path.isdir(refer_path):
    raise ValueError("Wrong Path")

# load data
hypo_files = os.listdir(hypo_path)

hypos = []
refers = []

for file in hypo_files:
    hypo_f = open(hypo_path+'/'+file, 'r')
    hypos.append(hypo_f.readline().strip())
    if not os.path.isfile(refer_path+'/'+file):
        raise ValueError("file {} not exit in reference file.".format(file))
    refer_f = open(refer_path+'/'+file, 'r')
    tmp = []
    for line in refer_f:
        tmp.append(line.strip())
    refers.append(tmp)

sample_count = len(hypos)

# calculate rouge score
max_rouge_l = 0
mean_rouge_l = 0
for idx, hypo in enumerate(hypos):
    f, p, r = [[] for i in range(3)]
    for refer in refers[idx]:
        _f, _p, _r = rouge_l_sentence_level(hypo, refer)
        f.append(_f)
        p.append(_p)
        r.append(_r)
    max_rouge_l += np.max(f)
    mean_rouge_l += np.mean(f)

result_f = open(args.save_dir+'/result.log', 'w')
print('Rouge-L-Max:', max_rouge_l/sample_count, 'Rouge-L-Mean:', mean_rouge_l/sample_count)
result_f.write('Rouge-L-Max:' + str(max_rouge_l/sample_count) + ' Rouge-L-Mean:' + str(mean_rouge_l/sample_count))
result_f.close()