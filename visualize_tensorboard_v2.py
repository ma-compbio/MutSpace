from tensorboardX import SummaryWriter
import torch
import json
import sys
import os
import argparse
import numpy

def parse_arg():
    ''' This Function Parse the Argument '''
    p=argparse.ArgumentParser( description = 'Example: %(prog)s -h', epilog='Library dependency :')
    p.add_argument('--data',type=str,dest="data_path",help="data path")
    p.add_argument('--ckpt',type=str,dest="ckpt_path",help="ckpt path")
    p.add_argument('--feature',type=str,dest="feature",help="feature json path")
    p.add_argument('--patient',type=str,dest="patient",help="patient mapping json path")
    p.add_argument('--output',type=str,dest="output",help="output file")
    if len(sys.argv) < 2:
        print(p.print_help())
        exit(1)
    return p.parse_args()

def main():
    global args
    args = parse_arg()
    data_path = args.data_path
    ckpt_path = args.ckpt_path
    feature_path = args.feature
    ckpt = torch.load(ckpt_path, map_location='cpu')
    embedding_matrix = ckpt['embedding.weight'].numpy()
    feature = json.load(open(feature_path))
    patient_mapping_fn = args.patient
    #if os.path.exists(patient_mapping_fn):
    #    patients = json.load(open(patient_mapping_fn))
    #else:
    if True:
        datalist = os.listdir(data_path)
        datalist = [".".join(name.split('.')[:-1]) for name in datalist if name.endswith('tsv')]
        patients = dict()
        print(datalist)
        for cname in datalist:
            path = os.path.join(data_path, f'{cname}.tsv')
            for i, line in enumerate(open(path).readlines()):
                if i == 0:
                    continue
                entry = line.strip().split('\t')
                pid = '_'.join(entry[3].split('_')[:-1])
                if pid not in patients:
                    patients[pid] = [[feature[pid], feature[cname]]]
                else:
                    continue
        #json.dump(patients, open(args.patient, 'w'))
    embeddings = []
    labels = []
    for k, v in patients.items():
        label = '_'.join(k.split('_')[:-1])
        labels.append(label)
        cur_emb = (embedding_matrix[v[-1][0]] + embedding_matrix[v[-1][1]]) / pow(2, 0.5)
        embeddings.append(cur_emb)

    writer = SummaryWriter(logdir=args.output)
    writer.add_embedding(mat = embeddings, metadata=labels, tag=ckpt_path)
    writer.close()

if __name__ == "__main__":
    main()
