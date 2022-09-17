from tensorboardX import SummaryWriter
import torch
import json
import sys
import os
import argparse
import numpy
import pickle
import pandas

def parse_arg():
    ''' This Function Parse the Argument '''
    p=argparse.ArgumentParser( description = 'Example: %(prog)s -h', epilog='Library dependency :')
    p.add_argument('--data',type=str,dest="data_path",help="data path")
    p.add_argument('--ckpt',type=str,dest="ckpt_path",nargs="+",help="ckpt path")
    p.add_argument('--ckpt_label',type=str,dest="ckpt_label",nargs="+",help="ckpt label")
    p.add_argument('--anno',type=str,dest="anno",help="patient annotation file")
    p.add_argument('--anno_json',type=str,dest="anno_json",help="annotation json config file")
    p.add_argument('--output',type=str,dest="output",help="output file")
    if len(sys.argv) < 2:
        print(p.print_help())
        exit(1)
    return p.parse_args()

def load_anno(anno_file, config_json, patient_list):
    """
    parse anno file (tsv) with header
    config_json indicates with column is patient id which column is cancer_type (if 'NA' then look for 'cancer_type' in the json), which column is subtype
    """
    config = json.load(open(config_json))
    anno = {}
    col_list = config.get('col_list', None) 
    with open(anno_file, 'r') as fin:
        num = 0
        header = {}
        for line in fin:
            if line.strip() == '':
                continue
            row = line.strip().split('\t')
            if num == 0:
                for idx,name in enumerate(row):
                    header[name] = idx
                num += 1
            else:
                patient_id = row[header[config['patient_id']]]
                anno[patient_id] = {}
                # patient_id
                anno[patient_id]['patient_id'] = patient_id
                # cancer type
                if config.get('cancer_type', None) in ['NA', None]:
                    anno[patient_id]['cancer'] = config.get('cancer_type_default', 'NA')
                else:
                    anno[patient_id]['cancer'] = row[header[config['cancer_type']]]
                # subtype 
                anno[patient_id]['subtype'] = row[header[config['cancer_subtype']]]
                # other clinical info
                if col_list is not None:
                    for k,v in col_list.items():
                        x = row[header[v]]
                        if x == '':
                            x = 'NA'
                        anno[patient_id][k] = x
    for patient_id in patient_list:
        if anno.get(patient_id, None) is None:
            anno[patient_id] = {'cancer': 'NA', 'subtype': 'NA'}
            anno[patient_id]['patient_id'] = patient_id
            if col_list is not None:
                for k,v in col_list.items():
                    anno[patient_id][k] = 'NA'
    return anno

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def to_float(element):
    try:
        return float(element)
    except ValueError:
        return -10.0

def cal_percentile(v, k):
    # add small noise to the data
    for idx, value in enumerate(v):
        if abs(value) < 1e-6:
            v[idx] = numpy.random.rand()*1e-3
    x = numpy.array(v)
    x = x[x>-1]
    y = numpy.percentile(x, range(0, 101, k))
    return y

def float2decile(table):
    x = list(table.keys())[0]
    col_list = list(table[x].keys())
    for col in col_list:
        if is_float(table[x][col]):
            patient_id_list = list(table.keys())
            array = [to_float(table[patient_id][col]) for patient_id in patient_id_list]
            try:
                label = pandas.cut(array, bins = [-100.0] + list(cal_percentile(array, 10)), right = True, labels = ['%s_NA' % (col)] + ["%s_%d" % (col, idx + 1) for idx in range(10)])
            except:
                try:
                    label = pandas.cut(array, bins = [-100.0] + list(cal_percentile(array, 25)), right = True, labels = ['%s_NA' % (col)] + ["%s_%d" % (col, idx + 1) for idx in range(4)])
                except:
                    continue
            assert len(patient_id_list) == len(label)
            for idx,patient_id in enumerate(patient_id_list):
                table[patient_id][col+'_per'] = label[idx]
    return table

def main():
    global args
    args = parse_arg()
    # Load patient information
    patient2row = json.load(open("%s/patient2row.json" % (args.data_path)))
    row2patient = {}
    for k,v in patient2row.items():
        if '_' in k:
            row2patient[v] = '_'.join(k.split('_')[:-1])
        else:
            row2patient[v] = k
    # Load patient annotation
    if args.anno is not None and args.anno_json is not None:
        patient_anno = load_anno(args.anno, args.anno_json, patient2row.keys())
        # for continuous variable in patient_anno further convert it to decile
        patient_anno = float2decile(patient_anno)
        meta_header = list(patient_anno[list(patient_anno.keys())[0]].keys())
        meta = []
        for idx in range(len(row2patient)):
            meta.append([patient_anno[row2patient[idx]].get(col, 'NA') for col in meta_header])
    else:
        patient_anno = None
    # Prepare tensorboard
    writer = SummaryWriter(logdir=args.output)
    # MutSpace
    if args.ckpt_path is not None:
        mutspace_result = []
        for ckpt_path, ckpt_label in zip(args.ckpt_path, args.ckpt_label):
            if not os.path.exists(ckpt_path):
                print("Skip %s" % (ckpt_path), file = sys.stderr)
                continue
            # load check point data
            ckpt = torch.load(ckpt_path, map_location='cpu')
            embedding_matrix = ckpt['embedding.weight'].numpy()
            # load feature 
            feature_path = "%s/feature_dict.json" % ('/'.join(ckpt_path.split('/')[:-1]))
            if os.path.exists(feature_path):
                feature = json.load(open(feature_path))
            else:
                print("Can't find file %s" % (feature_path))
                exit(1)
            # 
            patient_mapping_fn = "%s/patient_mapping.json" % ('/'.join(ckpt_path.split('/')[:-1]))
            if os.path.exists(patient_mapping_fn):
                patients = json.load(open(patient_mapping_fn))
            else:
                print("Can't find file %s" % (patient_mapping_fn), file = sys.stderr)
                exit(1)
            # 
            embeddings = []
            labels = []
            for k, v in patients.items():
                if patient_anno is None:
                    label = '_'.join(k.split('_')[:-1])
                    labels.append(label)
                # e.g, patient embedding = sample embedding + cancer embedding normalized by scale
                #cur_emb = (embedding_matrix[v[-1][0]] + embedding_matrix[v[-1][1]]) / pow(2, 0.5)
                # the following script is important it sum up embeddings of patient's feature into final embeddings of each patient
                N = len(embedding_matrix[v[-1]])
                cur_emb = embedding_matrix[v[-1]].sum(axis = 0) / pow(N, 0.5)
                embeddings.append(cur_emb)
            if patient_anno is None:
                writer.add_embedding(mat = embeddings, metadata = labels, tag = ckpt_label + '.' + ckpt_path)
            else:
                writer.add_embedding(mat = embeddings, metadata = meta, tag = ckpt_label + '.' + ckpt_path, metadata_header = meta_header) 
    writer.close()

if __name__ == "__main__":
    main()
