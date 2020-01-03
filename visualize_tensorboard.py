from tensorboardX import SummaryWriter
import torch
import json
import os


if __name__ == "__main__":
    data_path = '../synthetic/syn_subtype_yx'
    ckpt_path = 'ckpt/1210-gen/39.pth'
    feature_path = 'ckpt/1210-gen/feature_dict.json'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    embedding_matrix = ckpt['embedding.weight'].numpy()
    feature = json.load(open(feature_path))
    patient_mapping_fn = 'ckpt/1210-gen/patient_mapping.json'
    if os.path.exists(patient_mapping_fn):
        patients = json.load(open(patient_mapping_fn))
    else:
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
                pid = entry[3]
                if pid not in patients:
                    patients[pid] = [feature[pid], feature[cname]]
                else:
                    continue
        json.dump(patients, open('ckpt/1210-gen/patient_mapping.json', 'w'))
    embeddings = []
    labels = []
    for k, v in patients.items():
        label = '_'.join(k.split('_')[:-1])
        labels.append(label)
        cur_emb = (embedding_matrix[v[0]] + embedding_matrix[v[1]]) / pow(2, 0.5)
        embeddings.append(cur_emb)
        print(k, v)

    writer = SummaryWriter(logdir='log')
    writer.add_embedding(embeddings, metadata=labels, tag=ckpt_path)
    writer.close()
