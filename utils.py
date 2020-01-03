import json
from os.path import join
import os
from random import sample
import torch
import h5py
from tqdm import tqdm


def get_seq_mapping():
    """convert mutation type to integer (index)"""
    #TODO add support for indel and make it more convenient to modify this function
    mut_mapping = {'C->A': 0, 'C->G': 1, 'C->T': 2, 'T->A': 3, 'T->C': 4, 'T->G': 5}
    nuk_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return mut_mapping, nuk_mapping


def get_rev_seq_mapping():
    """reverse operation for ger_seq_mapping"""
    mut_mapping = {0: 'C->A', 1: 'C->G', 2: 'C->T', 3: 'T->A', 4: 'T->C', 5: 'T->G'}
    nuk_mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    return mut_mapping, nuk_mapping


def index(seq):
    """convert an arbitary mutation pattern, ie the up- and downstream of mutation into an index
    # [up, down, mut]
    # [3, 1, 0, 2, 4] -> 1159
    the last element indicates the nucleotide mapping for the reference base
    the first half (except the last one) is the nucleotides mapping for the upstream bases
    the second half (except the last one) is the nucleotides mapping for the downstream bases
    """
    idx = 0
    for i, s in enumerate(seq):
        idx += (4 ** i) * s
    return idx


def rev_index(num, n_bits):
    # 1159 -> [3, 1, 0, 2, 4]
    # here n_bits = 5
    seq = []
    for i in range(n_bits):
        base = 4 ** (n_bits - i - 1)
        idx = num // base
        num = num - idx * base
        seq.insert(0, idx)
    return seq


class Dataset:
    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.data = []

    def process(self):
        """
        This funtion should process the raw input into a format of
        [[id, a_features, b_featuers]]
        :return:
        """
        return None

    def sample(self, n_items, drop_id=None):
        if drop_id is not None:
            samples = [case for case in sample(self.data, n_items) if case[0] != drop_id]
        else:
            samples = sample(self.data, n_items)
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class MutationDataset:
    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.mut_mapping, self.nuk_mapping = get_seq_mapping()
        self.ring_width = self.config.ring_width
        self.ring_num = self.config.ring_num
        self.feature_dict = dict()
        self.feature_num = (6 * 4 ** (2 * self.config.ring_width)) * self.config.ring_num
        self.data = self.process()
        self.num_fa = len(self.data[-1][1])
        self.num_fb = len(self.data[-1][2])

    def get_feature_id(self, f):
        """
        assign feature index to a new feature
        by default self.feature_num is the number of rings
        note f is a string so that only categorical features are supported
        """
        if f not in self.feature_dict:
            self.feature_dict[f] = self.feature_num
            self.feature_num += 1
        return self.feature_dict[f]

    def process(self):
        data = []
        patient_mapping = dict()
        header = json.load(open(join(self.data_path, 'meta.json')))
        datalist = os.listdir(self.config.data_path)
        datalist = [".".join(name.split('.')[:-1]) for name in datalist if name.endswith('tsv')]
        print(datalist)
        for cname in datalist:
            print(f'Loading {cname}...')
            fin = open(os.path.join(self.data_path, f'{cname}.tsv'), 'r')
            for i, line in enumerate(fin.readlines()):
                line = line.strip().split('\t')
                line.append(cname)
                if i == 0:
                    continue

                # uid is patient id or sample id
                uid = self.get_feature_id(line[header['uid']])
                # a_features: categorical feature for each patient/sample
                # b_features: cetegorical feature for each project/data
                a_features, b_features = [], []

                # convert mutation and its context into index
                rings = self.decompose(line[header['upstream']], line[header['downstream']], line[header['var_type']])
                a_features.extend(rings)

                for col in header['a']:
                    a_features.append(self.get_feature_id(line[col]))

                for col in header['b']:
                    b_features.append(self.get_feature_id(line[col]))

                case = [uid, a_features, b_features]
                data.append(case)

                pname = line[header['uid']]
                if pname not in patient_mapping:
                    patient_mapping[pname] = case

            if self.config.debug:
                break
        json.dump(self.feature_dict, open(join(self.config.ckpt_path, 'feature_dict.json'), 'w'))
        json.dump(patient_mapping, open(join(self.config.ckpt_path, 'patient_mapping.json'), 'w'))
        return data

    def decompose(self, upstream, downstream, mutation):
        rings = []
        w = self.ring_width
        mutation = [self.mut_mapping[mutation]]
        base = 6 * 4 ** (2 * self.ring_width)
        for i in range(self.ring_num):
            if i == 0:
                up = upstream[-w:]
            else:
                up = upstream[-(i + 1) * w:-i * w]
            down = downstream[i * w:(i + 1) * w]
            up = [self.nuk_mapping[s] for s in up]
            down = [self.nuk_mapping[s] for s in down]
            rings.append(index(up + down + mutation) + i * base)
        return rings

    def sample(self, n_items, drop_id=None):
        if drop_id is not None:
            samples = [case for case in sample(self.data, n_items) if case[0] != drop_id]
        else:
            samples = sample(self.data, n_items)
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def save_hdf5(self, path):
        f = h5py.File(path, 'w')
        f['feature_num'] = self.feature_num
        f['num_fa'] = self.num_fa
        f['num_fb'] = self.num_fb
        f['total'] = len(self.data)
        for i, item in enumerate(tqdm(self.data)):
            f.create_group(str(i))
            f[str(i)]['uid'] = item[0]
            f[str(i)]['a_features'] = item[1]
            f[str(i)]['b_features'] = item[2]
        f.close()


class MutationDatasetH5PY:
    def __init__(self, config, data_path, patient_mapping_path):
        self.data = h5py.File(data_path, 'r')
        self.config = config
        self.feature_num = self.data['feature_num'][()]
        self.num_fa = self.data['num_fa'][()]
        self.num_fb = self.data['num_fb'][()]
        self.total = self.data['total'][()]
        self.patient_mapping = json.load(open(patient_mapping_path))

    def __getitem__(self, item):
        uid = self.data[str(item)]['uid'][()]
        fa = list(self.data[str(item)]['a_features'])
        fb = list(self.data[str(item)]['b_features'])
        return [uid, fa, fb]

    def __len__(self):
        return self.total

    def sample(self, n_negative, drop_id):
        keys = sample(self.patient_mapping.keys(), n_negative)
        samples = [self.patient_mapping[k] for k in keys if self.patient_mapping[k][0] != drop_id]
        return samples


class MyCollator:
    def __init__(self, config, dataset):
        self.dataset = dataset
        self.num_fa = dataset.num_fa
        self.num_fb = dataset.num_fb
        self.n_negative = config.n_negative

    def __call__(self, batch):
        N = len(batch)
        pos_a_features = [] 
        pos_b_features = []
        neg_b_features = []
        neg_mask = []
        for i, pos_case in enumerate(batch):
            neg_sample = self.dataset.sample(self.n_negative, drop_id=pos_case[0])
            pos_a_features.append(pos_case[1])
            pos_b_features.append(pos_case[2])
            neg_b_features.append([])
            neg_mask.append([])
            for j, neg_case in enumerate(neg_sample):
                neg_mask[i].append(1)
                neg_b_features[i].append(neg_case[2])
            pad_num = self.n_negative - len(neg_mask[i])
            neg_mask[i].extend([0] * pad_num)
            neg_b_features[i].extend([[0] * self.num_fb] * pad_num)

        return {'pos_a': torch.LongTensor(pos_a_features).cuda(), 
                'pos_b': torch.LongTensor(pos_b_features).cuda(), 
                'neg_b': torch.LongTensor(neg_b_features).cuda(),
                'neg_mask': torch.FloatTensor(neg_mask).cuda()}
