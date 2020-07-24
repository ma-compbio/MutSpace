import sys
import torch
import argparse
from utils import *
from model import MutSpace
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np


def save_settings():
    '''
    Save parameters into a json file
    '''
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    os.makedirs(config.ckpt_path, exist_ok=True)
    setting_fn = os.path.join(config.ckpt_path, f'setting.json')
    json.dump(config.__dict__, open(setting_fn, 'w'))

def parse_arg():
    ''' 
    parse parameters
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="Name or ID of this run. A folder with this name will be created in the directory where this script is being executed")
    parser.add_argument('--data_path', type=str, help="Directory of mutation data")
    parser.add_argument('--ring_num', type=int, default=6, help="Number of sub-components for the sequence context")
    parser.add_argument('--ring_width', type=int, default=1, help="The width of sequence sub-component")
    parser.add_argument('--margin', type=float, default=1.0, help="The constant margin parameter in the hinge-loss function, default is 1.0")
    parser.add_argument('--max_norm', type=float, default=10.0, help="The Max norm of embeddings, default is 10.0")
    parser.add_argument('--emb_dim', type=int, default=200, help="Dimension of embedded vector, default is 200")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs before training is stopped")
    parser.add_argument('--batch_size', type=int, default=4096, help="Batch size, default is 4096")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate, default is 1e-3")
    parser.add_argument('--n_negative', type=int, default=15, help="Number of negative samples per positive sample generated through negative sampling, default is 15")
    #parser.add_argument('--verbose_step', type=int, default=500, help="verbose level")
    parser.add_argument('--debug', action='store_true', help="")
    parser.add_argument('--temp', type=float, default=1.0, help="Normalization parameter for calculation of similarity")
    parser.add_argument('--seed', type=int, default=888, help="Random seed, default is 888")
    if len(sys.argv) < 2:
        print(parser.print_help())
        exit(1)
    return parser.parse_args()

if __name__ == "__main__":
    # set parameters
    config = parse_arg()
    # set check point output folder
    config.ckpt_path = f'ckpt/{config.name}'
    # save parameters
    save_settings()
    record_fout = open(os.path.join(config.ckpt_path, 'record.txt'), 'a')
    # load mutation data
    dataset = MutationDataset(config, config.data_path)
    patient_mapping_fn = join(config.ckpt_path, 'patient_mapping.json')
    print(f'Total mutation = {len(dataset)}')
    mycollator = MyCollator(config, dataset)
    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=0,
                            collate_fn=mycollator)
    # training process
    model = MutSpace(config=config, n_features=dataset.feature_num)
    model = model.cuda()
    config.verbose_step = len(dataloader) // 2
    range_loss = range_ps = range_ns = 0
    for epc in range(config.epochs):
        for step, batch in enumerate(tqdm(dataloader)):
            loss, ps, ns = model.train_batch(batch)
            range_loss += loss
            range_ps += ps
            range_ns += ns
            if (step + 1) % config.verbose_step == 0:
                log = f'loss = {range_loss / config.verbose_step} pos score = {range_ps / config.verbose_step}, neg score = {range_ns / config.verbose_step}\n'
                print(log)
                record_fout.writelines(log)
                range_loss = range_ps = range_ns = 0

        torch.save(model.state_dict(), f'ckpt/{config.name}/{epc}.pth')
    # finish
    record_fout.close()
