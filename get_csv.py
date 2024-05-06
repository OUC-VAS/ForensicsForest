import csv
from glob import glob
from tqdm import tqdm
import os

import argparse

parser = argparse.ArgumentParser(description='Process some paths.')

parser.add_argument('--dataset_path', type=str, default='./StyleGAN/train/',)

parser.add_argument('--save_csv_path', type=str, default='./StyleGAN/train.csv')

args = parser.parse_args()

path_list = [0, 1]
headers = ['filename', 'label']
tmp = []
with open(args.save_csv_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for i in tqdm(range(len(path_list))):
        file_path = args.path + '/' + str(i)
        file_list = sorted(glob(file_path + '/*.png'))
        for j in range(len(file_list)):
            tmp.append(os.args.path.basename(file_list[j]))
            tmp.append(i)
            writer.writerow(tmp)
            tmp = []
