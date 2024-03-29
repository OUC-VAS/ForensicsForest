import csv
from glob import glob
from tqdm import tqdm
import os
path = './train/'
#输入路径（train/test的路径）

path_list = [0, 1]
headers = ['filename', 'label']
tmp = []
with open(path + '/train.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for i in tqdm(range(len(path_list))):
        file_path = path + '/' + str(i)
        file_list = sorted(glob(file_path + '/*.png'))
        for j in range(len(file_list)):
            tmp.append(os.path.basename(file_list[j]))
            tmp.append(i)
            writer.writerow(tmp)
            tmp = []
