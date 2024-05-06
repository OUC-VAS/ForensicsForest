import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some paths.')

parser.add_argument('--train_feature_path', type=str, default='/media/ForensicsForest-main/StyleGAN/N=4/train/')

parser.add_argument('--test_feature_path', type=str, default='/media/ForensicsForest-main/StyleGAN/N=4/test/')

parser.add_argument('--save_path', type=str, default='/media/ForensicsForest-main/StyleGAN/N=4/')

args = parser.parse_args()

a = np.load(args.train_feature_path + '0/X1.npy', allow_pickle=True)

b = np.load(args.train_feature_path + '1/X1.npy', allow_pickle=True)
X_train1 = np.vstack((a, b))

np.save(args.save_path + 'X_train1.npy', X_train1)

a = np.load(args.train_feature_path + '0/X2.npy', allow_pickle=True)
b = np.load(args.train_feature_path + '1/X2.npy', allow_pickle=True)
X_train2 = np.vstack((a, b))
np.save(args.save_path + 'X_train2.npy', X_train2)

a = np.load(args.train_feature_path + '0/X3.npy', allow_pickle=True)
b = np.load(args.train_feature_path + '1/X3.npy', allow_pickle=True)
X_train3 = np.vstack((a, b))
np.save(args.save_path + 'X_train3.npy', X_train3)

a = np.load(args.train_feature_path + '0/X4.npy', allow_pickle=True)
b = np.load(args.train_feature_path + '1/X4.npy', allow_pickle=True)
X_train4 = np.vstack((a, b))
np.save(args.save_path + 'X_train4.npy', X_train4)

a = np.load(args.test_feature_path + '0/X1.npy', allow_pickle=True)
b = np.load(args.test_feature_path + '1/X1.npy', allow_pickle=True)
X_test1 = np.vstack((a, b))
np.save(args.save_path + 'X_test1.npy', X_test1)

a = np.load(args.test_feature_path + '0/X2.npy', allow_pickle=True)
b = np.load(args.test_feature_path + '1/X2.npy', allow_pickle=True)
X_test2 = np.vstack((a, b))
np.save(args.save_path + 'X_test2.npy', X_test2)

a = np.load(args.test_feature_path + '0/X3.npy', allow_pickle=True)
b = np.load(args.test_feature_path + '1/X3.npy', allow_pickle=True)
X_test3 = np.vstack((a, b))
np.save(args.save_path + 'X_test3.npy', X_test3)

a = np.load(args.test_feature_path + '0/X4.npy', allow_pickle=True)
b = np.load(args.test_feature_path + '1/X4.npy', allow_pickle=True)
X_test4 = np.vstack((a, b))
np.save(args.save_path + 'X_test4.npy', X_test4)
