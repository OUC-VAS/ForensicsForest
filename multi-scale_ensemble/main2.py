from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from deepforest import CascadeForestClassifier
import numpy as np
import pandas as pd
import time
import argparse

parser = argparse.ArgumentParser(description='Process some paths.')

parser.add_argument('--dataset_path', type=str, default='/media/ForensicsForest-main/StyleGAN/', help='The path of final input features')

args = parser.parse_args()

tic = time.time()

X_train1 = np.load(args.dataset_path + 'N=2/X_train1.npy', allow_pickle=True)

X_train2 = np.load(args.dataset_path + 'N=2/X_train2.npy', allow_pickle=True)


X_test1 = np.load(args.dataset_path + 'N=2/X_test1.npy', allow_pickle=True)

X_test2 = np.load(args.dataset_path + 'N=2/X_test2.npy', allow_pickle=True)


# label====================================================================================================================
label = pd.read_csv(args.dataset_path + 'train.csv')
label = label["label"]
y_train = np.array(label)

label = pd.read_csv(args.dataset_path + 'test.csv')
label = label["label"]
y_test = np.array(label)


model = CascadeForestClassifier(random_state=1)


# training=======================================9.....=========================================================================
X_aug_train = model.fit_2patch(X_train1, X_train2, y_train)


# evaluating================================================================================================================
y_pred, proba, X_aug_test = model.predict_2patch(X_test1, X_test2, y_test)

np.save(args.dataset_path + 'X_aug_train2.npy', X_aug_train)
np.save(args.dataset_path + 'X_aug_test2.npy', X_aug_test)


acc = accuracy_score(y_test, y_pred) * 100
print("\nTesting Accuracy: {:.3f} %".format(acc))

fpr, tpr, thesholds = metrics.roc_curve(y_test, proba[:, 1], pos_label=1)
auc = metrics.auc(fpr, tpr)*100
print("\nTesting AUC: {:.3f} %".format(auc))

toc = time.time()
time = toc - tic
print("\ntime: {:.3f} s".format(time))
