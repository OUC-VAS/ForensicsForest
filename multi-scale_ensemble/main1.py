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

parser.add_argument('--dataset_path', type=str, default='./StyleGAN/', help='The path of final input features')

args = parser.parse_args()

tic = time.time()

X_train = np.load(args.dataset_path + 'N=1/X_train.npy', allow_pickle=True)
X_test = np.load(args.dataset_path + 'N=1/X_test.npy', allow_pickle=True)

X_aug_train2 = np.load(args.dataset_path + 'X_aug_train2.npy')
X_aug_test2 = np.load(args.dataset_path + 'X_aug_test2.npy')

X_aug_train3 = np.load(args.dataset_path + 'X_aug_train3.npy')
X_aug_test3 = np.load(args.dataset_path + 'X_aug_test3.npy')

X_aug_train4 = np.load(args.dataset_path + 'X_aug_train4.npy')
X_aug_test4 = np.load(args.dataset_path + 'X_aug_test4.npy')


a = []
for i in range(X_train.shape[0]):
    b = []
    b = list(X_train[i])
    b = b + list(X_aug_train2[i])
    b = b + list(X_aug_train3[i])
    b = b + list(X_aug_train4[i])

    a.append(b)
X_train = np.array(a)

a = []
for i in range(X_test.shape[0]):
    b = []
    b = list(X_test[i])
    b = b + list(X_aug_test2[i])
    b = b + list(X_aug_test3[i])
    b = b + list(X_aug_test4[i])

    a.append(b)
X_test = np.array(a)

# label====================================================================================================================
label = pd.read_csv(args.dataset_path + 'train.csv')
label = label["label"]
y_train = np.array(label)

label = pd.read_csv(args.dataset_path + 'test.csv')
label = label["label"]
y_test = np.array(label)


model = CascadeForestClassifier(random_state=1)

# training================================================================================================================
X_aug_train = model.fit_1patch(X_train, y_train)

# evaluating================================================================================================================
y_pred, proba, X_aug_test = model.predict_1patch(X_test, y_test)
acc = accuracy_score(y_test, y_pred) * 100
fpr, tpr, thesholds = metrics.roc_curve(y_test, proba[:, 1], pos_label=1)
auc = metrics.auc(fpr, tpr)*100
print("\nTesting Accuracy: {:.3f} %".format(acc))
print("\nTesting AUC: {:.3f} %".format(auc))

toc = time.time()
time = toc - tic
print("\ntime: {:.3f} s".format(time))
