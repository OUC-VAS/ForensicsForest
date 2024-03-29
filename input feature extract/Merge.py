import numpy as np

a = ('./train/N=4/0_fake/X1.npy')
b = ('./train/N=4/1_real/X1.npy')
X_train1 = np.hstack((a, b))
np.save('./N=4/X_train1.npy', X_train1)

a = ('./train/N=4/0_fake/X2.npy')
b = ('./train/N=4/1_real/X2.npy')
X_train2 = np.hstack((a, b))
np.save('./N=4/X_train2.npy', X_train2)

a = ('./train/N=4/0_fake/X3.npy')
b = ('./train/N=4/1_real/X3.npy')
X_train3 = np.hstack((a, b))
np.save('./N=4/X_train3.npy', X_train3)

a = ('./train/N=4/0_fake/X4.npy')
b = ('./train/N=4/1_real/X4.npy')
X_train4 = np.hstack((a, b))
np.save('./N=4/X_train4.npy', X_train4)

a = ('./test/N=4/0_fake/X1.npy')
b = ('./test/N=4/1_real/X1.npy')
X_test1 = np.hstack((a, b))
np.save('./N=4/X_test1.npy', X_test1)

a = ('./test/N=4/0_fake/X2.npy')
b = ('./test/N=4/1_real/X2.npy')
X_test2 = np.hstack((a, b))
np.save('./N=4/X_test2.npy', X_test2)

a = ('./test/N=4/0_fake/X3.npy')
b = ('./test/N=4/1_real/X3.npy')
X_test3 = np.hstack((a, b))
np.save('./N=4/X_test3.npy', X_test3)

a = ('./test/N=4/0_fake/X4.npy')
b = ('./test/N=4/1_real/X4.npy')
X_test4 = np.hstack((a, b))
np.save('./N=4/X_test4.npy', X_test4)
