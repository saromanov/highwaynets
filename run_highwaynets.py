from highwaynets import HighwayNetwork
from sklearn.datasets import load_digits
import numpy as np

dataset = load_digits()
X = dataset.data
X /=X.max()
Y = np.array([float(v) for v in dataset.target])
print(Y)
hn = HighwayNetwork(x=X, labels=Y, num_vis=64, num_hid=10, num_out=1797)
hn.train()
