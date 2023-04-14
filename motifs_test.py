import matrixprofile as mp
from matrixprofile.visualize import plot_motifs_mp
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
from vmdpy import VMD
import tslearn
matplotlib.rcParams['figure.figsize'] = (15.0, 10.0)
dataset=pd.read_csv("./data/apartment_2015/Apt1_2015.csv")
dataset.columns=['time','load']
dataset['load_name'] = pd.to_datetime(dataset['time'])#将所得到的数据转换成时间帧 
dataset = dataset.set_index('time').sort_index()
# 参数设置
alpha = 2000      # moderate bandwidth constraint
tau = 0.            # noise-tolerance (no strict fidelity enforcement)
K = 5            # 3 modes
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-7
 
u, u_hat, omega = VMD(dataset['load'].values, alpha, tau, K, DC, init, tol)

plt.figure(figsize=(10, 8))
for i in range(K):
    plt.subplot(K+1, 1, 1)
    plt.plot(dataset['load'])
    plt.title("outer")
    plt.subplot(K+1, 1, i+2)
    plt.plot(u[i, :], linewidth=0.2, c='r')
    plt.ylabel('u{}'.format(i + 1))
plt.tight_layout()

# window_size=48
# profile=mp.compute(dataset['load'].values,windows=window_size)
# profile = mp.discover.motifs(profile, exclusion_zone=window_size)
# mp.visualize(profile)
# plt.show()