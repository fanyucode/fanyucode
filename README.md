# python

# 数据集

## smart meter in london

[**数据链接**](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london?resource=download)

### pre-processing

```python
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import gc
import plotly.io as pio
pio.templates.default = "plotly_white"
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm #进度条库
from multiprocessing.pool import ThreadPool
from joblib import Parallel, delayed
np.random.seed()
tqdm.pandas()
```

选择一个数据转换数据

```python
block_1 = pd.read_csv(block_data_path / "block_0.csv", parse_dates=False)
block_1["day"] = pd.to_datetime(block_1["day"], yearfirst=True)
block_1.head()
```

生成5行50列数据

```python
block_1.groupby("LCLid")["day"].max().sample(5)
```

检查所有的时间序列的结束时期

```python
%%time   #原始的更耗时
max_date = None
for f in tqdm(block_data_path.glob("*.csv")):
    df = pd.read_csv(f, parse_dates=False)
    df["day"] = pd.to_datetime(df["day"], yearfirst=True)
    if max_date is None:
        max_date = df["day"].max()
    else:
        if df["day"].max() > max_date:
            max_date = df["day"].max()
print(f"Max Date across all blocks: {max_date}")
del df
```

```python
%%time
# 速度更快
file_paths = list(block_data_path.glob("*.csv"))#pathname匹配库
def get_max_time(path: Path):
    df = pd.read_csv(f, parse_dates=False)
    df["day"] = pd.to_datetime(df["day"], yearfirst=True)
    return df["day"].max()


with ThreadPool(32) as p:
    max_time_list = p.map(get_max_time, file_paths)
    p.close()
    p.join()

max_date = pd.Series(max_time_list).max()
print(f"Max Date across all blocks: {max_date}")
```

```python
# 将数据帧重新塑造为长形式，并沿行使用小时块
block_1 = (
    block_1.set_index(["LCLid", "day"])
    .stack()
    .reset_index()
    .rename(columns={"level_2": "hour_block", 0: "energy_consumption"})
)
# Creating a numerical hourblock column
block_1["offset"] = block_1["hour_block"].str.replace("hh_", "").astype(int)

block_1.head()
```

#### compact form

- 查找开始日期和时间序列标识符。
- 使用开始日期和全局结束日期创建标准DataFrame。
- Left将LCLid的DataFrame合并到标准DataFrame，将缺失的数据保留为np.nan。
- 根据日期对值进行排序。
- 返回时间序列数组，以及时间序列标识符、开始日期和时间序列的长度。

```python
def preprocess_compact(x):
    start_date = x["day"].min()
    name = x["LCLid"].unique()[0]
    ### 用NaN填充缺失的日期 ###
    #创建一个从最小到最大的日期范围
    dr = pd.date_range(start=x["day"].min(), end=max_date, freq="1D")
    # 将hh_0到hh_47添加到列中，并使用一些解栈魔术重新创建日期-hh_x组合
    dr = (
        pd.DataFrame(columns=[f"hh_{i}" for i in range(48)], index=dr)
        .unstack()
        .reset_index()
    )
    # 重命名列
    dr.columns = ["hour_block", "day", "_"]
    # 左合并数据帧到标准数据帧
    # 现在缺失的值将被保留为NaN
    dr = dr.merge(x, on=["hour_block", "day"], how="left")
    # 对行进行排序
    dr.sort_values(["day", "offset"], inplace=True)
    # 提取时间序列数组
    ts = dr["energy_consumption"].values
    len_ts = len(ts)
    return start_date, name, ts, len_ts
```

```python
def load_process_block_compact(
    block_df, freq="30min", ts_identifier="series_name", value_name="series_value"
):
    grps = block_df.groupby("LCLid")
    all_series = []
    all_start_dates = []
    all_names = []
    all_data = {}
    all_len = []
    for idx, df in tqdm(grps, leave=False):
        start_date, name, ts, len_ts = preprocess_compact(df)
        all_series.append(ts)
        all_start_dates.append(start_date)
        all_names.append(name)
        all_len.append(len_ts)

    all_data[ts_identifier] = all_names
    all_data["start_timestamp"] = all_start_dates
    all_data["frequency"] = freq
    all_data[value_name] = all_series
    all_data["series_length"] = all_len
    return pd.DataFrame(all_data)


block1_compact = load_process_block_compact(
    block_1, freq="30min", ts_identifier="LCLid", value_name="energy_consumption"
)
```

```python
block1_compact.head()
```

```
display(block1_compact.memory_usage(deep=True))
block1_compact.info()
```

#### Expanded form

```python
def preprocess_expanded(x):
    start_date = x["day"].min()
    ### Fill missing dates with NaN ###
    # Create a date range from  min to max
    dr = pd.date_range(start=x["day"].min(), end=x["day"].max(), freq="1D")
    # Add hh_0 to hh_47 to columns and with some unstack magic recreating date-hh_x combinations
    dr = (
        pd.DataFrame(columns=[f"hh_{i}" for i in range(48)], index=dr)
        .unstack()
        .reset_index()
    )
    # renaming the columns
    dr.columns = ["hour_block", "day", "_"]
    # left merging the dataframe to the standard dataframe
    # now the missing values will be left as NaN
    dr = dr.merge(x, on=["hour_block", "day"], how="left")
    dr["series_length"] = len(dr)
    return dr
```

```python
def load_process_block_expanded(block_df, freq="30min"):
    grps = block_df.groupby("LCLid")
    all_series = []
    for idx, df in tqdm(grps, leave=False):
        ts = preprocess_expanded(df)
        all_series.append(ts)

    block_df = pd.concat(all_series)
    # #重新创建偏移量，因为现在会有空行
    block_df["offset"] = block_df["hour_block"].str.replace("hh_", "").astype(int)
    # Creating a datetime column with the date | Will take some time because operation is not vectorized
    block_df["timestamp"] = (
        block_df["day"] + block_df["offset"] * 30 * pd.offsets.Minute()
    )
    block_df["frequency"] = freq
    block_df.sort_values(["LCLid", "timestamp"], inplace=True)
    block_df.drop(columns=["_", "hour_block", "offset", "day"], inplace=True)
    return block_df


#     del all_series
block1_expanded = load_process_block_expanded(block_1, freq="30min")
```

```python
block1_expanded.head()
```

```
display(block1_expanded.memory_usage())
block1_expanded.info()
```

```python
del block1_expanded, block_1, block1_compact
# 内存收集
gc.collect()
```

#### 读取并组合所有块数据到单个数据帧

```python
%%time
# Original Code
block_df_l = []
for file in tqdm(list(block_data_path.glob("*.csv")), desc="Processing Blocks.."):
    block_df = pd.read_csv(file, parse_dates=False)
    block_df['day'] = pd.to_datetime(block_df['day'], yearfirst=True)
    # Taking only from 2012-01-01
    block_df = block_df.loc[block_df['day']>="2012-01-01"]
    #Reshaping the dataframe into the long form with hour blocks along the rows
    block_df = block_df.set_index(['LCLid', "day"]).stack().reset_index().rename(columns={"level_2": "hour_block", 0: "energy_consumption"})
    #Creating a numerical hourblock column
    block_df['offset'] = block_df['hour_block'].str.replace("hh_", "").astype(int)
    block_df_l.append(load_process_block_compact(block_df, freq="30min", ts_identifier="LCLid", value_name="energy_consumption"))
```

# Python库

### tqdm

进度条

### matplotlib

#### pyplot

### numpy

#### 定义

对象是**多维数据**

```
# 通过列表创建一位数组
np.array([1,2,3])
# Rank=1,len(Axes)=3
np.array([(1,2,3),(4,5,6)])
# Rank=2,len(Axes)=3
```

```
# 创建一个三行四列的二维数组，全是0
np.zeros((3,4))
# 创建一个2*3*4的三维数组，全是1
np.ones((2,3,4))
```

```
# 创建一个5*5的二维数组，值都为2
np.full((5,5),2)
```

```
# 随机创建一个二行三列的数组
np.random.rand(2,3)
# 创建一个随机的二维数组，但数字小于5，二行三列
np.random.randint(5,size=(2,3))
```

```
# 假设a为一个二行三列的数组
# 对二行三列求和
np.sum(a)
# 对列求和,因为axis为0
np.sum(a,axis=0)
# 对行求和，因为axis为1
np.sum(a,axis=1)
```

```
# 对数组a求平均，也可以像上面一样按照行列操作
np.mean(a)
# 对数组a排序，可以按照行列排序
np.argsort()
# 按照列排序
np.argsort(axis=0)
```

```
# 对于二维数组A和B，如果求其乘法
A=np.array([[1,2],[3,4]])
B=np.array([[5,6],[7,8]])
# 可以使用dot方法
np.dot(A,B)
# 当然也可以先将其转化为矩阵，然后直接进行*运算就行
np.mat(A)*np.mat(B)
```

```
np.mean()# 平均值
np.cumsum(axis=0)# 按列累加
np.std()# 方差
np.var()# 标准差
np.argmax()# 最大值索引
```

#### 库函数

- 基本函数

	.ndim ：维度
	.[shape](https://so.csdn.net/so/search?q=shape&spm=1001.2101.3001.7020) ：各维度的尺度 （2，5）
	.size ：元素的个数 10
	.dtype ：元素的类型 dtype(‘int32’)
	.itemsize ：每个元素的大小，以字节为单位 ，每个元素占4个字节
	ndarray数组的创建
	np.arange(n) ; 元素从0到n-1的ndarray类型
	np.ones(shape): 生成全1
	np.zeros((shape)， ddtype = np.int32) ： 生成int32型的全0
	np.full(shape, val): 生成全为val
	np.eye(n) : 生成单位矩阵

	np.ones_like(a) : 按数组a的形状生成全1的数组
	np.zeros_like(a): 同理
	np.full_like (a, val) : 同理

	np.linspace（1,10,4）： 根据起止数据等间距地生成数组
	np.linspace（1,10,4, endpoint = False）：endpoint 表示10是否作为生成的元素
	np.concatenate():

- 数组的维度变换

	.reshape(shape) : 不改变当前数组，依shape生成
	.resize(shape) : 改变当前数组，依shape生成
	.swapaxes(ax1, ax2) : 将两个维度调换
	.flatten() : 对数组进行降维，返回折叠后的一位数组

- 数组类型变换

	数据类型的转换 ：a.astype(new_type) : eg, a.astype (np.float)
	数组向列表的转换： a.tolist()
	数组的索引和切片

- 

### multiprocess

用创建子进程，有效地避开了全局解释器锁（GIL）。 因此，multiprocessing模块允许程序员充分利用机器上的多个处理器。 目前，它可以在Unix和Windows上运行。

### pandas

- dropna :DataFrame丢弃缺失数据，默认丢弃出现NaN的行，设置axis=1表示丢弃列，how=‘all’（丢弃全为NaN的行或列），thresh=2（删除存在两个缺失值NaN的行或列

- read_csv：从文件、URL、文件型对象中加载带分隔符的数据。默认分隔符为逗号

- read_table：从文件、URL、文件型对象中加载带分隔符的数据。默认分隔符为制表符（’\t’）

|        **方法**        |                    **说明**                    |
| :--------------------: | :--------------------------------------------: |
|         count          |                  非NA值的数量                  |
|      **describe**      |     针对Series或各DataFrame列计算汇总统计      |
|    **min**、**max**    |               计算最大值和最小值               |
| **argmin**、**argmax** | 计算能够获取到最小值和最大值的索引位置（整数） |
|        **sum**         |                    值的总和                    |
|        **mean**        |                   值的平均数                   |
|       **median**       |          值的算术中位数（50%分位数）           |
|        **var**         |                  样本值的方差                  |
|        **std**         |                 样本值的标准差                 |
|        groupby         |                      分组                      |

- to_csv(将数据写入csv文件)

### plotly(可视化操作）

### joblib

joblib是python中提供一系列轻量级管道操作的 工具; 特别在如下3种工具:

- 函数的透明磁盘缓存和延迟重新计算(记忆模式);
- 容易且简单的平行计算;
- 比 pickle更快的 序列化和反序列化 的功能;

### gc

- 为新生成的对象分配库存
- 识别那些是垃圾对象
- 从垃圾对象那里回收内存

### os模块

os模块是[Python](https://so.csdn.net/so/search?q=Python&spm=1001.2101.3001.7020)中整理文件和目录最为常用的模块，该模块提供了非常丰富的方法用来处理文件和目录。

- os.listdir(path):传入任意一个path路径，返回的是该路径下所有`文件和目录`组成的列表；
- os.walk(path):传入任意一个path路径，深层次遍历指定路径下的所有子文件夹，返回的是一个由路径、文件夹列表、文件列表组成的元组。我代码中写的方式属于`元组拆包`；
- os.path.exists(path):传入一个path路径，判断指定路径下的目录是否存在。存在返回True，否则返回False；
- os.mkdir(path):传入一个path路径，创建单层(单个)文件夹；

# 聚类模型

- 使用划窗为L；将长度m分割成m-l+1个子序列；子序列之间的最小距离使用matrix  profile计算；两个序列之间相似性最小距离的子序列对的

- 网络建设：根据距离值；将时间序列视为顶点；将他们之间的关系视为边；将相似度视为权重，构建网络
- community detection:将建立的网络划分为不同的组

## matrixprofile

### annotation vector 注释向量

```python
def make_complexity_AV(ts, m):
    """
    返回具有窗口ma的时间序列ts的复杂解释的向量
    窗口复杂度是连续点之间的平均绝对差值
    """
    diffs = np.diff(ts, append=0)**2
    diff_mean, diff_std = movmeanstd(diffs, m)

    complexity = np.sqrt(diff_mean)
    complexity = complexity - complexity.min()
    complexity = complexity / complexity.max()
    return complexity
```

```python
def make_meanstd_AV(ts, m):
    """返回布尔注释量，它选择标准偏差值大于平均值的窗口"""
    _, std = movmeanstd(ts, m)
    mu = std.mean()
    return (std < mu).astype(int)
```

```python
def make_clipping_AV(ts, m):
    """
    如果窗口中有最小值/最大值，则返回与数字成比例的注释向量
    """
    av = (ts == ts.min()) | (ts == ts.max())
    av, _ = movmeanstd(av, m)
    return av
```

### discords不和谐

```python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import sys
import numpy as np

def discords(mp,ex_zone,k=3):
    """
    Computes the top k discords from a matrix profile

    Parameters
    ----------
    mp: matrix profile numpy array
    k: the number of discords to discover
    ex_zone: 在发现的不一致的任一侧要排除并设置为 Inf 的样本数
    返回代表不一致起始位置的索引列表.
    MaxInt 表示由于排除太多或配置文件太小而无法找到更多的不一致。 Discord 起始索引按最高矩阵配置文件值排序。
    """
    k = len(mp) if k > len(mp) else k

    mp_current = np.copy(mp)
    d = np.zeros(k, dtype='int')
    for i in range(k):
        maxVal = 0
        maxIdx = sys.maxsize
        for j, val in enumerate(mp_current):
            if not np.isinf(val) and val > maxVal:
                maxVal = val
                maxIdx = j

        d[i] = maxIdx
        mp_current[max([maxIdx-ex_zone, 0]):min([maxIdx+ex_zone, len(mp_current)])] = np.inf

    return d

```

```python
# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import numba
import numpy as np
from numba import njit, prange

from . import config, core
from .scraamp import prescraamp, scraamp
from .stump import _stump


def _preprocess_prescrump(T_A, m, T_B=None, s=None):
    """
    执行几个预处理并返回预挤算法所需的输出。
    参数
    ----------
    T_A : numpy.ndarray
       计算matrix profile时间序列
    m : int
        窗口大小
    T_B : numpy.ndarray, default None
       将用于注释 T_A 的时间序列或序列。 对于 T_A 中的每个子序列，将记录其在 T_B 中的最近邻居。
    s : int, default None
        默认的采样间隔
        `int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`
    Returns
    -------
    T_A : numpy.ndarray
        时间序列输入“T_A”的副本，其中所有 NaN 和 inf 值
		被替换为零。
    T_B : numpy.ndarray
         时间序列输入“T_B”的副本，其中所有 NaN 和 inf 值
         被替换为零。 如果未提供输入“T_B”（默认），
         这个数组只是 `T_A` 的一个副本。
    μ_Q : numpy.ndarray
        `T_A` 的滑动窗口均值
    σ_Q : numpy.ndarray
        `T_A` 的滑动窗口标准偏差
    M_T : numpy.ndarray
        `T_B` 的滑动窗口均值
    Σ_T : numpy.ndarray
        `T_B` 的滑动窗口标准偏差
    Q_subseq_isconstant : numpy.ndarray
        一个布尔数组，指示“Q”中的子序列是否为常量 (True)
    T_subseq_isconstant : numpy.ndarray
        一个布尔数组，指示“T”中的子序列是否为常量 (True)
    indices : numpy.ndarray
        用于计算 `prescrump` 的子序列索引
    s : int
        默认的采样间隔
        `int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`
    excl_zone : int
        禁区的半宽
    """
    if T_B is None:
        T_B = T_A
        excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    else:
        excl_zone = None

    T_A, μ_Q, σ_Q, Q_subseq_isconstant = core.preprocess(T_A, m)
    T_B, M_T, Σ_T, T_subseq_isconstant = core.preprocess(T_B, m)

    n_A = T_A.shape[0]
    l = n_A - m + 1

    if s is None:  # pragma: no cover
        if excl_zone is not None:  # self-join
            s = excl_zone
        else:  # AB-join
            s = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    indices = np.random.permutation(range(0, l, s)).astype(np.int64)

    return (
        T_A,
        T_B,
        μ_Q,
        σ_Q,
        M_T,
        Σ_T,
        Q_subseq_isconstant,
        T_subseq_isconstant,
        indices,
        s,
        excl_zone,
    )


@njit(fastmath=True)
def _compute_PI(
    T_A,
    T_B,
    m,
    μ_Q,
    σ_Q,
    M_T,
    Σ_T,
    Q_subseq_isconstant,
    T_subseq_isconstant,
    indices,
    start,
    stop,
    thread_idx,
    s,
    P_squared,
    I,
    excl_zone=None,
    k=1,
):
    """
    根据 preSCRIMP 算法计算（Numba JIT 编译）和更新平方（top-k）矩阵分布距离和矩阵分布指数。
    参数
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile
    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.
    m : int
        Window size
    μ_Q : numpy.ndarray
        Sliding window mean for `T_A`
    σ_Q : numpy.ndarray
        Sliding window standard deviation for `T_A`
    M_T : numpy.ndarray
        Sliding window mean for `T_B`
    Σ_T : numpy.ndarray
        Sliding window standard deviation for `T_B`
    Q_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` is constant (True)
    T_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` is constant (True)
    indices : numpy.ndarray
        The subsequence indices to compute `prescrump` for
    start : int
        The (inclusive) start index for `indices`
    stop : int
        The (exclusive) stop index for `indices`
    thread_idx : int
        The thread index
    s : int
        The sampling interval that defaults to
        `int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`
    P_squared : numpy.ndarray
        The squared (top-k) matrix profile
    I : numpy.ndarray
        The (top-k) matrix profile indices
    excl_zone : int
        The half width for the exclusion zone relative to the `i`.
    k : int, default 1
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1.
    Returns
    -------
    None
    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__
    See Algorithm 2
    """
    l = T_A.shape[0] - m + 1  # length of matrix profile
    w = T_B.shape[0] - m + 1  # length of distance profile
    squared_distance_profile = np.empty(w)
    QT = np.empty(w, dtype=np.float64)
    for i in indices[start:stop]:
        Q = T_A[i : i + m]
        QT[:] = core._sliding_dot_product(Q, T_B)
        squared_distance_profile[:] = core._mass(
            Q,
            T_B,
            QT,
            μ_Q[i],
            σ_Q[i],
            M_T,
            Σ_T,
            Q_subseq_isconstant[i],
            T_subseq_isconstant,
        )
        squared_distance_profile[:] = np.square(squared_distance_profile)
        if excl_zone is not None:
            core._apply_exclusion_zone(squared_distance_profile, i, excl_zone, np.inf)

        nn_i = np.argmin(squared_distance_profile)
        if (
            squared_distance_profile[nn_i] < P_squared[thread_idx, i, -1]
            and nn_i not in I[thread_idx, i]
        ):
            idx = np.searchsorted(
                P_squared[thread_idx, i],
                squared_distance_profile[nn_i],
                side="right",
            )
            core._shift_insert_at_index(
                P_squared[thread_idx, i], idx, squared_distance_profile[nn_i]
            )
            core._shift_insert_at_index(I[thread_idx, i], idx, nn_i)

        if P_squared[thread_idx, i, 0] == np.inf:  # pragma: no cover
            I[thread_idx, i, 0] = -1
            continue

        j = nn_i
        # Given the squared distance, work backwards and compute QT
        QT_j = (m - P_squared[thread_idx, i, 0] / 2.0) * (Σ_T[j] * σ_Q[i]) + (
            m * M_T[j] * μ_Q[i]
        )
        QT_j_prime = QT_j
        # Update top-k for both subsequences `S[i+g] = T[i+g:i+g+m]`` and
        # `S[j+g] = T[j+g:j+g+m]` (i.e., the right neighbors of `T[i : i+m]` and
        # `T[j:j+m]`) by using the distance between `S[i+g]` and `S[j+g]`
        for g in range(1, min(s, l - i, w - j)):
            QT_j = (
                QT_j
                - T_B[j + g - 1] * T_A[i + g - 1]
                + T_B[j + g + m - 1] * T_A[i + g + m - 1]
            )
            D_squared = core._calculate_squared_distance(
                m,
                QT_j,
                M_T[j + g],
                Σ_T[j + g],
                μ_Q[i + g],
                σ_Q[i + g],
                T_subseq_isconstant[j + g],
                Q_subseq_isconstant[i + g],
            )
            if (
                D_squared < P_squared[thread_idx, i + g, -1]
                and (j + g) not in I[thread_idx, i + g]
            ):
                idx = np.searchsorted(
                    P_squared[thread_idx, i + g], D_squared, side="right"
                )
                core._shift_insert_at_index(
                    P_squared[thread_idx, i + g], idx, D_squared
                )
                core._shift_insert_at_index(I[thread_idx, i + g], idx, j + g)

            if (
                excl_zone is not None
                and D_squared < P_squared[thread_idx, j + g, -1]
                and (i + g) not in I[thread_idx, j + g]
            ):
                idx = np.searchsorted(
                    P_squared[thread_idx, j + g], D_squared, side="right"
                )
                core._shift_insert_at_index(
                    P_squared[thread_idx, j + g], idx, D_squared
                )
                core._shift_insert_at_index(I[thread_idx, j + g], idx, i + g)

        QT_j = QT_j_prime
        # Update top-k for both subsequences `S[i-g] = T[i-g:i-g+m]` and
        # `S[j-g] = T[j-g:j-g+m]` (i.e., the left neighbors of `T[i : i+m]` and
        # `T[j:j+m]`) by using the distance between `S[i-g]` and `S[j-g]`
        for g in range(1, min(s, i + 1, j + 1)):
            QT_j = QT_j - T_B[j - g + m] * T_A[i - g + m] + T_B[j - g] * T_A[i - g]
            D_squared = core._calculate_squared_distance(
                m,
                QT_j,
                M_T[j - g],
                Σ_T[j - g],
                μ_Q[i - g],
                σ_Q[i - g],
                T_subseq_isconstant[j - g],
                Q_subseq_isconstant[i - g],
            )
            if (
                D_squared < P_squared[thread_idx, i - g, -1]
                and (j - g) not in I[thread_idx, i - g]
            ):
                idx = np.searchsorted(
                    P_squared[thread_idx, i - g], D_squared, side="right"
                )
                core._shift_insert_at_index(
                    P_squared[thread_idx, i - g], idx, D_squared
                )
                core._shift_insert_at_index(I[thread_idx, i - g], idx, j - g)

            if (
                excl_zone is not None
                and D_squared < P_squared[thread_idx, j - g, -1]
                and (i - g) not in I[thread_idx, j - g]
            ):
                idx = np.searchsorted(
                    P_squared[thread_idx, j - g], D_squared, side="right"
                )
                core._shift_insert_at_index(
                    P_squared[thread_idx, j - g], idx, D_squared
                )
                core._shift_insert_at_index(I[thread_idx, j - g], idx, i - g)

        # In the case of a self-join, the calculated distance profile can also be
        # used to refine the top-k for all non-trivial subsequences
        if excl_zone is not None:
            # Note that the squared distance, `squared_distance_profile[j]`,
            # between subsequences `S_i = T[i : i + m]` and `S_j = T[j : j + m]`
            # can be used to update the top-k for BOTH subsequence `i` and
            # subsequence `j`. We update the latter here.

            indices = np.flatnonzero(
                squared_distance_profile < P_squared[thread_idx, :, -1]
            )
            for j in indices:
                if i not in I[thread_idx, j]:
                    idx = np.searchsorted(
                        P_squared[thread_idx, j],
                        squared_distance_profile[j],
                        side="right",
                    )
                    core._shift_insert_at_index(
                        P_squared[thread_idx, j], idx, squared_distance_profile[j]
                    )
                    core._shift_insert_at_index(I[thread_idx, j], idx, i)


@njit(
    # "(f8[:], f8[:], i8, f8[:], f8[:], f8[:], f8[:], f8[:], i8, i8, f8[:], f8[:],"
    # "i8[:], optional(i8))",
    parallel=True,
    fastmath=True,
)
def _prescrump(
    T_A,
    T_B,
    m,
    μ_Q,
    σ_Q,
    M_T,
    Σ_T,
    Q_subseq_isconstant,
    T_subseq_isconstant,
    indices,
    s,
    excl_zone=None,
    k=1,
):
    """
    preSCRIMP 算法的 Numba JIT 编译实现。
    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile
    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.
    m : int
        Window size
    μ_Q : numpy.ndarray
        Sliding window mean for `T_A`
    σ_Q : numpy.ndarray
        Sliding window standard deviation for `T_A`
    M_T : numpy.ndarray
        Sliding window mean for `T_B`
    Σ_T : numpy.ndarray
        Sliding window standard deviation for `T_B`
    Q_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_A` is constant (True)
    T_subseq_isconstant : numpy.ndarray
        A boolean array that indicates whether a subsequence in `T_B` is constant (True)
    indices : numpy.ndarray
        The subsequence indices to compute `prescrump` for
    idx_ranges : numpy.ndarray
        The (inclusive) start indices and (exclusive) stop indices referenced
        in the `indices` array
    s : int
        The sampling interval that defaults to
        `int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`
    P_squared : numpy.ndarray
        The squared matrix profile
    I : numpy.ndarray
        The matrix profile indices
    excl_zone : int
        The half width for the exclusion zone relative to the `i`.
    k : int, default 1
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1.
    Returns
    -------
    out1 : numpy.ndarray
        The (top-k) matrix profile. When k=1 (default), the first (and only) column
        in this 2D array consists of the matrix profile. When k > 1, the output
        has exactly `k` columns consisting of the top-k matrix profile.
    out2 : numpy.ndarray
        The (top-k) matrix profile indices. When k=1 (default), the first (and only)
        column in this 2D array consists of the matrix profile indices. When k > 1,
        the output has exactly `k` columns consisting of the top-k matrix profile
        indices.
    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__
    See Algorithm 2
    """
    n_threads = numba.config.NUMBA_NUM_THREADS
    l = T_A.shape[0] - m + 1
    P_squared = np.full((n_threads, l, k), np.inf, dtype=np.float64)
    I = np.full((n_threads, l, k), -1, dtype=np.int64)

    idx_ranges = core._get_ranges(len(indices), n_threads, truncate=False)
    for thread_idx in prange(n_threads):
        _compute_PI(
            T_A,
            T_B,
            m,
            μ_Q,
            σ_Q,
            M_T,
            Σ_T,
            Q_subseq_isconstant,
            T_subseq_isconstant,
            indices,
            idx_ranges[thread_idx, 0],
            idx_ranges[thread_idx, 1],
            thread_idx,
            s,
            P_squared,
            I,
            excl_zone,
            k,
        )

    for thread_idx in range(1, n_threads):
        core._merge_topk_PI(P_squared[0], P_squared[thread_idx], I[0], I[thread_idx])

    return np.sqrt(P_squared[0]), I[0]


@core.non_normalized(prescraamp)
def prescrump(T_A, m, T_B=None, s=None, normalize=True, p=2.0, k=1):
    """
    围绕 Numba JIT 编译的并行化“_prescrump”函数的便利包装器，该函数根据 preSCRIMP 算法计算近似（top-k）矩阵配置文件。
    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile
    m : int
        Window size
    T_B : numpy.ndarray, default None
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.
    s : int, default None
        The sampling interval that defaults to
        `int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`
    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this function gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` function decorator.
    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == True`.
    k : int, default 1
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1.
    Returns
    -------
    P : numpy.ndarray
        The (top-k) matrix profile. When k = 1 (default), this is a 1D array
        consisting of the matrix profile. When k > 1, the output is a 2D array that
        has exactly `k` columns consisting of the top-k matrix profile.
    I : numpy.ndarray
        The (top-k) matrix profile indices. When k = 1 (default), this is a 1D array
        consisting of the matrix profile indices. When k > 1, the output is a 2D
        array that has exactly `k` columns consisting of the top-k matrix profile
        indices.
    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__
    See Algorithm 2
    """
    (
        T_A,
        T_B,
        μ_Q,
        σ_Q,
        M_T,
        Σ_T,
        Q_subseq_isconstant,
        T_subseq_isconstant,
        indices,
        s,
        excl_zone,
    ) = _preprocess_prescrump(T_A, m, T_B=T_B, s=s)

    P, I = _prescrump(
        T_A,
        T_B,
        m,
        μ_Q,
        σ_Q,
        M_T,
        Σ_T,
        Q_subseq_isconstant,
        T_subseq_isconstant,
        indices,
        s,
        excl_zone,
        k,
    )

    if k == 1:
        return P.flatten().astype(np.float64), I.flatten().astype(np.int64)
    else:
        return P, I


@core.non_normalized(
    scraamp,
    exclude=["normalize", "pre_scrump", "pre_scraamp", "p"],
    replace={"pre_scrump": "pre_scraamp"},
)
class scrump:
    """
    Compute an approximate z-normalized matrix profile
    This is a convenience wrapper around the Numba JIT-compiled parallelized
    `_stump` function which computes the matrix profile according to SCRIMP.
    Parameters
    ----------
    T_A : numpy.ndarray
        The time series or sequence for which to compute the matrix profile
    T_B : numpy.ndarray
        The time series or sequence that will be used to annotate T_A. For every
        subsequence in T_A, its nearest neighbor in T_B will be recorded.
    m : int
        Window size
    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
        `False`. Default is `True`.
    percentage : float
        Approximate percentage completed. The value is between 0.0 and 1.0.
    pre_scrump : bool
        A flag for whether or not to perform the PreSCRIMP calculation prior to
        computing SCRIMP. If set to `True`, this is equivalent to computing
        SCRIMP++ and may lead to faster convergence
    s : int
        The size of the PreSCRIMP fixed interval. If `pre_scrump=True` and `s=None`,
        then `s` will automatically be set to
        `s=int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`, the size of the exclusion
        zone.
    normalize : bool, default True
        When set to `True`, this z-normalizes subsequences prior to computing distances.
        Otherwise, this class gets re-routed to its complementary non-normalized
        equivalent set in the `@core.non_normalized` class decorator.
    p : float, default 2.0
        The p-norm to apply for computing the Minkowski distance. This parameter is
        ignored when `normalize == True`.
    k : int, default 1
        The number of top `k` smallest distances used to construct the matrix profile.
        Note that this will increase the total computational time and memory usage
        when k > 1.
    Attributes
    ----------
    P_ : numpy.ndarray
        The updated (top-k) matrix profile. When `k=1` (default), this output is
        a 1D array consisting of the matrix profile. When `k > 1`, the output
        is a 2D array that has exactly `k` columns consisting of the top-k matrix
        profile.
    I_ : numpy.ndarray
        The updated (top-k) matrix profile indices. When `k=1` (default), this output is
        a 1D array consisting of the matrix profile indices. When `k > 1`, the output
        is a 2D array that has exactly `k` columns consisting of the top-k matrix
        profile indiecs.
    left_I_ : numpy.ndarray
        The updated left (top-1) matrix profile indices
    right_I_ : numpy.ndarray
        The updated right (top-1) matrix profile indices
    Methods
    -------
    update()
        Update the matrix profile and the matrix profile indices by computing
        additional new distances (limited by `percentage`) that make up the full
        distance matrix. It updates the (top-k) matrix profile, (top-1) left
        matrix profile, (top-1) right matrix profile, (top-k) matrix profile indices,
        (top-1) left matrix profile indices, and (top-1) right matrix profile indices.
    See Also
    --------
    stumpy.stump : Compute the z-normalized matrix profile
    stumpy.stumped : Compute the z-normalized matrix profile with a distributed dask
        cluster
    stumpy.gpu_stump : Compute the z-normalized matrix profile with one or more GPU
        devices
    Notes
    -----
    `DOI: 10.1109/ICDM.2018.00099 \
    <https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf>`__
    See Algorithm 1 and Algorithm 2
    Examples
    --------
    >>> import stumpy
    >>> import numpy as np
    >>> approx_mp = stumpy.scrump(
    ...     np.array([584., -11., 23., 79., 1001., 0., -19.]),
    ...     m=3)
    >>> approx_mp.update()
    >>> approx_mp.P_
    array([2.982409  , 3.28412702,        inf, 2.982409  , 3.28412702])
    >>> approx_mp.I_
    array([ 3,  4, -1,  0,  1])
    """

    def __init__(
        self,
        T_A,
        m,
        T_B=None,
        ignore_trivial=True,
        percentage=0.01,
        pre_scrump=False,
        s=None,
        normalize=True,
        p=2.0,
        k=1,
    ):
        """
        Initialize the `scrump` object
        Parameters
        ----------
        T_A : numpy.ndarray
            The time series or sequence for which to compute the matrix profile
        m : int
            Window size
        T_B : numpy.ndarray, default None
            The time series or sequence that will be used to annotate T_A. For every
            subsequence in T_A, its nearest neighbor in T_B will be recorded.
        ignore_trivial : bool, default True
            Set to `True` if this is a self-join. Otherwise, for AB-join, set this to
            `False`. Default is `True`.
        percentage : float, default 0.01
            Approximate percentage completed. The value is between 0.0 and 1.0.
        pre_scrump : bool, default False
            A flag for whether or not to perform the PreSCRIMP calculation prior to
            computing SCRIMP. If set to `True`, this is equivalent to computing
            SCRIMP++
        s : int, default None
            The size of the PreSCRIMP fixed interval. If `pre_scrump=True` and `s=None`,
            then `s` will automatically be set to
            `s=int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))`, the size of the
            exclusion zone.
        normalize : bool, default True
            When set to `True`, this z-normalizes subsequences prior to computing
            distances. Otherwise, this class gets re-routed to its complementary
            non-normalized equivalent set in the `@core.non_normalized` class decorator.
        p : float, default 2.0
            The p-norm to apply for computing the Minkowski distance. This parameter is
            ignored when `normalize == True`.
        k : int, default 1
            The number of top `k` smallest distances used to construct the matrix
            profile. Note that this will increase the total computational time and
            memory usage when k > 1.
        """
        self._ignore_trivial = ignore_trivial

        if T_B is None:
            T_B = T_A
            self._ignore_trivial = True

        self._m = m
        (
            self._T_A,
            self._μ_Q,
            self._σ_Q_inverse,
            self._μ_Q_m_1,
            self._T_A_subseq_isfinite,
            self._T_A_subseq_isconstant,
        ) = core.preprocess_diagonal(T_A, self._m)

        (
            self._T_B,
            self._M_T,
            self._Σ_T_inverse,
            self._M_T_m_1,
            self._T_B_subseq_isfinite,
            self._T_B_subseq_isconstant,
        ) = core.preprocess_diagonal(T_B, self._m)

        if self._T_A.ndim != 1:  # pragma: no cover
            raise ValueError(
                f"T_A is {self._T_A.ndim}-dimensional and must be 1-dimensional. "
                "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
            )

        if self._T_B.ndim != 1:  # pragma: no cover
            raise ValueError(
                f"T_B is {self._T_B.ndim}-dimensional and must be 1-dimensional. "
                "For multidimensional STUMP use `stumpy.mstump` or `stumpy.mstumped`"
            )

        core.check_window_size(m, max_size=min(T_A.shape[0], T_B.shape[0]))
        self._ignore_trivial = core.check_ignore_trivial(
            self._T_A, self._T_B, self._ignore_trivial
        )

        self._n_A = self._T_A.shape[0]
        self._n_B = self._T_B.shape[0]
        self._l = self._n_A - self._m + 1
        self._k = k

        self._P = np.full((self._l, self._k), np.inf, dtype=np.float64)
        self._PL = np.full(self._l, np.inf, dtype=np.float64)
        self._PR = np.full(self._l, np.inf, dtype=np.float64)

        self._I = np.full((self._l, self._k), -1, dtype=np.int64)
        self._IL = np.full(self._l, -1, dtype=np.int64)
        self._IR = np.full(self._l, -1, dtype=np.int64)

        self._excl_zone = int(np.ceil(self._m / config.STUMPY_EXCL_ZONE_DENOM))
        if s is None:
            if self._excl_zone is not None:  # self-join
                s = self._excl_zone
            else:  # pragma: no cover  # AB-join
                s = int(np.ceil(self._m / config.STUMPY_EXCL_ZONE_DENOM))

        if pre_scrump:
            if self._ignore_trivial:
                (
                    T_A,
                    T_B,
                    μ_Q,
                    σ_Q,
                    M_T,
                    Σ_T,
                    Q_subseq_isconstant,
                    T_subseq_isconstant,
                    indices,
                    s,
                    excl_zone,
                ) = _preprocess_prescrump(T_A, m, s=s)
            else:
                (
                    T_A,
                    T_B,
                    μ_Q,
                    σ_Q,
                    M_T,
                    Σ_T,
                    Q_subseq_isconstant,
                    T_subseq_isconstant,
                    indices,
                    s,
                    excl_zone,
                ) = _preprocess_prescrump(T_A, m, T_B=T_B, s=s)

            P, I = _prescrump(
                T_A,
                T_B,
                m,
                μ_Q,
                σ_Q,
                M_T,
                Σ_T,
                Q_subseq_isconstant,
                T_subseq_isconstant,
                indices,
                s,
                excl_zone,
                k,
            )
            core._merge_topk_PI(self._P, P, self._I, I)

        if self._ignore_trivial:
            self._diags = np.random.permutation(
                range(self._excl_zone + 1, self._n_A - self._m + 1)
            ).astype(np.int64)
            if self._diags.shape[0] == 0:  # pragma: no cover
                max_m = core.get_max_window_size(self._T_A.shape[0])
                raise ValueError(
                    f"The window size, `m = {self._m}`, is too long for a self join. "
                    f"Please try a value of `m <= {max_m}`"
                )
        else:
            self._diags = np.random.permutation(
                range(-(self._n_A - self._m + 1) + 1, self._n_B - self._m + 1)
            ).astype(np.int64)

        self._n_threads = numba.config.NUMBA_NUM_THREADS
        self._percentage = np.clip(percentage, 0.0, 1.0)
        self._n_chunks = int(np.ceil(1.0 / percentage))
        self._ndist_counts = core._count_diagonal_ndist(
            self._diags, self._m, self._n_A, self._n_B
        )
        self._chunk_diags_ranges = core._get_array_ranges(
            self._ndist_counts, self._n_chunks, True
        )
        self._n_chunks = self._chunk_diags_ranges.shape[0]
        self._chunk_idx = 0

    def update(self):
        """
        Update the (top-k) matrix profile and the (top-k) matrix profile indices by
        computing additional new distances (limited by `percentage`) that make up
        the full distance matrix.
        """
        if self._chunk_idx < self._n_chunks:
            start_idx, stop_idx = self._chunk_diags_ranges[self._chunk_idx]

            P, PL, PR, I, IL, IR = _stump(
                self._T_A,
                self._T_B,
                self._m,
                self._M_T,
                self._μ_Q,
                self._Σ_T_inverse,
                self._σ_Q_inverse,
                self._M_T_m_1,
                self._μ_Q_m_1,
                self._T_A_subseq_isfinite,
                self._T_B_subseq_isfinite,
                self._T_A_subseq_isconstant,
                self._T_B_subseq_isconstant,
                self._diags[start_idx:stop_idx],
                self._ignore_trivial,
                self._k,
            )

            # Update (top-k) matrix profile and indices
            core._merge_topk_PI(self._P, P, self._I, I)

            # update left matrix profile and indices
            mask = PL < self._PL
            self._PL[mask] = PL[mask]
            self._IL[mask] = IL[mask]

            # update right matrix profile and indices
            mask = PR < self._PR
            self._PR[mask] = PR[mask]
            self._IR[mask] = IR[mask]

            self._chunk_idx += 1

    @property
    def P_(self):
        """
        Get the updated (top-k) matrix profile. When `k=1` (default), this output
        is a 1D array consisting of the updated matrix profile. When `k > 1`, the
        output is a 2D array that has exactly `k` columns consisting of the updated
        top-k matrix profile.
        """
        if self._k == 1:
            return self._P.flatten().astype(np.float64)
        else:
            return self._P.astype(np.float64)

    @property
    def I_(self):
        """
        Get the updated (top-k) matrix profile indices. When `k=1` (default), this
        output is a 1D array consisting of the updated matrix profile indices. When
        `k > 1`, the output is a 2D array that has exactly `k` columns consisting
        of the updated top-k matrix profile indices.
        """
        if self._k == 1:
            return self._I.flatten().astype(np.int64)
        else:
            return self._I.astype(np.int64)

    @property
    def left_I_(self):
        """
        Get the updated left (top-1) matrix profile indices
        """
        return self._IL.astype(np.int64)

    @property
    def right_I_(self):
        """
        Get the updated right (top-1) matrix profile indices
        """
        return self._IR.astype(np.int64)
```

## 代码部分

### 生成每个家庭电力脚本软件

```python
"""

用于处理原始智能电表数据文件并为每个家庭生成半小时电力读数的脚本。

"""

import pandas as pd
import numpy as np
import tqdm
import os
import zipfile

from asf_smart_meter_exploration import base_config, PROJECT_DIR

meter_data_zip_path = PROJECT_DIR / base_config["meter_data_zip_path"]
meter_data_folder_path = PROJECT_DIR / base_config["meter_data_folder_path"]
meter_data_merged_folder_path = (
    PROJECT_DIR / base_config["meter_data_merged_folder_path"]
)
meter_data_merged_file_path = PROJECT_DIR / base_config["meter_data_merged_file_path"]


def unzip_raw_data():
    """解压原始文件"""
    if not os.path.isfile(meter_data_zip_path):
        raise FileNotFoundError(
            "未找到压缩文件。 请检查文件位置或从 S3 重新下载数据。"
        )
    else:
        with zipfile.ZipFile(meter_data_zip_path, "r") as zip_ref:
            zip_ref.extractall("inputs")
        print("已解压!")


def produce_all_properties_df():
    """处理原始数据（拆分为子文件夹）并保存为单个 CSV 文件。"""
    if not os.path.isdir(meter_data_folder_path):
        print("未找到解压缩的文件夹。 解压缩")
        unzip_raw_data()

    halfhourly_dataset = pd.DataFrame()

    print("处理数据...")
    folder_names = os.listdir(meter_data_folder_path)
    for file_name in tqdm.tqdm(folder_names):
        df_temp = pd.read_csv(
            os.path.join(meter_data_folder_path, file_name),
            index_col="tstp",
            parse_dates=True,
            low_memory=False,
        )
        df_temp["file_name"] = file_name.split(".")[0]
        df_temp = df_temp.replace("Null", np.nan).dropna()
        df_temp["energy(kWh/hh)"] = df_temp["energy(kWh/hh)"].astype("float")
        halfhourly_dataset = pd.concat([halfhourly_dataset, df_temp])

    # 构造数据框，使索引是时间戳，列是家庭（最初在 LCLid 变量中）
    df_output = (
        halfhourly_dataset.groupby(["tstp", "LCLid"])["energy(kWh/hh)"]
        .mean(numeric_only=True)
        .unstack()
    )
    if not os.path.isdir(meter_data_merged_folder_path):
        os.makedirs(meter_data_merged_folder_path)

    df_output.to_csv(meter_data_merged_file_path)


if __name__ == "__main__":
    produce_all_properties_df()
```

### 创造季节特征

```python
smart_meter_train = smart_meter_train.drop(columns = ["Month", "Day", "Weekday", "Hour"])
```

```python
def create_season_feature(df):
    # 创建一个系列日期，其中包含DateTime日期的数字表示 (e.g. 春季的开始日期, 3月21 = 321)
    date = df.DateTime.dt.month*100 + df.DateTime.dt.day

    # 通过将日期值放入bins来分配季节特征
    # 由于pd.cut只采用独特的bin标签，而冬季月份跨越年底和年初，因此在“winter”之后添加了一个空格，之后添加了strip()

    df['Season'] = pd.cut(date,[0,321,620,922,1220,1300],
                       labels=["Winter","Spring","Summer","Autumn","Winter "]).str.strip()
    return df
```

```python
smart_meter_train = create_season_feature(smart_meter_train)
```

###  创造节假日特征

```python
def create_day_type_feature(df):
    
    # 通过将星期几值放入 bin 来分配 day_type 特征

    df["Day_type"] = pd.cut(df["DateTime"].dt.weekday,[-1,4,5, 6],
                       labels=["Weekday","Day before holiday","Holiday"])
    
    holiday_list = [datetime.datetime(2013, 1, 1), 
                   datetime.datetime(2013, 3, 29), 
                   datetime.datetime(2013, 4, 1), 
                   datetime.datetime(2013, 5, 6), 
                   datetime.datetime(2013, 8, 26), 
                   datetime.datetime(2013, 12, 25), 
                   datetime.datetime(2013, 12, 26)]
    
    day_before_holiday_list = [datetime.datetime(2013, 3, 28),
                              datetime.datetime(2013, 12, 24)]
    
    # 英国 2013 日历中的特定银行假期
    df.loc[df["DateTime"].isin(holiday_list), "Day_type"] = "Holiday"
    df.loc[df["DateTime"].isin(day_before_holiday_list), "Day_type"] = "Day before holiday"
    

    return df
```

###  给一天划分

```python
def create_time_slot_feature(df):

    df["Time_slot"] = pd.cut(df["DateTime"].dt.hour,[-1,3, 6, 11, 14, 17, 20, 23],
                       labels=["Midnight", 
                               "Early morning", 
                               "Morning", 
                               "Early afternoon", 
                               "Late afternoon", 
                               "Early evening", 
                               "Late evening"])
    return df
```

## 转换管道

```python
# 移除不必要的cols
smart_meter = smart_meter_train.drop(columns = ["Consumption", "LCLid", "DateTime", "Acorn"])
```

###  分类数据编码

```python
from sklearn.preprocessing import OneHotEncoder
cat_attribs = ["Season", "Day_type", "Time_slot","Acorn_grouped"]

# 从智能电表数据库中提取分类列
smart_meter_cat = smart_meter[cat_attribs]
# 对分类数据执行分类编码
cat_encoder = OneHotEncoder()
smart_meter_cat_onehot = cat_encoder.fit_transform(smart_meter_cat)
```

### 数值数据缺失数据插补

```python
from sklearn.impute import SimpleImputer
# 从智能电表数据库中提取数值列
smart_meter_num = smart_meter.drop(columns = cat_attribs)
imputer = SimpleImputer(strategy = "median")
smart_meter_num_impute = imputer.fit_transform(smart_meter_num)
```

###  数值转换

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#为数值列创建转换管道
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),  
    ("std_scaler", StandardScaler()),
    ])
# 创建 num_attribs 作为用于转换的数字属性列表
num_attribs = list(smart_meter_num)
```

```python
from sklearn.compose import ColumnTransformer
# 创建完整的数据转换管道
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs), 
    ("cat", OneHotEncoder(), cat_attribs), 
    ])
# 将转换管道应用于数据集
smart_meter_prepared = full_pipeline.fit_transform(smart_meter)
```

# Model selection

```python
smart_meter_labels = smart_meter_train["Consumption"].copy()
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
forest_reg = RandomForestRegressor()
forest_reg.fit(smart_meter_prepared,smart_meter_labels)
smart_meter_predictions = forest_reg.predict(smart_meter_prepared)
forest_mse = mean_squared_error(smart_meter_labels, smart_meter_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
# 获得重要性
importance = forest_reg.feature_importances_
# 总结特征的重要性
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
```

