# -*- coding:utf-8 -*-
import numpy as np
from ..utils.fixes import bincount

# 计算权重，权重 = 各个值出现的频率 * 最小值
def _balance_weights(y):
    """Compute sample weights such that the class distribution of y becomes
       balanced.

    Parameters
    ----------
    y : array-like
        Labels for the samples.

    Returns
    -------
    weights : array-like
        The sample weights.

    y = [2,2,3,5,2,10,20]
    np.asarray(y)
    #  array([ 2,  2,  3,  5,  2, 10, 20])

    y = np.searchsorted(np.unique(y),y)
    # array([0, 0, 1, 2, 0, 3, 4])

    bins
    # array([3, 1, 1, 1, 1])

    bins.take(y)
    # array([3, 3, 1, 1, 3, 1, 1])

    """
    y = np.asarray(y)
    y = np.searchsorted(np.unique(y), y)
    bins = bincount(y)

    weights = 1. / bins.take(y)
    weights *= bins.min()

    return weights
