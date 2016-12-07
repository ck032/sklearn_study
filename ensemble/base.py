# -*- coding:utf-8 -*-
# 所有集成学习器的基础类

"""
Base class for ensemble-based estimators.
"""

# Authors: Gilles Louppe
# License: BSD 3 clause

import numpy as np

from ..base import clone
from ..base import BaseEstimator
from ..base import MetaEstimatorMixin
from ..utils import _get_n_jobs

# base_estimator 基学习器
# n_estimators 学习器的个数（比如，多少颗树）
# estimator_params 学习器的参数，定义为tuple()
# estimators_ 定义为list

class BaseEnsemble(BaseEstimator, MetaEstimatorMixin):
    """Base class for all ensemble classes.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the ensemble is built.

    n_estimators : integer
        The number of estimators in the ensemble.

    estimator_params : list of strings
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    base_estimator_ : list of estimators
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.
    """

    def __init__(self, base_estimator, n_estimators=10,
                 estimator_params=tuple()):
        # Set parameters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

        # Don't instantiate estimators now! Parameters of base_estimator might
        # still change. Eg., when grid-searching with the nested object syntax.
        # This needs to be filled by the derived classes.
        self.estimators_ = []

    # 验证参数
    def _validate_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""

        # 如果 学习器个数n_estimators <= 0，报错
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        # 如果没有指定基学习器base_estimator，那么采用默认
        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        # 必须指定基学习器
        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    # 把学习器的参数estimator_params，应用到基学习器base_estimator上，得到estimator
    # append 是创建多个学习器
    # p 是参数， self指的是基学习器
    # 注意学习getattr是获取参数的具体值，set_params是应用参数
    # 另外，**dict() 来创建字典，也是很好的方式

    def _make_estimator(self, append=True):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator_)
        estimator.set_params(**dict((p, getattr(self, p))
                                    for p in self.estimator_params))

        if append:
            self.estimators_.append(estimator)

        return estimator

    # 学习器的个数
    def __len__(self):
        """Returns the number of estimators in the ensemble."""
        return len(self.estimators_)

    # 第几个学习器
    def __getitem__(self, index):
        """Returns the index'th estimator in the ensemble."""
        return self.estimators_[index]

    # 生成迭代器
    def __iter__(self):
        """Returns iterator over estimators in the ensemble."""
        return iter(self.estimators_)

# 把不同的学习器estimators往不同的作业（jobs)分发
# 一个作业，可能会分到多个estimators
def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs

    # 获取作业数、学习器个数的最小值
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()
