"""
chumpy の基本機能を模倣するスタブライブラリ
"""

import numpy as np
from scipy import sparse
import sys


class ch_array:
    """chumpy array のスタブクラス"""
    
    def __init__(self, x):
        if isinstance(x, np.ndarray):
            self.r = x
        else:
            self.r = np.array(x)
    
    def __array__(self):
        return self.r
    
    @property
    def shape(self):
        return self.r.shape
    
    def __getitem__(self, key):
        return self.r[key]


def array(x):
    """numpy配列をchumpy風に包む"""
    return ch_array(x)


# scipyのsparse行列も同様に処理
def csc_matrix(data, shape=None):
    """CSC行列のスタブ"""
    return sparse.csc_matrix(data, shape=shape)


def csr_matrix(data, shape=None):
    """CSR行列のスタブ"""
    return sparse.csr_matrix(data, shape=shape)


# chumpy の Ch クラスのスタブ
class Ch(ch_array):
    """chumpy の Ch クラスのスタブ"""
    
    def __init__(self, x):
        super().__init__(x)


# chumpy.ch モジュールのスタブ
class ChModule:
    """chumpy.ch モジュールのスタブ"""
    
    Ch = Ch
    
    @staticmethod
    def array(x):
        return ch_array(x)
    
    @staticmethod
    def zeros(shape):
        return ch_array(np.zeros(shape))
    
    @staticmethod
    def ones(shape):
        return ch_array(np.ones(shape))


# メインモジュールレベルでChクラスを追加
globals()['Ch'] = Ch

# サブモジュールの設定
ch = ChModule()
sys.modules['chumpy.ch'] = ch