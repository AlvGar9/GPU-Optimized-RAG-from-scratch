import numpy as np
from src.data_processing.distances import l2_numpy, cosine_numpy

def test_l2():
    x = np.array([0.,3.,4.])
    y = np.zeros(3)
    assert l2_numpy(x,y) == 5.0

def test_cosine():
    x = np.array([1.,0.])
    y = np.array([0.,1.])
    assert pytest.approx(cosine_numpy(x,y)) == 1.0
