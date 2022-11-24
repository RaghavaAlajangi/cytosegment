import numpy as np

from unet.ml_criterions import FocalTverskyLoss

from .helper_methods import get_test_tensors


def test_focaltverskyloss():
    alpha = 0.3
    beta = 0.7
    gamma = 0.75
    criterion = FocalTverskyLoss(alpha, beta, gamma)
    predict, target = get_test_tensors()
    test_loss = criterion(predict, target)
    assert np.allclose(test_loss, 0.2543, atol=0.3, rtol=0)
