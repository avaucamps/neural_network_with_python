import numpy as np

def categorical_crossentropy_loss(predictions, desired_predictions):
    return -np.sum(desired_predictions * np.log(predictions))


def quadratic_loss(predictions, desired_predictions):
    return (1/2) * np.sum(np.sqrt(np.abs(desired_predictions - predictions)))