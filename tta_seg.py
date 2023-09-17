import numpy as np


class tta_seg():
    def __init__(self, model):
        self.model = model

    def predict(self, samples):
        pred = []
        for x in samples:
            x = x.reshape(1, *x.shape)
            orig = self.model.predict(x)
            lr = self.model.predict(np.fliplr(x))
            ud = self.model.predict(np.flipud(x))
            lr_ud = self.model.predict(np.fliplr(np.flipud(x)))
            avg_pred = (orig + np.fliplr(lr) + np.flipud(ud) + np.fliplr(np.flipud(lr_ud))) / 4
            pred.append(avg_pred)
        return np.array(pred)
