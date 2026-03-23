import numpy as np
import multiprocessing
import time


def cpr(x, y):
    # x(N), y(N)
    sorted_index = np.argsort(x)[::-1]
    y = y[sorted_index]
    precisions = []
    recalls = []
    p_num = np.sum((y == 1.).astype(np.float32))
    for i in range(1, y.shape[0] + 1):
        tp = np.sum((y[:i] == 1.).astype(np.float32))
        fp = np.sum((y[:i] == 0.).astype(np.float32))
        p = tp / (tp + fp)
        r = tp / p_num
        precisions.append(p)
        recalls.append(r)
    return precisions, recalls


def capvoc(x, y):
    precision, recall = cpr(x, y)

    precision = np.concatenate(([0.], precision, [0.]), axis=0)
    recall = np.concatenate(([0.], recall, [1.]), axis=0)

    for index in range(precision.shape[0] - 1, 0, -1):
        precision[index - 1] = np.maximum(precision[index - 1], precision[index])

    index = np.where(recall[1:] != recall[:-1])[0]

    return np.sum((recall[index + 1] - recall[index]) * precision[index + 1])


def cmapvoc(x, y):
    # x(N, C), y(N, C)
    c = x.shape[1]
    aps = []
    pool = multiprocessing.Pool()
    for i in range(c):
        cur_x = x[:, i]
        cur_y = y[:, i]

        res = pool.apply_async(func=capvoc, args=(cur_x, cur_y))
        aps.append(res)
    pool.close()
    pool.join()
    aps = np.asarray([float(_.get()) for _ in aps], dtype=np.float32)

    return aps


class CmapvocPool:
    def __init__(self):
        self.pool = multiprocessing.Pool()
        self.result = []

    def put_job(self, x, y):
        c = x.shape[1]
        for i in range(c):
            cur_x = x[:, i]
            cur_y = y[:, i]

            res = self.pool.apply_async(func=capvoc, args=(cur_x, cur_y))
            self.result.append(res)
        return

    def get_result(self):
        self.pool.close()
        self.pool.join()
        aps = np.asarray([float(_.get()) for _ in self.result], dtype=np.float32)
        return aps
