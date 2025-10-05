""" Miscellaneous Functions """


from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from collections import defaultdict
from time import time
from typing import Dict, Any
    

class MetricsLogger(object):
    def __init__(self, folder: str = './logs'):
        self.writer = SummaryWriter(folder)
        self.cache = defaultdict(list)
        self.last_step = time()

    def add_scalar(self, tag: str, value: Any, step: int = None, wall_time: int = None):
        self.writer.add_scalar(tag, value, step, wall_time)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

    def step(self, metrics: dict, epoch: int, batch: int):
        elapse = time() - self.last_step
        print(f'Elapse: {elapse: .4f}s, {1 / elapse: .2f} steps/sec')
        self.last_step = time()

        for key in metrics:
            self.cache[key].append(metrics[key].cpu().item())

    def epoch_end(self, epoch: int):
        losses = []
        for key in self.cache:
            avg = np.mean(self.cache[key])
            if 'loss' in key:
                losses.append(avg)
            self.writer.add_scalar(f'Average/{key}', avg, epoch)

        avg = np.mean(losses)
        self.writer.add_scalar('Average/loss', avg, epoch)
        self.cache.clear()


def log_metrics(metrics: Dict[str, Tensor]):
    log = '[ '
    log += ' ] [ '.join([f'{k} = {v.cpu().item():.4f}' for k, v in metrics.items()])
    log += ' ]'
    print(log)


def cast_to_float(x):
    if isinstance(x, list):
        return [cast_to_float(y) for y in x]
    elif isinstance(x, dict):
        return {k: cast_to_float(v) for k, v in x.items()}
    return x.float()