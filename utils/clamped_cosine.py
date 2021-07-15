from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from training_tools.pytorch_training.extensions.lr_scheduler import LRScheduler
from training_tools.pytorch_training.triggers import get_trigger


class ClampedCosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, max_update, eta_min=0, last_epoch=-1, verbose=False):
        self.max_update = max_update
        super(ClampedCosineAnnealingLR, self).__init__(optimizer, T_max, eta_min, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch > self.max_update:
            return [self.eta_min for group in self.optimizer.param_groups]
        else:
            return super().get_lr()


if __name__ == '__main__':

    segmentation_network = nn.Linear(10, 10)
    optimizer_opts = {
        'betas': (0.5, 0.99),
        'lr': 0.001,
    }
    optimizer = Adam(segmentation_network.parameters(), **optimizer_opts)
    schedulers = dict(encoder=ClampedCosineAnnealingLR(optimizer, 10, eta_min=0.00001, max_update=5))
    lr_scheduler = LRScheduler(schedulers, trigger=get_trigger((1, 'iteration')))
