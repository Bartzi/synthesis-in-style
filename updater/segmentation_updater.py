import torch
from torch import nn

from pytorch_training import Updater
from pytorch_training.reporter import get_current_reporter
from pytorch_training.updater import GradientApplier


class SegmentationUpdater(Updater):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss()

    def update_core(self):
        batch = next(self.iterators['images'])
        batch = {key: value.to(self.device) for key, value in batch.items()}
        reporter = get_current_reporter()

        network = self.networks['segmentation']

        with GradientApplier([network], [self.optimizers['main']]):
            segmentation_prediction = network(batch['images'])
            batch_size, num_classes, height, width = segmentation_prediction.shape
            segmentation_prediction = segmentation_prediction.permute(0, 2, 3, 1)
            segmentation_prediction = torch.reshape(segmentation_prediction, (batch_size * height * width, num_classes))

            label_image = batch['segmented']
            label_image = label_image.permute(0, 2, 3, 1)
            label_image = label_image.reshape((-1,))
            loss = self.loss(segmentation_prediction, label_image)

            loss.backward()
        reporter.add_observation({"softmax": loss}, 'loss')
