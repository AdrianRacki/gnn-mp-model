import torch
from kedro.io import AbstractDataset


class GraphDataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self):
        loader = torch.load(self._filepath)
        return loader

    def _save(self, dataloader):
        torch.save(dataloader, self._filepath)

    def _describe(self):
        return dict(filepath=self._filepath)
