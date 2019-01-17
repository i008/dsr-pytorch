from torch.utils.data import DataLoader, Dataset
import PIL
from skimage import io
import numpy as np

class NeuronSegmDataset(Dataset):
    def __init__(self, neuron_train_path, neuron_target_path, image_transform=None, augmenter=None):
        self.target_image = io.imread(neuron_train_path)
        self.target = io.imread(neuron_target_path)
        self.image_transform = image_transform
        self.augmenter = augmenter

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, ix):
        X = self._load_image(ix)
        y = self._load_target(ix)

        if self.augmenter:
            X, y = self.do_augment(X, y)
        if self.image_transform is not None:
            X = self.image_transform(X)
            y = self.image_transform(y)

        return X, y

    def _load_image(self, ix):
        return PIL.Image.fromarray(self.target_image[ix])

    def _load_target(self, ix):
        return PIL.Image.fromarray(self.target[ix])

    def collate_func(self, batch):
        pass

    def do_augment(self, X, y):
        X = np.array(X)
        y = np.array(y)
        res = self.augmenter(image=X, mask=y)

        return PIL.Image.fromarray(res['image']), PIL.Image.fromarray(res['mask'])
