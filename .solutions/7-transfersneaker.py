
import PIL
from PIL import Image
import io
import requests
import torch
from torch.utils.data import Sampler


class OneClassImageClassificationDataset(Dataset):
    def __init__(self, annotations, image_transform):
        """
        annotations is a pandas dataframe
        
        """
        super().__init__()
        self.annotations = annotations
        self.image_transform = image_transform

    def __len__(self):
        """
        Return the length of the annotations dataframe
        """
        # your code here
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Using methods you wrote:
        1 - load image from disk for given index  (self.load_from_disk)
        2 - transform image (self.image_transform)
        3 - Load target (self.load_target)
        return Xi, yi
        """
        
        # YOUR CODE HERE

        Xi = self.load_from_disk(index)
        Xi = self.image_transform(Xi)
        yi = self.load_target(index)
        return Xi, yi

    def load_to_pil(self, uri):
        """
        Write a helper function that uses PIL.Image to load a file and returns it
        """

        image_pil = Image.open(uri)
        image_pil = image_pil.convert("RGB")
        # image_pil = YOUR CODE HERE
        return image_pil


    def load_from_disk(self, index):
        """
        Loads an image from disk given a index.
        It gets the path of an image with the corresponding index from the metadata 
        It passes the URI to the self.load_to_pil and returns a PIL.Image
        """
        image_path = self.annotations.iloc[index]['image_path']
        #image_path = # YOUR CODE HERE
        return self.load_to_pil(image_path)

    def load_target(self, index):
        """
        This function should get the tag for a given index from the annotations dataframe
        You .iloc can become useful.    
        This methods should return, either a 0 or a 1.
        """
        
        #label = # YOUR CODE HERE
        label = self.annotations.iloc[index]['tags']

        return label
    
    
class BaseSampler(Sampler):
    def __init__(self, df, n_samples):
        self.df = df
        self.n_samples = n_samples
        
    def __iter__(self):
        return iter(self._get_sample())
        
    def __len__(self):
        return self.n_samples
    
    def _get_sample(self):
        return np.random.choice(len(self.df), self.n_samples, replace=False)
        

def binary_classification_model():
    """
    Write a function that loads a resnet50 model from pretrainedmodels, freezes its layers
    replaces the last_linear with the proper output number. As we did in previous example.
    replace avgpool with adaptiv pooling.
    """
    model = resnet50()
    for p in model.parameters():
        p.requires_grad = False
    inft = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features=inft, out_features=2)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    
    # model = YOUR CODE HERE
    return model


from imgaug import augmenters as iaa

aug_seq = iaa.Sequential([
    iaa.Fliplr(p=0.5),
    iaa.Sometimes(
        0.3,
        iaa.Multiply((0.9, 1.2))
    ),
    iaa.Sometimes(
        0.3,
        iaa.AdditiveGaussianNoise()
    ),
    iaa.Affine(
        scale=(0.5, 2),
        translate_percent=(-0.2, 0.2)
    )
])
def augment(self, augmenter, image):
    augmenter = augmenter.to_deterministic()
    img_aug = augmenter.augment_image(np.array(image))
    img_aug = Image.fromarray(img_aug)
    return img_aug