from typing import Tuple, Dict, Optional, List
import os
import torch
import numpy as np

from ..encoding import Encoder, NullEncoder

from torchvision import transforms

import warnings


class SplitMNIST(torch.utils.data.Dataset):
    # language=rst
    """
    Handles loading of a split MNIST where you can either
    load digits 0-4 or 5-9
 
    """

    def __init__(
        self,
        path: str,
        shuffle: bool = True,
        train: bool = True,
        num_samples: int = -1,
        selection: int = 1,
        transform: Optional[List]= None,
        image_encoder: Optional[Encoder] = None,
        label_encoder: Optional[Encoder] = None,
    ) -> None:
        # language=rst
        """
        Constructor for the ``SplitMNIST`` object.
        :param shuffle: Whether to randomly permute order of dataset.
        :param train: Load training split if true else load test split
        :param num_samples: Number of samples to pass to the batch
        :param selection: Which half of digits to select; 1 for 0-4; 2 for 5-9
        """
        super().__init__()

        if not os.path.isdir(path):
            print("Path does not exist!")
            exit(1)

        self.path = path
        self.shuffle = shuffle
        self.num_samples = num_samples
        self.selection = selection
        self.transform = transform

        # Allow the passthrough of None, but change to NullEncoder
        if image_encoder is None:
            image_encoder = NullEncoder()

        if label_encoder is None:
            label_encoder = NullEncoder()

        self.image_encoder = image_encoder
        self.label_encoder = label_encoder

        if train:
            self.images, self.labels = self._get_train()
        else:
            self.images, self.labels = self._get_test()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ind)-> Dict[str, torch.Tensor]:
        image = self.images[ind]
        label = self.labels[ind]

        image = np.expand_dims(image, axis=0)

        if self.transform is not None:
            image = self.transform(image)
 
        image = torch.from_numpy(image)

        output = {
            "image": image,
            "label": label,
            "encoded_image": self.image_encoder(image),
            "encoded_label": self.label_encoder(label),
        }

        return output

    def _get_train(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # language=rst
        """
        Gets the MNIST training images and labels.

        :return: Spoken MNIST training images and labels.
        """

        # Load the data from disk as numpy arrays
        if self.selection == 1:
            images = np.load(self.path + "/train/digits_0_4_images.npy")
            labels = np.load(self.path + "/train/digits_0_4_labels.npy")
        else:
            images = np.load(self.path + "/train/digits_5_9_images.npy")
            labels = np.load(self.path + "/train/digits_5_9_labels.npy")


        if self.shuffle:
            perm = np.random.permutation(np.arange(labels.shape[0]))
            images = images[perm]
            labels = labels[perm]

        return torch.from_numpy(images), labels

    def _get_test(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # language=rst
        """
        Gets the MNIST test images and labels.

        :return: The MNIST test images and labels.
        """
        # Load the data from disk as numpy arrays
        if self.selection == 1:
            images = np.load(self.path + "/test/digits_0_4_images.npy")
            labels = np.load(self.path + "/test/digits_0_4_labels.npy")
        else:
            images = np.load(self.path + "/test/digits_5_9_images.npy")
            labels = np.load(self.path + "/test/digits_5_9_labels.npy")


        if self.shuffle:
            perm = np.random.permutation(np.arange(labels.shape[0]))
            images = images[perm]
            labels = labels[perm]

        return torch.from_numpy(images), labels


