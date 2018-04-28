import os
from collections import OrderedDict
import numpy as np
from keras.utils import Sequence, to_categorical
from . import utils


class CamVidGenerator(Sequence):
    """CamVid dataset generator.

    The dataset must be arranged as in https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.

    Args:
        root_dir (string): Root directory path.
        batch_size(int): The batch size.
        shape (tuple): The requested size in pixels, as a 2-tuple:
            (width,height).
        mode (string): The type of dataset: 'train' for training set, 'val'
            for validation set, and 'test' for test set.

    """

    # Training dataset root folders
    train_folder = 'train'
    train_lbl_folder = 'trainannot'

    # Validation dataset root folders
    val_folder = 'val'
    val_lbl_folder = 'valannot'

    # Test dataset root folders
    test_folder = 'test'
    test_lbl_folder = 'testannot'

    # Images extension
    img_extension = '.png'

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
        ('sky', (128, 128, 128)),
        ('building', (128, 0, 0)),
        ('pole', (192, 192, 128)),
        ('road', (128, 64, 128)),
        ('pavement', (60, 40, 222)),
        ('tree', (128, 128, 0)),
        ('sign_symbol', (192, 128, 128)),
        ('fence', (64, 64, 128)),
        ('car', (64, 0, 128)),
        ('pedestrian', (64, 64, 0)),
        ('bicyclist', (0, 128, 192)),
        ('unlabelled', (0, 0, 0))
    ])  # yapf: disable

    def __init__(self, root_dir, batch_size, shape=None, mode='train'):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shape = shape
        self.mode = mode

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_images = utils.get_files(
                os.path.join(root_dir, self.train_folder),
                extension_filter=self.img_extension
            )

            self.train_labels = utils.get_files(
                os.path.join(root_dir, self.train_lbl_folder),
                extension_filter=self.img_extension
            )
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_images = utils.get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.img_extension
            )

            self.val_labels = utils.get_files(
                os.path.join(root_dir, self.val_lbl_folder),
                extension_filter=self.img_extension
            )
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_images = utils.get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.img_extension
            )

            self.test_labels = utils.get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                extension_filter=self.img_extension
            )
        else:
            raise RuntimeError(
                "Unexpected dataset mode. "
                "Supported modes are: train, val and test"
            )

    def __getitem__(self, index):
        """Gets a full batch of data.

        Args:
            index (int): index of the batch size to return.

        Returns:
        A tuple of ``numpy.array`` (image_batch, label_batch) where
        image_batch is a batch of images from tis dataset and label_batch
        are the corresponding ground-truth labels in categorical format.

        """
        # Create the variables that will contain the batch to return
        image_batch = None
        label_batch = None

        # Fill image_paths and label_paths with a batch size of image paths
        if self.mode.lower() == 'train':
            image_paths = self.train_images[index * self.batch_size:
                                            (index + 1) * self.batch_size]
            label_paths = self.train_labels[index * self.batch_size:
                                            (index + 1) * self.batch_size]
        elif self.mode.lower() == 'val':
            image_paths = self.val_images[index * self.batch_size:
                                          (index + 1) * self.batch_size]
            label_paths = self.val_labels[index * self.batch_size:
                                          (index + 1) * self.batch_size]
        elif self.mode.lower() == 'test':
            image_paths = self.test_images[index * self.batch_size:
                                           (index + 1) * self.batch_size]
            label_paths = self.test_labels[index * self.batch_size:
                                           (index + 1) * self.batch_size]
        else:
            raise RuntimeError(
                "Unexpected dataset mode. "
                "Supported modes are: train, val and test"
            )

        # Load the batch size to PIL images and convert them to numpy arrays.
        # Labels are converted to a binary class matrix using to_categorical.
        for idx, image_path in enumerate(image_paths):
            image, label = utils.pil_loader(
                image_path,
                label_paths[idx],
                self.shape,
            )

            # PIL to numpy
            # TODO: load images straight to numpy instead of PIL
            image = np.asarray(image)

            # Change format from class integers to categorical
            num_classes = len(self.color_encoding)
            label = to_categorical(np.asarray(label), num_classes)

            # Initialize image_batch and label_batch if needed
            if image_batch is None:
                image_batch = np.empty(
                    (self.batch_size, ) + image.shape, dtype=np.uint8
                )
            if label_batch is None:
                label_batch = np.empty(
                    (self.batch_size, ) + label.shape, dtype=np.uint8
                )

            # Fill image_batch and label_batch iteratively
            image_batch[idx] = image.astype(np.uint8)
            label_batch[idx] = label.astype(np.uint8)

        return image_batch, label_batch

    def __len__(self):
        """Returns the number of batch sizes in this dataset."""
        if self.mode.lower() == 'train':
            return int(
                np.ceil(len(self.train_images) / float(self.batch_size))
            )
        elif self.mode.lower() == 'val':
            return int(np.ceil(len(self.val_images) / float(self.batch_size)))
        elif self.mode.lower() == 'test':
            return int(np.ceil(len(self.test_images) / float(self.batch_size)))
        else:
            raise RuntimeError(
                "Unexpected dataset mode. "
                "Supported modes are: train, val and test"
            )
