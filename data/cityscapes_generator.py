import os
from collections import OrderedDict
import numpy as np
from keras.utils import Sequence, to_categorical
from . import utils


class CityscapesGenerator(Sequence):
    """Cityscapes dataset https://www.cityscapes-dataset.com/.

    Args:
        root_dir (string): Root directory path.
        batch_size(int): The batch size.
        shape (tuple): The requested size in pixels, as a 2-tuple:
            (width,height).
        mode (string): The type of dataset: 'train' for training set, 'val'
            for validation set, and 'test' for test set.

    """

    # Training dataset root folders
    train_folder = "leftImg8bit_trainvaltest/leftImg8bit/train"
    train_lbl_folder = "gtFine_trainvaltest/gtFine/train"

    # Validation dataset root folders
    val_folder = "leftImg8bit_trainvaltest/leftImg8bit/val"
    val_lbl_folder = "gtFine_trainvaltest/gtFine/val"

    # Test dataset root folders
    test_folder = "leftImg8bit_trainvaltest/leftImg8bit/test"
    test_lbl_folder = "gtFine_trainvaltest/gtFine/test"

    # Filters to find the images
    img_extension = '.png'
    lbl_name_filter = 'labelIds'

    # The values associated with the 35 classes
    full_classes = (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, -1
    )
    # The values above are remapped to the following
    new_classes = (
        0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 0, 0, 0, 6, 0, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 0, 0, 17, 18, 19, 0
    )

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
        ('Unlabeled', (0, 0, 0)),
        ('Road', (128, 64, 128)),
        ('Sidewalk', (244, 35, 232)),
        ('Building', (70, 70, 70)),
        ('Wall', (102, 102, 156)),
        ('Fence', (190, 153, 153)),
        ('Pole', (153, 153, 153)),
        ('Traffic_light', (250, 170, 30)),
        ('Traffic_sign', (220, 220, 0)),
        ('Vegetation', (107, 142, 35)),
        ('Terrain', (152, 251, 152)),
        ('Sky', (70, 130, 180)),
        ('Person', (220, 20, 60)),
        ('Rider', (255, 0, 0)),
        ('Car', (0, 0, 142)),
        ('Truck', (0, 0, 70)),
        ('Bus', (0, 60, 100)),
        ('Train', (0, 80, 100)),
        ('Motorcycle', (0, 0, 230)),
        ('Bicycle', (119, 11, 32))
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
                name_filter=self.lbl_name_filter,
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
                name_filter=self.lbl_name_filter,
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
                name_filter=self.lbl_name_filter,
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
            image = np.asarray(image)
            label = np.asarray(label)

            # Expand the channels dimension if there isn't one
            if np.ndim(image) == 2:
                image = np.expand_dims(image, -1)
            if np.ndim(label) == 2:
                label = np.expand_dims(label, -1)

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

        # Remap class labels
        label_batch = utils.remap(label_batch, self.full_classes, self.new_classes)  # yapf: disable

        # Change format from class integers to categorical
        num_classes = len(self.color_encoding)
        label_batch = to_categorical(label_batch, num_classes)

        return image_batch, label_batch

    def __len__(self):
        """Returns the number of batch sizes in this dataset.

        Returns:
            int: number of batch sizes in this dataset.

        """
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
