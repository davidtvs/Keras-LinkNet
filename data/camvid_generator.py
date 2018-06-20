import os
from collections import OrderedDict
import numpy as np
from keras.utils import Sequence, to_categorical
from . import utils


class CamVidGenerator(Sequence):
    """CamVid dataset generator.

    Args:
        root_dir (string): Root directory path.
        batch_size(int): The batch size.
        shape (tuple): The requested size in pixels, as a 2-tuple:
            (width,height).
        mode (string): The type of dataset: 'train' for training set, 'val'
            for validation set, and 'test' for test set.

    """

    # Training dataset root folders
    samples_folder = '701_StillsRaw_full'
    labels_folder = 'LabeledApproved_full'

    # Image names files for each dataset split (same as SegNet)
    train_names_file = 'segnet_train.txt'
    val_names_file = 'segnet_val.txt'
    test_names_file = 'segnet_test.txt'

    # Label name suffix
    label_suffix = '_L'

    # Images extension
    img_extension = '.png'

    # The values associated with the 35 classes
    full_classes = (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    )
    # The values above are remapped to the following
    new_classes = (
        0, 2, 8, 6, 6, 8, 5, 5, 3, 3, 2, 7, 7, 7, 1, 2, 2, 4, 4, 4, 4, 0, 10,
        10, 10, 11, 11, 9, 9, 9, 9, 9
    )

    _color_encoding32 = OrderedDict([
        ('Unlabeled', (0, 0, 0)),
        ('Building', (128, 0, 0)),
        ('Wall', (64, 192, 0)),
        ('Tree', (128, 128, 0)),
        ('VegetationMisc', (192, 192, 0)),
        ('Fence', (64, 64, 128)),
        ('Sidewalk', (0, 0, 192)),
        ('ParkingBlock', (64, 192, 128)),
        ('Column_Pole', (192, 192, 128)),
        ('TrafficCone', (0, 0, 64)),
        ('Bridge', (0, 128, 64)),
        ('SignSymbol', (192, 128, 128)),
        ('Misc_Text', (128, 128, 64)),
        ('TrafficLight', (0, 64, 64)),
        ('Sky', (128, 128, 128)),
        ('Tunnel', (64, 0, 64)),
        ('Archway', (192, 0, 128)),
        ('Road', (128, 64, 128)),
        ('RoadShoulder', (128, 128, 192)),
        ('LaneMkgsDriv', (128, 0, 192)),
        ('LaneMkgsNonDriv', (192, 0, 64)),
        ('Animal', (64, 128, 64)),
        ('Pedestrian', (64, 64, 0)),
        ('Child', (192, 128, 64)),
        ('CartLuggagePram', (64, 0, 192)),
        ('Bicyclist', (0, 128, 192)),
        ('MotorcycleScooter', (192, 0, 192)),
        ('Car', (64, 0, 128)),
        ('SUVPickupTruck', (64, 128, 192)),
        ('Truck_Bus', (192, 128, 192)),
        ('Train', (192, 64, 128)),
        ('OtherMoving', (128, 64, 64))
    ])  # yapf: disable

    # Default encoding for pixel value, class name, and class color
    _color_encoding12 = OrderedDict([
        ('Unlabeled', (0, 0, 0)),
        ('Sky', (128, 128, 128)),
        ('Building', (128, 0, 0)),
        ('Pole', (192, 192, 128)),
        ('Road', (128, 64, 128)),
        ('Pavement', (60, 40, 222)),
        ('Tree', (128, 128, 0)),
        ('SignSymbol', (192, 128, 128)),
        ('Fence', (64, 64, 128)),
        ('Car', (64, 0, 128)),
        ('Pedestrian', (64, 64, 0)),
        ('Bicyclist', (0, 128, 192))
    ])  # yapf: disable

    def __init__(
        self,
        root_dir,
        batch_size,
        shape=None,
        mode='train'
    ):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shape = shape
        self.mode = mode
        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []
        self.test_images = []
        self.test_labels = []

        samples_folder = os.path.join(root_dir, self.samples_folder)
        labels_folder = os.path.join(root_dir, self.labels_folder)

        with open(os.path.join(root_dir, self.train_names_file)) as f:
            train_names = f.read().splitlines()
        with open(os.path.join(root_dir, self.val_names_file)) as f:
            val_names = f.read().splitlines()
        with open(os.path.join(root_dir, self.test_names_file)) as f:
            test_names = f.read().splitlines()

        if self.mode.lower() == 'train':
            for filename in train_names:
                # Add the path to the sample to the list
                self.train_images.append(
                    os.path.join(samples_folder, filename)
                )

                # The labels share most of the corresponding sample name, but
                # they have a suffix. So before adding to the list we have to
                # account for that
                name, ext = os.path.splitext(filename)
                label_filename = name + self.label_suffix + ext
                self.train_labels.append(
                    os.path.join(labels_folder, label_filename)
                )
        elif self.mode.lower() == 'val':
            for filename in val_names:
                # Add the path to the sample to the list
                self.val_images.append(os.path.join(samples_folder, filename))

                # The labels share most of the corresponding sample name, but
                # they have a suffix. So before adding to the list we have to
                # account for that
                name, ext = os.path.splitext(filename)
                label_filename = name + self.label_suffix + ext
                self.val_labels.append(
                    os.path.join(labels_folder, label_filename)
                )
        elif self.mode.lower() == 'test':
            for filename in test_names:
                # Add the path to the sample to the list
                self.test_images.append(os.path.join(samples_folder, filename))

                # The labels share most of the corresponding sample name, but
                # they have a suffix. So before adding to the list we have to
                # account for that
                name, ext = os.path.splitext(filename)
                label_filename = name + self.label_suffix + ext
                self.test_labels.append(
                    os.path.join(labels_folder, label_filename)
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

        # Convert 32-class labels from RGB to categorical format
        label_batch = utils.rgb_to_categorical(label_batch, self._color_encoding32)  # yapf: disable

        # Remap class labels; returns label in categorical format
        label_batch = utils.remap(label_batch, self.full_classes, self.new_classes)  # yapf: disable

        # Change format from class integers to categorical
        num_classes = len(self._color_encoding12)
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

    def get_class_rgb_encoding(self):
        """
        Returns:
            An ordered dictionary encoding for pixel value, class name, and
            class color.
        """
        return self._color_encoding12.copy()
