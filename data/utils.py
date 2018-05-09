import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_files(folder, name_filter=None, extension_filter=None):
    """Returns the list of files in a folder.

    Args:
        folder (string): The path to a folder.
        name_filter (string, optional): The returned files must contain
            this substring in their filename. Default: None; files are not
            filtered.
        extension_filter (string, optional): The desired file extension.
            Default: None; files are not filtered.

    Returns:
        The list of files.

    """
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    # Filename filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files that do not
    # contain "name_filter"
    if name_filter is None:
        # This looks hackish...there is probably a better way
        name_cond = lambda filename: True
    else:
        name_cond = lambda filename: name_filter in filename

    # Extension filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files whose extension
    # is not "extension_filter"
    if extension_filter is None:
        # This looks hackish...there is probably a better way
        ext_cond = lambda filename: True
    else:
        ext_cond = lambda filename: filename.endswith(extension_filter)

    filtered_files = []

    # Explore the directory tree to get files that contain "name_filter" and
    # with extension "extension_filter"
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_cond(file) and ext_cond(file):
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)

    return filtered_files


def pil_loader(data_path, label_path, shape):
    """Loads a sample and label image given their path as PIL images.

    Args:
        data_path (string): The filepath to the image.
        label_path (string): The filepath to the ground-truth image.
        shape (tuple): The requested size in pixels, as a 2-tuple:
            (width,height). If set to ``None``, resizing is not performed.

    Returns:
        The image and the label as PIL images.

    """
    data = Image.open(data_path)
    label = Image.open(label_path)

    if shape is not None:
        if data.size != shape:
            data = data.resize(shape)
        if label.size != shape:
            label = label.resize(shape)

    return data, label


def remap(image, old_values, new_values):
    """Replaces pixels values with new values.

    Pixel values from ``old_values`` in ``image`` are replaced index by
    index with values from ``new_values``.

    Args:
        image (numpy.ndarray): The image to process.
        old_values (tuple): A tuple of values to be replaced.
        new_values (tuple): A tuple of new values to replace ``old_values``.

    Returns:
        The image with remapped classes.

    """
    assert type(new_values) is tuple, "new_values must be of type tuple"
    assert type(old_values) is tuple, "old_values must be of type tuple"
    assert len(new_values) == len(old_values), (
        "new_values and old_values must have the same length"
    )

    # Images with more than one channel are assumed to be in categorical format
    # therefore, they are converted to integer format
    if image.shape[-1] > 1:
        image = np.argmax(image, axis=-1)

    # Replace old values by the new ones
    remapped_img = np.zeros_like(image)
    for old, new in zip(old_values, new_values):
        # Since tmp is already initialized as zeros we can skip new values
        # equal to 0
        if new != 0:
            remapped_img[image == old] = new

    return remapped_img


def imshow_batch(image_batch, nrows=1, figsize=None):
    """Shows a batch of images in a grid.

    Note: Blocks execution until the figure is closed.

    Args:
        image_batch (numpy.ndarray): A batch of images. Dimension is assumed
            as (batch, height, width, channels); or, (height, width, channels)
            which is transformed into (1, height, width, channels).
        nrows (int): The number of rows of the image grid. The number of
            columns is infered from the rows and the batch size.
        figsize (tuple, optional): The size of the figure (width, height)
            in inches. Default: None (defaults to rc figure.figsize)

    """
    assert nrows > 0, "number of rows must be greater than 0"
    assert figsize is None or isinstance(
        figsize, tuple
    ), ("expect type None or tuple for figsize")

    if (np.ndim(image_batch) == 3):
        image_batch = np.expand_dims(image_batch, 0)

    # Compute the number of columns needed to plot the batch given the rows
    ncols = int(np.ceil(image_batch.shape[0] / nrows))

    # Show the images with subplot
    if figsize is None:
        figsize = plt.rcParams.get('figure.figsize')

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for idx in range(image_batch.shape[0]):
        if nrows == 1:
            axes[idx].imshow(image_batch[idx].astype(int))
        else:
            col = idx % ncols
            row = idx // ncols
            axes[row, col].imshow(image_batch[idx].astype(int))

    plt.show()


def categorical_to_rgb(categorical_batch, class_to_rgb):
    """Converts label(s) from categorical format to RGB representation.

    Args:
        categorical_batch (numpy.ndarray): A batch of labels in categorical
            format. Dimension is assumed as (batch, height, width, channels);
            or, (height, width, channels) which is transformed into
            (1, height, width, channels).
        class_to_rgb (OrderedDict): An ordered dictionary that relates pixel
            values, class names, and class colors.

    Returns:
        The label(s) as RGB images.

    """
    if (np.ndim(categorical_batch) == 3):
        categorical_batch = np.expand_dims(categorical_batch, 0)

    rgb_batch = np.zeros(
        (
            categorical_batch.shape[0],
            categorical_batch.shape[1],
            categorical_batch.shape[2],
            3,
        ),
        dtype=np.uint8
    )
    for idx, image in enumerate(categorical_batch):
        image = np.argmax(image, axis=-1).squeeze()
        for class_value, (class_name, rgb) in enumerate(class_to_rgb.items()):
            rgb_batch[idx][image == class_value] = rgb

    return rgb_batch


def rgb_to_categorical(image_batch, class_to_rgb):
    """Converts labels from RGB to categorical representation.

    Args:
        image_batch (numpy.ndarray): A batch of labels in the RGB color-space
            Dimension is assumed as (batch, height, width, channels);
            or, (height, width, channels) which is transformed into
            (1, height, width, channels).
        class_to_rgb (OrderedDict): An ordered dictionary that relates pixel
            values, class names, and class colors.

    Returns:
        The label(s) in categorical format.

    """
    if (np.ndim(image_batch) == 3):
        image_batch = np.expand_dims(image_batch, 0)

    categorical_batch = np.zeros(
        (
            image_batch.shape[0],
            image_batch.shape[1],
            image_batch.shape[2],
            len(class_to_rgb),
        ),
        dtype=np.uint8
    )
    for idx, image in enumerate(image_batch):
        for class_value, (class_name, rgb) in enumerate(class_to_rgb.items()):
            # Create mask of pixels that match the rgb code for this class
            mask = np.all(image == rgb, axis=-1)

            # Assign the one-hot vector representation of the class to the
            # categorical image inside the batch. The line below outputs the
            # following:
            # k = 1; M = 3 -> (0, 1, 0)
            # k = 2; M = 3 -> (0, 0, 1)
            onehot = np.eye(1, M=len(class_to_rgb), k=class_value).ravel()
            categorical_batch[idx][mask] = onehot

    return categorical_batch


def enet_weighing(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper.

        w_class = 1 / (ln(c + p_class)),
    where c is usually 1.02 and p_class is the propensity score of that
    class:
        propensity_score = freq_class / total_pixels.

    References: https://arxiv.org/abs/1606.02147

    Args:
        dataloader (keras.utils.Sequence): A data loader to iterate over the
            dataset.
        num_classes (int): The number of classes.
        c (int, optional): AN additional hyper-parameter which restricts
            the interval of values for the weights. Default: 1.02.

    Returns:
        The class weights as a ndarray of ints.

    """
    class_count = 0
    total = 0
    # Can't do "for _, label in dataloader:" becuase Keras implements __iter__
    # as an inifinite loop (see https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py)
    for i in range(len(dataloader)):
        # dataloader should contain labels in categorical format, but lets
        # handle both categorical and int formats
        _, label = dataloader[i]

        # int format
        if label.shape[-1] == 1:
            count = np.bincount(label.flatten(), minlength=num_classes)
        # Categorical format
        else:
            label_reshape = label.reshape(-1, num_classes)
            count = np.sum(label_reshape, axis=0)

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += count
        total += label.shape[0] * label.shape[1] * label.shape[2]

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def median_freq_balancing(dataloader, num_classes):
    """Computes class weights using median frequency balancing.

        w_class = median_freq / freq_class,
    where freq_class is the number of pixels of a given class divided by
    the total number of pixels in images where that class is present, and
    median_freq is the median of freq_class.

    References: https://arxiv.org/abs/1411.4734

    Args:
        dataloader (keras.utils.Sequence): A data loader to iterate over the
            dataset.
        num_classes (int): The number of classes

    Returns:
        The class weights as a ndarray of ints.

    """
    class_count = 0
    total = 0
    # Can't do "for _, label in dataloader:" becuase Keras implements __iter__
    # as an inifinite loop (see https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py)
    for i in range(len(dataloader)):
        # dataloader should contain labels in categorical format, but lets
        # handle both categorical and int formats
        _, label = dataloader[i]

        # int format
        if label.shape[-1] == 1:
            count = np.bincount(label.flatten(), minlength=num_classes)
        # Categorical format
        else:
            label_reshape = label.reshape(-1, num_classes)
            count = np.sum(label_reshape, axis=0)

        # Create of mask of classes that exist in the label
        mask = count > 0
        # Multiply the mask by the pixel count. The resulting array has
        # one element for each class. The value is either 0 (if the class
        # does not exist in the label) or equal to the pixel count (if
        # the class exists in the label)
        size = label.shape[0] * label.shape[1] * label.shape[2]
        total += mask * size

        # Sum up the number of pixels found for each class
        class_count += count

    # Compute the frequency and its median
    freq = class_count / total
    med = np.median(freq)

    return med / freq
