import os
from PIL import Image
import numpy as np


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
        image (PIL.Image or numpy.ndarray): The image to process.
        old_values (tuple): A tuple of values to be replaced.
        new_values (tuple): A tuple of new values to replace ``old_values``.

    Returns:
        The remapped image of the same type as `image`

    """
    assert isinstance(image, Image.Image) or isinstance(
        image, np.ndarray
    ), "image must be of type PIL.Image or numpy.ndarray"
    assert type(new_values) is tuple, "new_values must be of type tuple"
    assert type(old_values) is tuple, "old_values must be of type tuple"
    assert len(new_values) == len(
        old_values
    ), "new_values and old_values must have the same length"

    # If image is a PIL.Image convert it to a numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Replace old values by the new ones
    remapped_img = np.zeros_like(image)
    for old, new in zip(old_values, new_values):
        # Since tmp is already initialized as zeros we can skip new values
        # equal to 0
        if new != 0:
            remapped_img[image == old] = new

    # If the input is a PIL image return as a PIL.Image too, else return
    # numpy array
    if isinstance(image, Image.Image):
        remapped_img = Image.fromarray(remapped_img)

    return remapped_img


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
        # dataloader contains labels in categorical format, convert it to int
        # format
        _, label = dataloader[i]
        label_int = np.argmax(label, axis=-1)

        # Flatten label
        flat_label = label_int.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

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
        # dataloader contains labels in categorical format, convert it to int
        # format
        _, label = dataloader[i]
        label_int = np.argmax(label, axis=-1)

        # Flatten label
        flat_label = label_int.flatten()

        # Sum up the class frequencies
        bincount = np.bincount(flat_label, minlength=num_classes)

        # Create of mask of classes that exist in the label
        mask = bincount > 0
        # Multiply the mask by the pixel count. The resulting array has
        # one element for each class. The value is either 0 (if the class
        # does not exist in the label) or equal to the pixel count (if
        # the class exists in the label)
        total += mask * flat_label.size

        # Sum up the number of pixels found for each class
        class_count += bincount

    # Compute the frequency and its median
    freq = class_count / total
    med = np.median(freq)

    return med / freq
