import numpy as np
import matplotlib.pyplot as plt


def imshow_batch(image_batch, nrows=1):
    """Shows a batch of images in a grid.

    Note: Blocks execution until the figure is closed.

    Args:
        image_batch (numpy.ndarray): A batch of images. Dimension is assumed
            as (batch, height, width, channels); or, (height, width, channels)
            which is transformed into (1, height, width, channels).
        nrows (int): The number of rows of the image grid. The number of
            columns is infered from the rows and the batch size.

    """
    if (np.ndim(image_batch) == 3):
        image_batch = np.expand_dims(image_batch, 0)

    # Compute the number of columns needed to plot the batch given the rows
    ncols = int(np.ceil(image_batch.shape[0] / nrows))

    # Show the images with subplot
    fig, axes = plt.subplots(nrows, ncols)
    for idx in range(image_batch.shape[0]):
        axes[idx].imshow(image_batch[idx].astype(int))

    plt.show()


def categorical_to_rgb(categorical_batch, class_to_rgb):
    """Converts a label from categorical format to its RGB representation.

    Args:
        categorical_batch (numpy.ndarray): A batch of labels in categorical
            format. Dimension is assumed as (batch, height, width, channels);
            or, (height, width, channels) which is transformed into
            (1, height, width, channels).
        class_to_rgb (OrderedDict): An ordered dictionary that relates pixel
            values, class names, and class colors.

    """
    if (np.ndim(categorical_batch) == 3):
        categorical_batch = np.expand_dims(categorical_batch, 0)

    rgb_image = np.zeros(
        (
            categorical_batch.shape[0],
            categorical_batch.shape[1],
            categorical_batch.shape[2],
            3,
        ),
        dtype=np.uint8
    )
    for idx in range(categorical_batch.shape[0]):
        image = np.argmax(categorical_batch[idx], axis=-1).squeeze()
        for class_value, (class_name, rgb) in enumerate(class_to_rgb.items()):
            rgb_image[idx][image == class_value] = rgb

    return rgb_image
