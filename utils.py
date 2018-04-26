import numpy as np
import matplotlib.pyplot as plt


def imshow_batch(image_batch, nrows=1):
    if (np.ndim(image_batch) == 3):
        image_batch = np.expand_dims(image_batch, 0)

    # Compute the number of columns needed to plot the batch given the rows
    ncols = int(np.ceil(image_batch.shape[0] / nrows))

    # Show the images with subplot
    fig, axes = plt.subplots(nrows, ncols)
    for idx in range(image_batch.shape[0]):
        axes[idx].imshow(image_batch[idx].astype(int))

    plt.show()


def categorical_to_rgb(categorical_image, class_to_rgb):
    if (np.ndim(categorical_image) == 3):
        categorical_image = np.expand_dims(categorical_image, 0)

    rgb_image = np.zeros(
        (
            categorical_image.shape[0],
            categorical_image.shape[1],
            categorical_image.shape[2],
            3,
        )
    )
    for idx in range(categorical_image.shape[0]):
        image = np.argmax(categorical_image[idx], axis=-1).squeeze()
        for class_value, (class_name, rgb) in enumerate(class_to_rgb.items()):
            rgb_image[idx][image == class_value] = rgb

    return rgb_image
