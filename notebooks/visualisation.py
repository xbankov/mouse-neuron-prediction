import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def visualise_class(images: np.ndarray, cls: str, class_map: np.ndarray, save_path: Optional[str] = None) -> None:
    """
    Visualizes class-specific images from a dataset.

    Args:
        images (np.ndarray): A 3D numpy array representing the dataset of images
                            with shape (height, width, num_images).
        cls (str): The class label to visualize.
        class_map (np.ndarray): An array of class labels corresponding to each image.
        save_path (str, optional): If provided, the path to save the visualization as an image.

    Returns:
        None
    """
    # Select class-specific images
    indices = np.where(class_map == cls)[0]
    num_cols = 20
    num_rows = len(indices) // num_cols + 1

    image_width = 90
    image_height = 68

    # Calculate the exact figure size based on image dimensions
    fig_width = num_cols * (image_width / 100)  # Adjust the divisor for desired spacing
    fig_height = num_rows * (image_height / 100)  # Adjust the divisor for desired spacing

    # Create a figure and axis
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

    # Loop through the images and plot them
    for i, ax in enumerate(axs.flat):
        if i < len(indices):
            ax.imshow(images[:, 90:180, indices[i]], cmap='gray')
        ax.axis('off')  # Turn off axis labels and ticks

    # Adjust spacing between subplots for better visualization
    plt.tight_layout()

    # Reduce spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Display the plot
    plt.show()
