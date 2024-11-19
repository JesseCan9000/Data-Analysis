import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter, label

# Constants for scale
scale_length_nm = 20  # Scale bar length in nm
scale_length_pixels = 69  # Scale bar length in pixels
pixel_area = (scale_length_nm / scale_length_pixels)**2  # Area in nm² per pixel

# Gather all .tif files in the directory
tif_files = ["Pt Thin Films/" + f for f in os.listdir("Pt Thin Films") if f.endswith('.tif')]

# Process each .tif file
for tif_file in tif_files:
    # Load image
    image = Image.open(tif_file).convert("L")
    image_array = np.array(image)

    # Initial values for sliders
    initial_threshold = 0.5
    initial_sigma = 50

    # Create a new figure for this file
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), gridspec_kw={'wspace': 0.1})  # Add extra subplot for histogram

    # Define axes for each subplot
    ax_raw, ax_normalized, ax_binary, ax_contour, ax_histogram = axes

    # Plot raw image
    ax_raw.imshow(image, cmap="gray")
    ax_raw.set_title("Raw Image", fontsize=10)
    ax_raw.axis("off")

    # Create mutable container for the contour plot
    contour_plot_container = [None]  # Using a list to make the variable mutable in `update`

    # Function to update all plots based on current slider values
    def update():
        sigma = slider_sigma.val
        threshold = slider_threshold.val

        # Update normalized image with new Gaussian filter
        background = gaussian_filter(image_array, sigma=sigma)
        normalized_image = image_array - background
        normalized_image_clipped = np.clip(normalized_image, 0, 255)

        # Update binary image
        binary_image = (normalized_image_clipped >= threshold * 256).astype(int)

        # Update normalized image plot
        im_normalized.set_data(normalized_image_clipped)

        # Update binary image plot
        im_binary.set_data(binary_image)

        # Update contour plot
        if contour_plot_container[0] is not None:
            for coll in contour_plot_container[0].collections:
                coll.remove()  # Clear previous contours
        contour_plot_container[0] = ax_contour.contour(binary_image, colors="red", linewidths=0.2)

        # Calculate and plot histogram of dark region areas
        labeled_array, num_features = label(binary_image)  # Label connected dark regions
        region_areas = [
            np.sum(labeled_array == region) * pixel_area for region in range(1, num_features + 1)
        ]  # Calculate areas in nm²

        # Update histogram plot
        ax_histogram.clear()
        ax_histogram.hist(region_areas, bins=20, color="blue", edgecolor="black")
        ax_histogram.set_title("Histogram of Dark Region Areas", fontsize=10)
        ax_histogram.set_xlabel("Area (nm²)", fontsize=8)
        ax_histogram.set_ylabel("Count", fontsize=8)

        # Update area ratio text
        dark_pixels = np.sum(binary_image)
        light_pixels = binary_image.size - dark_pixels
        dark_area = dark_pixels * pixel_area
        light_area = light_pixels * pixel_area
        dark_to_total_ratio = dark_area / (dark_area + light_area)
        ratio_text.set_text(f"Dark Area/Total Area: {dark_to_total_ratio:.3f}")

        fig.canvas.draw_idle()

    # Initial Gaussian filtering and normalization
    background = gaussian_filter(image_array, sigma=initial_sigma)
    normalized_image = image_array - background
    normalized_image_clipped = np.clip(normalized_image, 0, 255)

    # Initial binary image
    binary_image = (normalized_image_clipped >= initial_threshold * 256).astype(int)

    # Plot normalized image
    im_normalized = ax_normalized.imshow(normalized_image_clipped, cmap="gray")
    ax_normalized.set_title("Normalized (Gaussian Filtered) Image", fontsize=10)
    ax_normalized.axis("off")

    # Plot binary image
    im_binary = ax_binary.imshow(binary_image, cmap="gray")
    ax_binary.set_title("Binary Image", fontsize=10)
    ax_binary.axis("off")

    # Plot contour image
    im_contour = ax_contour.imshow(image, cmap="gray")
    contour_plot_container[0] = ax_contour.contour(binary_image, colors="red", linewidths=0.2)
    ax_contour.set_title("Contours", fontsize=10)
    ax_contour.axis("off")

    # Add histogram of dark region areas
    labeled_array, num_features = label(binary_image)  # Label connected dark regions
    region_areas = [
        np.sum(labeled_array == region) * pixel_area for region in range(1, num_features + 1)
    ]  # Calculate areas in nm²
    ax_histogram.hist(region_areas, bins=20, color="blue", edgecolor="black")
    ax_histogram.set_title("Histogram of Dark Region Areas", fontsize=10)
    ax_histogram.set_xlabel("Area (nm²)", fontsize=8)
    ax_histogram.set_ylabel("Count", fontsize=8)

    # Add text for the dark-to-total area ratio
    dark_pixels = np.sum(binary_image)
    light_pixels = binary_image.size - dark_pixels
    dark_area = dark_pixels * pixel_area
    light_area = light_pixels * pixel_area
    dark_to_total_ratio = dark_area / (dark_area + light_area)
    ratio_text = ax_contour.text(
        0.5, -0.05, f"Dark Area/Total Area: {dark_to_total_ratio:.3f}",
        fontsize=10, ha="center", transform=ax_contour.transAxes
    )

    # Create sliders for threshold and sigma adjustment
    ax_slider_threshold = plt.axes([0.62, 0.06, 0.3, 0.03], facecolor="lightgray")  # Threshold slider position
    slider_threshold = Slider(ax_slider_threshold, "Binary Image Threshold", 0.0, 1.0, valinit=initial_threshold, valstep=0.01)

    ax_slider_sigma = plt.axes([0.18, 0.06, 0.3, 0.03], facecolor="lightgray")  # Sigma slider position
    slider_sigma = Slider(ax_slider_sigma, "Gaussian Filter Sigma", 1, 100, valinit=initial_sigma, valstep=1)

    # Connect sliders to the update function
    slider_threshold.on_changed(lambda val: update())
    slider_sigma.on_changed(lambda val: update())

    # Show the plot
    plt.suptitle(tif_file.split("/")[-1], fontsize=12, y=0.95)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make space for sliders
    plt.show()
