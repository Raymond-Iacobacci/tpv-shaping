import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot images saved as .npy files in contour graphs."
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        required=True,
        help="Path to the LOG_DIR containing image .npy files."
    )
    parser.add_argument(
        '--iteration',
        type=int,
        default=None,
        help="Specific iteration number to plot. If not specified, plots all iterations."
    )
    parser.add_argument(
        '--image-index',
        type=int,
        default=None,
        help="Specific image index to plot. If not specified, plots all images."
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help="Directory to save the contour plots. Defaults to 'plots'."
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help="If set, display the plots interactively."
    )
    parser.add_argument(
        '--saved-state',
        action='store_true',
        help="If set, only generate plots for .npy files without existing images."
    )

    return parser.parse_args()

def get_image_files(log_dir, iteration=None, image_index=None):
    """
    Retrieve image file paths based on the specified iteration and image index.
    """
    files = os.listdir(log_dir)
    image_files = []
    for file in files:
        if not file.endswith('.npy'):
            continue
        try:
            # Expecting filename format: {image_index}.{iteration}.npy
            base, ext = file.split('.npy')[0].split('.')
            img_idx = int(base)
            iter_num = int(ext)
        except ValueError:
            # Skip files that don't match the expected format
            continue

        if iteration is not None and iter_num != iteration:
            continue
        if image_index is not None and img_idx != image_index:
            continue

        image_files.append((img_idx, iter_num, os.path.join(log_dir, file)))

    return image_files

def plot_contour(image_array, title, output_path, show_plot=False):
    """
    Plot a single image array as a contour graph and save it.
    """
    plt.figure(figsize=(8, 6))
    contour = plt.plot(image_array)
    # plt.colorbar(contour, format='%.2f')
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Improve layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300)
    print(f"Saved contour plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

def main():
    args = parse_arguments()

    log_dir = args.log_dir
    iteration = args.iteration
    image_index = args.image_index
    output_dir = args.output_dir
    show_plot = args.show

    # Validate LOG_DIR
    if not os.path.isdir(log_dir):
        print(f"Error: The specified log directory '{log_dir}' does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Retrieve image files based on the provided arguments
    image_files = get_image_files(log_dir, iteration, image_index)

    if not image_files:
        print("No image files found matching the specified criteria.")
        return

    print(f"Found {len(image_files)} image file(s) to plot.")

    for img_idx, iter_num, file_path in sorted(image_files, key=lambda x: (x[1], x[0])):
        # Define output plot filename first
        plot_filename = f"image_{img_idx}_iteration_{iter_num}.png"
        output_path = os.path.join(output_dir, plot_filename)

        # Skip if file already exists
        if os.path.exists(output_path):
            continue

        # Only proceed with loading and plotting if file doesn't exist
        try:
            image_data = np.load(file_path)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue

        # Determine plot title
        title = f"Image {img_idx} | Iteration {iter_num}"

        # Plot and save the contour graph
        plot_contour(image_data, title, output_path, show_plot)

if __name__ == "__main__":
    main()
