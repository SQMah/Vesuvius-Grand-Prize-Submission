from utils import get_all_segment_paths
import os
from collections import defaultdict
import matplotlib.pyplot as plt


def get_num_layers_across_all_data(data_dir):
    layers = defaultdict(int)
    segments_paths = get_all_segment_paths(data_dir)
    for segment_path in segments_paths:
        segment_layer_path = os.path.join(segment_path, "layers")
        if os.path.isdir(segment_layer_path):
            print(f"Processing segment layer path {segment_layer_path}")
            for filename in os.listdir(segment_layer_path):
                if filename.endswith(".tif"):
                    filename_without_ext = os.path.splitext(filename)[0]
                    layers[int(filename_without_ext)] += 1

    # Plot layers as a bar chart
    plt.bar(layers.keys(), layers.values())
    plt.show()

    # Write plot
    plt.savefig(os.path.join("./", "num_layers.png"))


if __name__ == "__main__":
    get_num_layers_across_all_data("./data")
