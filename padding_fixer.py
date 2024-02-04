import os
import cv2
from segment_utils import get_segment_id_paths_dict
from cfg import CFG

data_path = "data/scroll1_hari"
label_path = "labels"
resized_label_path = "../label_test"

"""
We have to resize padding before we rotate it, hence it's important that we use the base data directory as reference,
not the rotate data directory.
"""


def add_or_remove_padding_ink_label_padding(segment_id_paths_dict, label_path, resized_label_path):
    for filename in os.listdir(label_path):
        if filename.endswith(".png"):
            print(f"Resizing {filename}")
            label_segment_path = os.path.join(label_path, filename)
            segment_id = os.path.splitext(filename)[0].split("_")[0]
            if segment_id not in segment_id_paths_dict:
                print(f"{segment_id} does not exist in data dir! Skipping")
                continue
            data_segment_tif = os.path.join(segment_id_paths_dict[segment_id], f"{segment_id}.tif")
            h, w = cv2.imread(data_segment_tif).shape[:2]
            label_data = cv2.imread(label_segment_path, 0)
            orig_h, orig_w = label_data.shape[:2]
            resized_label_data = label_data[0:h, 0:w]
            cv2.imwrite(os.path.join(resized_label_path, filename), resized_label_data)
            print(f"[Success] Resizing filename {filename}. Original h {orig_h}, w {orig_w}. Final h {h}, w {w}.")


if __name__ == "__main__":
    add_or_remove_padding_ink_label_padding(segment_id_paths_dict=get_segment_id_paths_dict(CFG.base_data_dir),
                                            label_path=label_path, resized_label_path=resized_label_path)
