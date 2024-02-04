import os
from cfg import CFG
from utils import make_symlink
from segment_utils import get_segment_id_paths_dict
import shutil

data_path = CFG.base_data_dir
train_val_data_path = CFG.train_val_dir
test_data_path = CFG.test_data_dir
label_path = CFG.processed_labels_dir


def create_links_from_list(base_dir, segment_ids, segment_id_paths):
    for segment_id in segment_ids:
        if segment_id not in segment_id_paths:
            print(f"{segment_id} does not exist in segment id paths in data path.")
            continue
        new_path = os.path.join(base_dir, segment_id)
        prev_path = os.path.join(os.getcwd(), segment_id_paths[segment_id])
        ink_label_name = f"{segment_id}_inklabels.png"
        new_ink_path = os.path.join(prev_path, ink_label_name)
        if os.path.exists(new_path):
            os.remove(new_path)
        # if os.path.exists(new_ink_path):
        #     os.remove(new_ink_path)
        make_symlink(prev_path, os.path.join(os.getcwd(), new_path))
        try:
            make_symlink(os.path.join(os.getcwd(), os.path.join(label_path, ink_label_name)), new_ink_path)
        except FileExistsError:
            print(f"Skipping {segment_id} ink label because already created.")


def setup_data(segment_id_paths_dict):
    create_links_from_list(train_val_data_path, CFG.train_segment_ids + CFG.val_segment_ids, segment_id_paths_dict)
    create_links_from_list(test_data_path, CFG.test_segment_ids, segment_id_paths_dict)


if __name__ == "__main__":
    # shutil.rmtree(os.path.join(data_path, "scroll1", "20230702185753"))
    segment_id_paths_dict = get_segment_id_paths_dict(data_path)
    setup_data(segment_id_paths_dict)
