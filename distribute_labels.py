import os
from cfg import CFG
from utils import make_symlink
from segment_utils import get_segment_id_paths_dict

label_path = CFG.processed_labels_dir
train_val_segment_ids_set = set(CFG.train_val_segment_ids)
data_dir = CFG.base_data_dir


def distribute_labels(label_dir, train_val_segments_ids_set, segment_id_paths_dict):
    labels_segment_ids_added = set()
    for label in os.listdir(label_dir):
        if label.endswith(".png"):
            segment_label_path = os.path.join(label_dir, label)
            # Assume the filename scheme is {segment_id}_inklabels.
            label_segment_id = label.split("_")[0]
            labels_segment_ids_added.add(label_segment_id)
            new_segment_label_path = os.path.join(os.path.join(segment_id_paths_dict[label_segment_id], label))
            make_symlink(segment_label_path, new_segment_label_path)
    diff = train_val_segments_ids_set - labels_segment_ids_added
    if diff:
        raise Exception(f"Expected all train/val segment ids to have corresponding ink labels, but segments "
                        f"{diff} do not.")


if __name__ == "__main__":
    distribute_labels(label_path, train_val_segment_ids_set, get_segment_id_paths_dict(data_dir))
