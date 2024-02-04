import os
from functools import lru_cache
from typing import Dict


@lru_cache(maxsize=10)
def get_segment_id_paths_dict(data_dir) -> Dict[str, str]:
    segment_id_path_keyed_by_segment_id = dict()
    for data_source in os.listdir(data_dir):
        data_source_path = os.path.join(data_dir, data_source)
        if os.path.isdir(data_source_path):
            for segment_id in os.listdir(data_source_path):
                segment_id_path = os.path.join(data_source_path, segment_id)
                if os.path.isdir(segment_id_path):
                    if segment_id in segment_id_path_keyed_by_segment_id:
                        raise Exception(f"{segment_id} was already previously found with path "
                                        f"{segment_id_path_keyed_by_segment_id[segment_id]}, "
                                        f"now trying to insert new path {segment_id_path}")
                    segment_id_path_keyed_by_segment_id[segment_id] = segment_id_path
    return segment_id_path_keyed_by_segment_id
