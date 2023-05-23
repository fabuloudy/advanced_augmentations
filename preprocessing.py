import pandas as pd

import torchaudio

def get_class_name(file_path):
    return file_path.split('/')[-2]


def load_from_file(source_files: list, class_to_id):
    data = []
    for file_path in source_files:
        try:
            samples, _ = torchaudio.load(file_path, normalize=True)
        except Exception:
            print(f'Problem with downloading file {file_path}')
            continue
        data.append({"samples": samples, "class_id": class_to_id[get_class_name(file_path)]})
    return pd.DataFrame(data)
