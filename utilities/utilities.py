
import datetime
import time
import os
import pickle as pkl

start_time = time.time()


def get_uptime():
    return '{}'.format(datetime.timedelta(seconds=time.time() - start_time))


def list_files_in_dir(path: str):
    li = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    li.sort()
    return li


def list_file_paths_in_dir(path: str):
    li = [os.path.join(path, f) for f in os.listdir(
        path) if os.path.isfile(os.path.join(path, f))]
    li.sort()
    return li

def save_similarity_vector(vector, path):
    with open(path, "wb") as f:
        pkl.dump(vector, f)

def load_similarity_vector(path):
    with open(path, "rb") as f:
        return pkl.load(f)
