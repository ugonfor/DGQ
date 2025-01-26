
# delete folders that contain only 'run.log' file.

import os
import shutil

def clean_folder(folder):
    for root, dirs, files in os.walk(folder):
        if len(dirs) == 0 and len(files) == 1 and files[0] == 'run.log':
            print(f"Deleting {root}")
            shutil.rmtree(root)

clean_folder('results')