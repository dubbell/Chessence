import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import_paths = [
    "game",
    "game/model"
]

for path in import_paths:
    os_path = os.path.dirname(os.path.dirname(__file__))
    for directory in path.split("/"):
        os_path = os.path.join(os_path, directory)
    sys.path.append(os_path)

