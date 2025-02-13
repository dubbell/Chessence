import os
import sys
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH, "game"
)
print(PROJECT_PATH, SOURCE_PATH)
sys.path.append(SOURCE_PATH)