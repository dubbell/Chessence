import os
import sys
PROJECT_PATH = os.getcwd()
GAME_PATH = os.path.join(
    PROJECT_PATH, "game")
GAME_MODEL_PATH = os.path.join(
    GAME_PATH, "model")
sys.path.append(GAME_PATH)
sys.path.append(GAME_MODEL_PATH)