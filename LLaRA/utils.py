import os
import random
import numpy as np
import json
import pickle
import gzip

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_txt(file_path):
    with open(file_path, "r") as f:
        return f.readlines()
