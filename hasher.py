import hashlib
import numpy as np, os
import torch
import pickle as pk
  
def hash_dict(d):
    # Convert the dictionary into a sorted string
    d_str = str(sorted(d.items())).encode('utf-8')
    
    # Use SHA-256 to hash the string and return the hexdigest
    return hashlib.sha256(d_str).hexdigest()

# folder_raw = './Cross_weak_intSearch/Raw/'
# folder_hashed = './Cross_weak_intSearch/Hashed/'
# files = os.listdir(folder_raw)
# for file in files:
#     f = open(folder_raw + file, 'rb')
#     dict = pk.load(f)
#     f.close()
#     os.rename(folder_raw+file, folder_raw+hash_dict(dict)+'.pk')
    








