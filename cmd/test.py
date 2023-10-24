import os
import json
import numpy as np

TEST_STATS_PATH = None
if os.getenv('TEST_STATS_PATH') is not None:
    TEST_STATS_PATH = os.getenv('TEST_STATS_PATH')
else:
    TEST_STATS_PATH = '/home/arthur/Documents/ifc/tc/code/sipd-mlops/stats/test'

file = open(os.path.join(TEST_STATS_PATH, 'matrix_confusion.json'))

# load json content in file
json = json.load(file)

file.close()

print('json: ', np.array(json).flatten())