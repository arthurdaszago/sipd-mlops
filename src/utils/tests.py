import os
from .open import open_file

# =====================================
TEST_STATS_PATH = os.getenv('TEST_STATS_PATH')
# =====================================

def load_test_stats():
    test_stats_path = os.path.join(TEST_STATS_PATH, 'stats.json')
    test_stats = open_file(test_stats_path)

    return test_stats


def load_test_stats_remaked():
    remaked_test_stats_path = os.path.join(TEST_STATS_PATH, 'retrain_stats.json')
    remaked_test_stats = open_file(remaked_test_stats_path)

    return remaked_test_stats