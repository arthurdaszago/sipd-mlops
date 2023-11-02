from .tests import load_test_stats, load_test_stats_remaked

def detect_experimet_concept_drift(stats):
    test_stats = load_test_stats()

    if test_stats['F1-Score'] - stats['F1-Score'] >= 0.05:
        return True
    
    return False



def detect_experiment_remake_concept_drift(stats):
    remaked_test_stats = load_test_stats_remaked()

    if remaked_test_stats['F1-Score'] - stats['F1-Score'] >= 0.05:
        return True
    
    return False