from sklearn.metrics import f1_score, adjusted_rand_score

PolimiHouse = {
    'dataset': 'data/PolimiHouse.csv',
    'alpha': 0.99,
    'bandwidth': 'scott',
    'grid_size': 100,
    'kernel': 'gaussian',
    'reset_time': 14,
    'sample_test_size': 47,
    'window': 'standard',
}
ARAS = {
    'dataset': 'data/ARAS.csv',
    'alpha': 0.99,
    'bandwidth': 'silverman',
    'grid_size': 100,
    'kernel': 'gaussian',
    'reset_time': 13,
    'sample_test_size': 76,
    'window': 'standard',
}
VanKastareen = {
    'dataset': 'data/VanKastareen.csv',
    'alpha': 0.99,
    'bandwidth': 'scott',
    'grid_size': 100,
    'kernel': 'gaussian',
    'reset_time': 60,
    'sample_test_size': 28,
    'window': 'standard',
}
DAMADICS = {
    'dataset': 'data/DAMADICS.csv',
    'alpha': 0.99,
    'bandwidth': 'scott',
    'grid_size': 100,
    'kernel': 'gaussian',
    'reset_time': 32,
    'sample_test_size': 80,
    'window': 'standard',
}

CovtFD = {
    'dataset': 'data/covtFD.csv',
    'alpha': 0.95,
    'bandwidth': 'variance',
    'grid_size': 100,
    'kernel': 'gaussian',
    'reset_time': 10,
    'sample_test_size': 50,
    'window': 'standard'
}

scoring = {
        'F1': f1_score,
        'ARI': adjusted_rand_score
    }