import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from experiments.config import PolimiHouse, ARAS, VanKastareen, DAMADICS, CovtFD, scoring
from src.furaki.tree import FurakiTree


def execute(params: dict):
    df = pd.read_csv(params.get('dataset'))
    y = df.drifting.to_list()
    X = df.drop("drifting", axis=1)

    tree = FurakiTree(**params)
    tree.fit(X)
    labels = np.array(tree.predict())

    print(params.get("dataset")[5:-4],'F1', scoring['F1'](y, labels, average='macro'))
    print(params.get("dataset")[5:-4],'ARI', scoring['ARI'](y, labels))


def main():
    
    execute(PolimiHouse)
    execute(ARAS)
    execute(VanKastareen)
    execute(DAMADICS)
    execute(CovtFD)

if __name__ == '__main__':
    main()