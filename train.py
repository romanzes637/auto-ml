import json
import argparse
from joblib import dump

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='train_input.json')
    cmd_args = vars(parser.parse_args())
    print(cmd_args)
    with open(cmd_args['input']) as f:
        conf_args = json.load(f)
    print(conf_args)
    seed = conf_args['seed']
    X, y = np.array(conf_args['X']), np.array(conf_args['y'])
    # X, y = make_classification(n_samples=100,
    #                            n_features=4,
    #                            n_informative=2,
    #                            n_redundant=2,
    #                            n_repeated=0,
    #                            n_classes=2,
    #                            n_clusters_per_class=2,
    #                            random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=seed)
    model = MLPClassifier(
        hidden_layer_sizes=(10, 20),
        random_state=seed,
        max_iter=600,
        verbose=True).fit(X_train, y_train)
    with open(conf_args['model_path'], 'wb') as f:
        dump(model, f)
    output_data = {
        'model_path': conf_args['model_path'],
        'score': model.score(X_test, y_test)
    }
    with open(conf_args['output_path'], 'w') as f:
        json.dump(output_data, f)
