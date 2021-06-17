import json
import argparse
from joblib import load

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='predict_input.json')
    cmd_args = vars(parser.parse_args())
    print(cmd_args)
    with open(cmd_args['input']) as f:
        conf_args = json.load(f)
        print(conf_args)
    X = np.array(conf_args['X'])
    model = load(conf_args['model_path'])
    output_data = {
        'y_pred': model.predict(X).tolist()
    }
    with open(conf_args['output_path'], 'w') as f:
        json.dump(output_data, f)
