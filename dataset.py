import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config as cfg



def preprocessing():
    print('Preprocessing data...')
    data = pd.read_csv(cfg.data_path)

    cols = data.columns.values

    dense_features = [f for f in cols if f[0] == 'I']
    sparse_features = [f for f in cols if f[0] == 'C']

    def process_dense_features(data, features):
        d = data.copy()
        d = d[features].fillna(0.)
        for f in features:
            d[f] = d[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)
        return d
    data_dense = process_dense_features(data, dense_features)

    def process_sparse_features(data, features):
        d = data.copy()
        d = d[features].fillna('-1')
        for f in features:
            label_encoder = LabelEncoder()
            d[f] = label_encoder.fit_transform(d[f])
        return d
    data_sparse = process_sparse_features(data, sparse_features)

    total_data = pd.concat([data_dense, data_sparse], axis=1)
    total_data['label'] = data['label']

    print('done')
    return dense_features, sparse_features, total_data
