# Feature Engineering & Encoding

import yaml
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler


def feature_transformation(data: pd.DataFrame, encoder_path: str) -> pd.DataFrame:
    """ Performs scaling over the middle columns of the dataframe.
    Looks for a trained encoder in the encoder_path and loads it if it exists. Otherwise it trains one.
    This way we can easily use this function on deployment
    :param data: pd.DataFrame
    :param encoder_path: str, path to the encoder
    :return data: pd.DataFrame, scaled dataframe
    """
    columns = data.columns[1:-1].to_list()
    if os.path.exists(encoder_path):
        encoder = pickle.load(open(encoder_path, 'rb'))
        data[columns] = encoder.transform(data[columns])
    else:
        encoder = MinMaxScaler().fit(data[columns])
        data[columns] = encoder.transform(data[columns])
        # Dump encoder for API usage
        pickle.dump(encoder, open(encoder_path, 'wb'))

    return data


def main():
    # Read parameters
    params = yaml.safe_load(open("params.yaml"))['feature_engenc']
    encoder_path = params['encoder_path']

    # Read data
    data = pd.read_pickle('data/prepared.pkl')

    # Feature & Labels segregation
    features = data.drop(columns='Class')
    labels = data['Class']

    # Remove encoder if exists
    if os.path.exists(encoder_path):
        os.remove(encoder_path)
    # Feature transformations
    features = feature_transformation(features, encoder_path)

    features.to_pickle('data/features.pkl')
    labels.to_pickle('data/labels.pkl')


if __name__ == '__main__':
    main()
