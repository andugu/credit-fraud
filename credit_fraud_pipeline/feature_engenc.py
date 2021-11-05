# Feature Engineering & Encoding

import yaml
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Union


def feature_transformation(data: pd.DataFrame, encoder=None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, object]]:
    """ Performs scaling over the middle columns of the dataframe.
    Looks for a trained encoder in the encoder_path and loads it if it exists. Otherwise it trains one.
    This way we can easily use this function on deployment
    :param data: pd.DataFrame
    :param encoder: MinMaxScaler
    :return data: pd.DataFrame, scaled dataframe
    :return encoder: MinMaxScaler, optional
    """
    columns = data.columns[1:-1].to_list()
    if encoder is not None:
        data[columns] = encoder.transform(data[columns])
        return data
    else:
        encoder = MinMaxScaler().fit(data[columns])
        data[columns] = encoder.transform(data[columns])
        return data, encoder


def main():
    # Read parameters
    params = yaml.safe_load(open("params.yaml"))['feature_engenc']
    encoder_path = params['encoder_path']

    # Read data
    data = pd.read_pickle('data/prepared.pkl')

    # Feature & Labels segregation
    features = data.drop(columns='Class')
    labels = data['Class']

    # Feature transformations
    features, encoder = feature_transformation(features)

    features.to_pickle('data/features.pkl')
    labels.to_pickle('data/labels.pkl')
    with open(encoder_path, 'wb') as file:
        pickle.dump(encoder, file)


if __name__ == '__main__':
    main()
