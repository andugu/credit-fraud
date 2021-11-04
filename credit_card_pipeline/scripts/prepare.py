# Prepare script

import yaml
import pandas as pd


def main():
    # Read parameters
    params = yaml.safe_load(open('params.yaml'))['prepare']
    data_path = params['data_path']

    # Read data (in this case there isn't any special preprocessing)
    data = pd.read_csv(data_path)

    data.to_pickle('data/prepared.pkl')


if __name__ == '__main__':
    main()
