# Prepare script
import yaml
import pandas as pd
import zipfile
import os


def main():
    # Read parameters
    params = yaml.safe_load(open('params.yaml'))['prepare']
    data_path = params['data_path']
    archive_path = params['archive_path']

    # Read data & download if missing (in this case there isn't any special preprocessing)
    if not os.path.exists(data_path):
        zipfile.ZipFile(archive_path, 'r').extractall('data')
    data = pd.read_csv(data_path)

    data.to_pickle('data/prepared.pkl')


if __name__ == '__main__':
    main()
