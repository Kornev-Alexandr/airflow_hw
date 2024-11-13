# <YOUR_IMPORTS>
import dill
import glob
import pandas as pd
import os
import json




path = os.environ.get('PROJECT_PATH', '.')
mod = sorted(os.listdir(f'{path}/data/models/'))[-1]

with open(f'{path}/data/models/{mod}', 'rb') as file:
    model = dill.load(file)


def predict():
    # <YOUR_CODE>

    data_list = list()
    for test_file in glob.glob(f'{path}/data/test/*.json', recursive=True):
        with open(test_file, 'r') as j:
            test_json = json.load(j)
        df_test = pd.DataFrame(test_json, index=[0])
        preds = model.predict(df_test)

        data = {'test': test_file[12:22], 'preds': preds[0]}
        data_list.append(data)

    df_preds = pd.DataFrame(data_list)
    df_preds.to_csv(f'{path}/data/predictions/preds.csv', index = False)

    pass


if __name__ == '__main__':
    predict()
