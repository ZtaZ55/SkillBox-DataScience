import logging
import os
import json
from datetime import datetime

import dill
import pandas as pd


path = os.environ.get('PROJECT_PATH', 'C:/Users/stani/PycharmProjects/airflow_hw')


def predict():
    model_filename = f'{path}/data/models/' + os.listdir(path=f'{path}/data/models/')[0]
    with open(model_filename, 'rb') as file:
        model = dill.load(file)

    pred_list = os.listdir(path=f'{path}/data/test/')

    res_pred = pd.DataFrame(columns=['car_id', 'pred'])
    num = 0
    for elem in pred_list:
        with open(f'{path}/data/test/' + elem) as test_data:
            X = pd.DataFrame.from_dict([json.load(test_data)])
        y = model.predict(X)
        res_pred.loc[num] = [X.id[0] ,y[0]]
        num += 1

    result = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'

    res_pred.to_csv(path_or_buf=result, index=False)



    logging.info(f'Model is saved as {result}')




if __name__ == '__main__':
    predict()
