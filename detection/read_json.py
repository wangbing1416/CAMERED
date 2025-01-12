import os
import json
import pandas as pd
import numpy as np

path = './json/weibo20-R2-16-2/'
files = os.listdir(path)
item = ''
dic24 = []
dic25 = []
for file_name in files:
    id = file_name[-14:-5]
    model = file_name[:-21]
    seed = file_name[-16]
    # model = file_name[:-18]
    fp = open(os.path.join(path, file_name))
    try:
        assert 'seed' in file_name
        data = json.load(fp)[0]
        data = {'seed': seed, 'baseline': model, 'id': id, **data}
        # data = {'baseline': model, 'id': id, **data}
        if '1231-' in id:
            dic24.append(data)
        else:
            dic25.append(data)
    except:
        print('error id: {}'.format(file_name))

dic24 = sorted(dic24, key=(lambda x: x['id']))  # sort by id
dic25 = sorted(dic25, key=(lambda x: x['id']))  # sort by id
dic = dic24 + dic25

def insert_empty_dicts(data):
    result = []
    count = 0

    for i, item in enumerate(data):
        if count != 0 and count % 5 == 0:
            result.append({})
            result.append({})
        result.append(item)
        count += 1

    return result


dic = insert_empty_dicts(dic)

data_pd = pd.DataFrame(dic)
data_pd.to_excel(excel_writer='read_result.xlsx', float_format='%.4f', index=False)

print("read results have been saved in ./read_result.xlsx !")
