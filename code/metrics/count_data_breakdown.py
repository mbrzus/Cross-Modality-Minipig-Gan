import json

with open("../metadata/structure.json") as f:
    data = json.load(f)

# print(data['train'])
train_count = 0
for sub in data['train']:
    for sess in data['train'][sub]:
        if len(data['train'][sub][sess]['t1w']) > 0 and len(data['train'][sub][sess]['t2w']) > 0:
            train_count += 1

def count_data_use(data_use):
    count = 0
    for sub in data[data_use]:
        for sess in data[data_use][sub]:
            if len(data[data_use][sub][sess]['t1w']) > 0 and len(data[data_use][sub][sess]['t2w']) > 0:
                count += 1
    print(f"{data_use}: {count}")

count_data_use('train')
count_data_use('test')
count_data_use('validation')