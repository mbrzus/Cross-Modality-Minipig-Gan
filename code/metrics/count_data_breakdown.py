import json

with open("../metadata/structure.json") as f:
    data = json.load(f)

def count_data_use(data_use):
    sub_count = 0
    count = 0
    for sub in data[data_use]:
        new_sub = False
        for sess in data[data_use][sub]:
            if len(data[data_use][sub][sess]['t1w']) > 0 and len(data[data_use][sub][sess]['t2w']) > 0:
                count += 1
                new_sub = True
        if new_sub:
            sub_count += 1
    print(f"{data_use} sessions: {count}")
    print(f"{data_use} subjects: {sub_count}")

count_data_use('train')
count_data_use('test')
count_data_use('validation')