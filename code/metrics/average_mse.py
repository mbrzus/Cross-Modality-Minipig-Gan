import json
import matplotlib.pyplot as plt

with open("mean_squared_error.json") as f:
    data = json.load(f)

mae_t2gen = []
mae_t1gt = []
mae_percents = []
for session in data:
    # print(data[session])
    mae_t2gen.append(data[session]['t2gen_vs_t2gt'])
    mae_t1gt.append(data[session]['t1gt_vs_t2gt'])
    mae_percent =  data[session]['t2gen_vs_t2gt'] / data[session]['t1gt_vs_t2gt'] * 100
    mae_percents.append(mae_percent)



print(f"Average Mean Absolute Error t2gen: {sum(mae_t2gen) / len(mae_t2gen)}")
print(f"Average Mean Absolute Error t1gt: {sum(mae_t1gt) / len(mae_t1gt)}")
print(f"Average Mean Absolute Error percentages: {sum(mae_percents) / len(mae_percents)}")