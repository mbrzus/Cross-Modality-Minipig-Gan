import json
import matplotlib.pyplot as plt

with open("mean_absolute_error.json") as f:
    data = json.load(f)

mae_percents = []
for session in data:
    # print(data[session])
    mae_percent =  data[session]['t2gen_vs_t2gt'] / data[session]['t1gt_vs_t2gt'] * 100
    mae_percents.append(mae_percent)

fig1, ax1 = plt.subplots()
ax1.set_title("Mean Absolute Error Generated vs. Identity Percentage")
ax1.boxplot(mae_percents)