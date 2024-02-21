import json
import pickle

#
with open('./data_use_for_graph/array.pkl', 'rb') as f:
    arrays = pickle.load(f)

with open('./data_use_for_graph/test_data.json', 'r') as f:
    data = json.load(f)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

model_dic = arrays['model']
# to keep plot legend sequence 1-4
model_lst = sorted([(key, value) for key, value in model_dic.items()], key=lambda x: x[1], reverse=True)
agents_dic = arrays['agent']
level_dic = arrays['level']
won_rate = arrays['won_rate']
dic_id_name = {'1-1': 'TransRL-1',
               '1-2': 'TransRL-2',
               '1-3': 'TransRL-3',
               '1-4': 'TransRL-4',
               }
dic_id_marker = {'1-1': 'o',
                 '1-2': 'v',
                 '1-3': 's',
                 '1-4': '^',
                 }
dic_id_level = {'1-1': 10,
                '1-2': 11,
                '1-3': 13,
                '1-4': 14,
                }
dic_id_color = {'1-1': 'tab:red',
                '1-2': 'tab:blue',
                '1-3': 'tab:green',
                '1-4': 'tab:orange',
                }
dic_level_segment = {10: 1,
                     11: 2,
                     13: 3,
                     14: 4,
                     }
sizeH = 4
sizeV = 1.65
fig, ax = plt.subplots(sharey=True, figsize=[sizeH, sizeV], tight_layout=True)
for model_id, i in model_lst:
    # Convert agents_dic.values() to a list and use proper indexing for won_rate
    agents = list(agents_dic.values())
    level_index = level_dic[14]
    won_rate_data = won_rate[i, agents, level_index] if won_rate.ndim == 3 else won_rate[i, level_index]

    ax.plot(agents_dic.keys(), won_rate_data, marker=dic_id_marker[model_id],
            label=dic_id_name[model_id], color=dic_id_color[model_id])  # Call plot on ax, not fig
    ax.set(xlabel='Number of UAVs', ylabel='Arrival rate')
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
ax.set_ylim(0.4, 1.0)
ax.grid()
# plt.title('Arrival rate on cylinder-torus-torus-cylinder')
plt.legend(loc='center')
fig.savefig('./pic/test_14.jpg')
fig.savefig('./pic/test_14.pdf')
plt.show()

fig, ax = plt.subplots(sharey=True, figsize=[sizeH, sizeV], tight_layout=True)
for model_id, i in model_lst:
    # Convert agents_dic.values() to a list and use proper indexing for won_rate
    agents = list(agents_dic.values())
    level_index = level_dic[14]
    # print(level_dic[dic_level[model_id]])
    won_rate_data = won_rate[i, agents, level_dic[dic_id_level[model_id]]] if won_rate.ndim == 3 else won_rate[
        i, level_index]

    ax.plot(agents_dic.keys(), won_rate_data, marker=dic_id_marker[model_id],
            label=dic_id_name[model_id], color=dic_id_color[model_id])  # Call plot on ax, not fig
    ax.set(xlabel='Number of UAVs', ylabel='Arrival rate')
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
ax.set_ylim(0.87, 1.02)
ax.grid()
# plt.title('Arrival rate on cylinder-torus-torus-cylinder')
plt.legend(loc='center')
fig.savefig('./pic/test_each_level.jpg')
fig.savefig('./pic/test_each_level.pdf')
plt.show()

fig, ax = plt.subplots(sharey=True, figsize=[sizeH, sizeV], tight_layout=True)

for model_id, i in model_lst:
    # Convert agents_dic.values() to a list and use proper indexing for won_rate
    agents = 5
    level_index = [level_dic[i] for i in dic_id_level.values()]
    won_rate_data = won_rate[i, agents, level_index] if won_rate.ndim == 3 else won_rate[i, level_index]

    ax.plot(dic_level_segment.values(), won_rate_data, marker=dic_id_marker[model_id],
            label=dic_id_name[model_id], color=dic_id_color[model_id])  # Call plot on ax, not fig
    ax.set(xlabel='Number of corridors', ylabel='Arrival rate')
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
ax.set_ylim(0.40, 1.05)
ax.grid()
# plt.title('Arrival rate on cylinder-torus-torus-cylinder')
plt.legend(loc='center')
fig.savefig('./pic/test_5_UAV.jpg')
fig.savefig('./pic/test_5_UAV.pdf')
plt.show()

##########################################################3
import pandas as pd

plot_max_step = 6.1 * 1e6
dfs = [pd.read_csv(f'./reward/1-{i}.csv') for i in range(1, 5)]
fig, axs = plt.subplots(ncols=2, figsize=[9, 1.95], tight_layout=True)  # Use subplots to create a figure and an axes
lables = list(dic_id_name.values())
for i, df in enumerate(dfs):
    index_step = 0
    for step in df['Step']:
        if step < plot_max_step:
            index_step += 1
    print(i)
    axs[0].plot(df['Step'][:index_step], df['Value'][:index_step], label=lables[i],
                color=list(dic_id_color.values())[i])
axs[0].set(xlabel='Number of steps\n(a)', ylabel='Normalized\ncumulative reward')
axs[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
# # ax.set_ylim(0.40, 1.05)
axs[0].grid()
# 'lower right'
plt.legend(loc=4)

dfs = [pd.read_csv(f'./arrival/1-{i}.csv') for i in range(1, 5)]
lables = list(dic_id_name.values())
for i, df in enumerate(dfs):
    index_step = 0
    for step in df['Step']:
        if step < plot_max_step:
            index_step += 1
    print(i)
    axs[1].plot(df['Step'][:index_step], df['Value'][:index_step], label=lables[i],
                color=list(dic_id_color.values())[i])
axs[1].set(xlabel='Number of steps\n(b)', ylabel='Normalize\narrival rate')
axs[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
axs[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
# # ax.set_ylim(0.40, 1.05)
axs[1].grid()
# 'lower right'
plt.legend(loc=4)
fig.savefig('./pic/training_process.jpg')
fig.savefig('./pic/training_process.pdf')
plt.show()
print(1)
