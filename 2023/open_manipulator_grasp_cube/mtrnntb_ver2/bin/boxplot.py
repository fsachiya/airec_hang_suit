import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# test_losses = []
# for idx in [1,3]:
#     test_loss_idx = []
#     for tau in [2,4,6]:
#         test_loss_tau = []
#         result_json_path = f'./output/test_pos_{idx}/fast_tau_{tau}/*/result.json'
#         result_json_files = glob.glob(result_json_path)
#         for result_json_file in result_json_files:
#             with open(result_json_file) as f:
#                 result = json.load(f)
#                 test_loss_tau.append(result["loss"]["test"])
#         test_loss_idx.append(test_loss_tau)
#     test_losses.append(test_loss_idx)

normal_test_losses = []
for idx in [1,3]:
    # normal_ver
    test_loss_idx = []
    for tau in [2,4,6]:
        test_loss_tau = []
        result_json_path = f'/home/fujita/job/2023/open_manipulator_grasp_cube/mtrnn_ver/output/test_pos_{idx}/fast_tau_{tau}/*/result.json'
        result_json_files = glob.glob(result_json_path)
        for result_json_file in result_json_files:
            with open(result_json_file) as f:
                result = json.load(f)
                test_loss_tau.append(result["loss"]["test"])
        test_loss_idx.append(test_loss_tau)
    normal_test_losses.append(test_loss_idx)

tb_test_losses = []
for idx in [1,3]:
        # tb_ver
    test_loss_idx = []
    result_json_path = f'./output/test_pos_{idx}/*/result.json'
    result_json_files = glob.glob(result_json_path)
    for result_json_file in result_json_files:
        with open(result_json_file) as f:
            result = json.load(f)
            test_loss_idx.append(result["loss"]["test"])
    tb_test_losses.append(test_loss_idx)

# import ipdb; ipdb.set_trace()
normal_test_losses = np.array(normal_test_losses)
tb_test_losses = np.array(tb_test_losses).reshape(2,1,-1)
test_losses = np.concatenate([normal_test_losses, tb_test_losses], 1)

# for i, idx in enumerate([1,3]):
#     fig = plt.figure()
#     plt.boxplot(test_losses[i].T, labels=["tau_2", "tau_4", "tau_6", "tb_tau"])
#     plt.savefig(f'./output/test_pos_{idx}/boxplot.png')
#     plt.clf()
#     plt.close()




# sns.set()
# sns.set_style('whitegrid')
# sns.set_palette('Set3')

# np.random.seed(2018)

# df = pd.DataFrame({
#     'leaf': np.random.normal(10, 2, 20),
#     'stem': np.random.normal(15, 3, 20),
#     'root': np.random.normal(5, 1, 20)
# })

# df_melt = pd.melt(df)
# print(df_melt.head())
# ##   variable      value
# ## 0     leaf   9.446465
# ## 1     leaf  11.163702
# ## 2     leaf  14.296799
# ## 3     leaf   7.441026
# ## 4     leaf  11.004554

for i, idx in enumerate([1,3]):
    fig, ax = plt.subplots()
    sns.boxplot(data=test_losses[i].T, ax=ax)
    p = sns.stripplot(data=test_losses[i].T, jitter=True, color='black', ax=ax)
    p.set_title(f'boxplot at pos{idx}', y=1.0)
    x_ticks = [0, 1, 2, 3]  # 目盛りの位置を適切に設定
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["tau_2", "tau_4", "tau_6", "tb_tau"])
    ax.set_xlabel("models")
    ax.set_ylabel("test_loss")
    plt.savefig(f'./output/test_pos_{idx}/boxplot.png')
    plt.clf()
    plt.close()
