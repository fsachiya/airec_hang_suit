import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# tau_2 = [0.016695678234100342,
#          0.011217322200536728,
#          0.00760648213326931,
#          0.006161264143884182,
#          0.007880572229623795,
#          0.009736238978803158,
#          0.006762391421943903,
#          0.009089922532439232]

test_losses = []
for idx in [1,3]:
    test_loss_idx = []
    for tau in [2,4,6]:
        test_loss_tau = []
        result_json_path = f'./output/test_pos_{idx}/fast_tau_{tau}/*/result.json'
        result_json_files = glob.glob(result_json_path)
        for result_json_file in result_json_files:
            with open(result_json_file) as f:
                result = json.load(f)
                test_loss_tau.append(result["loss"]["test"])
        test_loss_idx.append(test_loss_tau)
    test_losses.append(test_loss_idx)

test_losses = np.array(test_losses)
# df_1 = pd.DataFrame(test_losses[0])

for i, idx in enumerate([1,3]):
    fig = plt.figure()
    plt.boxplot(test_losses[i].T, labels=["tau_2", "tau_4", "tau_6"])
    plt.savefig(f'./output/test_pos_{idx}/boxplot.png')
    plt.clf()
    plt.close()

