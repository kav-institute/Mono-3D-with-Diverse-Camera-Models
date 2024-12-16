import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report, latex_table

results = [
    "/home/appuser/data/train_depth_CARLA/resnet34_UnitVec/results.json",
    "/home/appuser/data/train_depth_CARLA/resnet34_Deflection/results.json",
    "/home/appuser/data/train_depth_CARLA/resnet34_CameraTensor/results.json",
    "/home/appuser/data/train_depth_CARLA/resnet34_None/results.json",
    "/home/appuser/data/train_depth_CARLA/resnet34_None_FIX_FOV/results.json",
]

names = ["UnitVec", "Deflection", "CameraTensor", "Augmentation", "Pinhole 35°"]

np.random.seed(42)
#pd.set_option('display.max_columns', 7)

data = pd.DataFrame()
for name, jpath in zip(names, results):
    with open(jpath) as json_data:
        config = json.load(json_data)
        json_data.close()
    tmp_list = []
    for FOV_key in config.keys():
        for D_key in config[FOV_key].keys():
            tmp_list.append(config[FOV_key][D_key]["rmse"])
    
    data[name] = np.array(tmp_list)

result = autorank(data, alpha=0.05, verbose=False, order="ascending")

fig, ax = plt.subplots()
ax = plot_stats(result, ax=ax)
plt.savefig("cd_rmse.pdf")
plt.show()