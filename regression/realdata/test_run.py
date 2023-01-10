import sys, os, json
sys.path.insert(1, 'utils')
import json

import pandas as pd
import numpy as np
from run_real import run_setting_gptj


def compute_rae():
	test_df = pd.read_csv("data/servo/servo_test.csv")
	y_true = test_df.to_numpy()[:, -1]
	y_mean = np.mean(y_true)

	raes = []
	for sim in range(1,4):
		pred_path = f"test_run_folder/data_{sim}/servo_full_all.json"

		with open(pred_path,'r') as fp:
			pred_json = json.load(fp)

		y_pred = pred_json['gptj_test_y']
		rae = sum(abs(y_pred - y_true)) / sum(abs(y_mean - y_true))
		raes.append(rae)
	print(np.mean(raes), np.std(raes), raes)


if __name__ == "__main__":
# 	run_setting_gptj(
# 	"test_run_folder", 
# 	n_sims = 3, 
# 	data_list = ['servo'], 
# 	lr_list = [0.05, 0.1, 0.2],
# 	prefix_list = ['_', '_fn_'],
# 	pc_list = ['full'],
# 	epochs =  [2,6,10],
# 	cuda_idx = 0,
# 	batch_size = 4
# )
	compute_rae()
