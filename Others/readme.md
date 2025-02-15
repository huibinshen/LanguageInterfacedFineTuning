# LIFT on Advanced Regression Tasks

In this folder, we perform experiments on advanced regression tasks including

* Does Contradicting Contextual Information Hurt LIFT?
* Can LIFT Perform Ridge Regression?

## Does Contradicting Contextual Information Hurt LIFT?

Working directory: `contradictingPrompts`

### Loading python modules

```python
from contradictingPrompts import *
```

### Running experiments on LIFT

Note that we generate the datasets and finetuning GPT-3 first by calling the following function. It will conducting multiple experiments by adding the setting to the parameter list. For example, if you specify `mode_list = ['standard', 'correct']` and `n_train_list= [4,8]`, calling the following function will generate datasets and running 4 groups of experiments on datasets with: i) 4 standard training prompts, ii) 8 standard training prompts, iii) 4 correct training prompts, and iv) 8 correct training prompts. 

```python
run_setting_gpt3(
  data_dir, # the directory for saving the datasets and experiment results
  mode_list = ['standard', 'correct', 'contradict'], # the set of prompts 
  n_train_list = [4, 8, 16, 32, 64, 125, 256, 512, 1024], # the number of training samples
  n_valid = 50, # number of validation samples
  n_test = 100, # number of test samples
  num_epochs = 10, # number of epochs
  batch_size = 5, # batch size
  lr_list = [0.05, 0.1, 0.2], # select learning rate multiplier from this list 
  openai_key = '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' # openai key
)
```

Then, we need to finetune GPT-J based on the datasets generated by the code above. 

```python
run_setting_gptj(
  data_dir_list, # Run GPT-J on the following data directories -> 
  cuda_idx = 0, # use cuda:0
  epochs = [2,6,10] # select epochs from this list
)
```

Then, all the experiment results will be saved in files end with `all.json`.

## Can LIFT Perform Ridge Regression?

Working directory: `ridge`

### Loading python modules

```python
from ridge import *
```

### Running experiments on LIFT

Most of the parameters are identical to what we have introduced above. 

```python
run_setting_gpt3(
	data_dir, 
	lbd_list = [0, 10, 50, 100, 1000], # the list of penalty
	n_train = 200, 
	n_valid = 50,
  n_test = 100, 
  p_list = [1,10,50,100], # list of number of features
  num_epochs = 10, 
  batch_size = 5, 
  lr_list = [0.05, 0.1, 0.2], 
  openai_key = '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'
 )
```

Then, we finetune GPT-J. 

```python
run_setting_gptj(
	data_dir_list, 
	cuda_idx = 0, 
	epochs = [2,6,10], 
	batch_size = 4
)
```

Then, all the experiment results will be saved in files end with `all.json`.