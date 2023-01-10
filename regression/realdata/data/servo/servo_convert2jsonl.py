#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from functools import partial
import pandas as pd
import random


# In[2]:


def data2text(row, integer = False, label = True):
    prompt = "When we have " 
    for i in range(1,len(row)-label):
        if integer:
            if row[i] == 'A':
                row_i = 1
            elif row[i] == 'B':
                row_i = 2
            elif row[i] == 'C':
                row_i = 3
            elif row[i] == 'D':
                row_i = 4
            elif row[i] == 'E':
                row_i = 5
            else:
                row_i = row[i]
            prompt += "x%d=%d, " % (i, row[i])
        else:
            if row[i] == 'A':
                row_i = 1
            elif row[i] == 'B':
                row_i = 2
            elif row[i] == 'C':
                row_i = 3
            elif row[i] == 'D':
                row_i = 4
            elif row[i] == 'E':
                row_i = 5
            else:
                row_i = row[i]
#             print(row_i,type(row_i))
            prompt += "x%d=%.4f, " % (i, row_i) 
#     print(prompt)
    prompt += "what should be the y value?"
    if not label:
        return "%s###" % prompt
    else:
        if integer:
            completion = "%d" % row['C']
        else:
            completion = "%.4f" % row['C']
        return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)

def df2jsonl(df, filename, integer = False):
    jsonl = '\n'.join(df.apply(func = partial(data2text, integer = integer), axis = 1).tolist())
    with open(os.path.join(filename), 'w') as f:
        f.write(jsonl)


# In[3]:


"""
Given a servomechanism with motor ### and screw ###, 
with voltage gain = ###, power gain = ###,
what's its rise time?

The servomechanism's rise time is ###
"""
def data2text_feature_name(row, integer = False, label = True):
    prompt = "Given a servomechanism, with motor %s and screw %s, whose voltage gain is %d and power gain is %d, what is its rise time?" % (row['M'],row['S'],row['P'],row['V'])
    
    completion = "%.4f" % row['C']
    return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)

def df2jsonl_feature_name(df, filename, integer = False):
    jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name, integer = integer), axis = 1).tolist())
    with open(os.path.join(filename), 'w') as f:
        f.write(jsonl)


# In[4]:


# split the dataset - with feature names
data = pd.read_csv("./servo.data", sep=",")
n = len(data)
print("Number of total samples:",n)
idx = np.arange(n)
random.shuffle(idx)
num_training = int(.7*n)
print("Number of total training samples:",num_training)

train_idx, valid_idx, test_idx = idx[:int(.7*n)], idx[int(.7*n):int(.85*n)], idx[int(.85*n):]
train_idx_20 = train_idx[:int(0.2*num_training)]
print("Number of 20% training samples:",len(train_idx_20))
train_idx_40 = train_idx[:int(0.4*num_training)]
print("Number of 40% training samples:",len(train_idx_40))
train_idx_60 = train_idx[:int(0.6*num_training)]
print("Number of 60%  training samples:",len(train_idx_60))
train_idx_80 = train_idx[:int(0.6*num_training)]
print("Number of 80%  training samples:",len(train_idx_80))

print("Number of validation samples:",len(valid_idx))
print("Number of testing samples:",len(test_idx))

data.loc[train_idx].to_csv("./servo_train_full.csv", sep=",")
data.loc[train_idx_20].to_csv("./servo_train_20.csv", sep=",")
data.loc[train_idx_40].to_csv("./servo_train_40.csv", sep=",")
data.loc[train_idx_60].to_csv("./servo_train_60.csv", sep=",")
data.loc[train_idx_80].to_csv("./servo_train_80.csv", sep=",")
data.loc[test_idx].to_csv("./servo_test.csv", sep=",")
data.loc[valid_idx].to_csv("./servo_valid.csv", sep=",")


# In[5]:


# prompts with feature names
train_data = pd.read_csv("./servo_train_full.csv", sep=",")
df2jsonl_feature_name(train_data,"servo_fn_full_train.jsonl")

train_data = pd.read_csv("./servo_train_20.csv", sep=",")
df2jsonl_feature_name(train_data,"servo_fn_20_train.jsonl")

train_data = pd.read_csv("./servo_train_40.csv", sep=",")
df2jsonl_feature_name(train_data,"servo_fn_40_train.jsonl")

train_data = pd.read_csv("./servo_train_60.csv", sep=",")
df2jsonl_feature_name(train_data,"servo_fn_60_train.jsonl")

train_data = pd.read_csv("./servo_train_80.csv", sep=",")
df2jsonl_feature_name(train_data,"servo_fn_80_train.jsonl")

test_data = pd.read_csv("./servo_test.csv", sep=",")
df2jsonl_feature_name(test_data,"servo_fn_test.jsonl")

valid_data = pd.read_csv("./servo_valid.csv", sep=",")
df2jsonl_feature_name(valid_data,"servo_fn_valid.jsonl")


# In[6]:


# prompts without feature names
train_data = pd.read_csv("./servo_train_full.csv", sep=",")
df2jsonl(train_data,"servo_full_train.jsonl")

train_data = pd.read_csv("./servo_train_20.csv", sep=",")
df2jsonl(train_data,"servo_20_train.jsonl")

train_data = pd.read_csv("./servo_train_40.csv", sep=",")
df2jsonl(train_data,"servo_40_train.jsonl")

train_data = pd.read_csv("./servo_train_60.csv", sep=",")
df2jsonl(train_data,"servo_60_train.jsonl")

train_data = pd.read_csv("./servo_train_80.csv", sep=",")
df2jsonl(train_data,"servo_80_train.jsonl")

test_data = pd.read_csv("./servo_test.csv", sep=",")
df2jsonl(test_data,"servo_test.jsonl")

valid_data = pd.read_csv("./servo_valid.csv", sep=",")
df2jsonl(valid_data,"servo_valid.jsonl")


# In[7]:


# train_data = pd.read_csv("./servo_train.csv", sep=",")
# df2jsonl(train_data,"servo_train.jsonl")

# test_data = pd.read_csv("./servo_test.csv", sep=",")
# df2jsonl(test_data,"servo_test.jsonl")

# valid_data = pd.read_csv("./servo_valid.csv", sep=",")
# df2jsonl(valid_data,"servo_valid.jsonl")


# In[8]:


# convert to numerical data
train_data = pd.read_csv("./servo_train_full.csv", sep=",")
train_data_num = train_data.copy()
train_data_num.loc[train_data['S'] == 'A', 'S'] = 1
train_data_num.loc[train_data['S'] == 'B', 'S'] = 2
train_data_num.loc[train_data['S'] == 'C', 'S'] = 3
train_data_num.loc[train_data['S'] == 'D', 'S'] = 4
train_data_num.loc[train_data['S'] == 'E', 'S'] = 5
train_data_num.loc[train_data['M'] == 'A', 'M'] = 1
train_data_num.loc[train_data['M'] == 'B', 'M'] = 2
train_data_num.loc[train_data['M'] == 'C', 'M'] = 3
train_data_num.loc[train_data['M'] == 'D', 'M'] = 4
train_data_num.loc[train_data['M'] == 'E', 'M'] = 5
train_data_num.to_csv("./servo_train_full_num.csv", sep=",")

train_data = pd.read_csv("./servo_train_20.csv", sep=",")
train_data_num = train_data.copy()
train_data_num.loc[train_data['S'] == 'A', 'S'] = 1
train_data_num.loc[train_data['S'] == 'B', 'S'] = 2
train_data_num.loc[train_data['S'] == 'C', 'S'] = 3
train_data_num.loc[train_data['S'] == 'D', 'S'] = 4
train_data_num.loc[train_data['S'] == 'E', 'S'] = 5
train_data_num.loc[train_data['M'] == 'A', 'M'] = 1
train_data_num.loc[train_data['M'] == 'B', 'M'] = 2
train_data_num.loc[train_data['M'] == 'C', 'M'] = 3
train_data_num.loc[train_data['M'] == 'D', 'M'] = 4
train_data_num.loc[train_data['M'] == 'E', 'M'] = 5
train_data_num.to_csv("./servo_train_20_num.csv", sep=",")

train_data = pd.read_csv("./servo_train_40.csv", sep=",")
train_data_num = train_data.copy()
train_data_num.loc[train_data['S'] == 'A', 'S'] = 1
train_data_num.loc[train_data['S'] == 'B', 'S'] = 2
train_data_num.loc[train_data['S'] == 'C', 'S'] = 3
train_data_num.loc[train_data['S'] == 'D', 'S'] = 4
train_data_num.loc[train_data['S'] == 'E', 'S'] = 5
train_data_num.loc[train_data['M'] == 'A', 'M'] = 1
train_data_num.loc[train_data['M'] == 'B', 'M'] = 2
train_data_num.loc[train_data['M'] == 'C', 'M'] = 3
train_data_num.loc[train_data['M'] == 'D', 'M'] = 4
train_data_num.loc[train_data['M'] == 'E', 'M'] = 5
train_data_num.to_csv("./servo_train_40_num.csv", sep=",")

train_data = pd.read_csv("./servo_train_60.csv", sep=",")
train_data_num = train_data.copy()
train_data_num.loc[train_data['S'] == 'A', 'S'] = 1
train_data_num.loc[train_data['S'] == 'B', 'S'] = 2
train_data_num.loc[train_data['S'] == 'C', 'S'] = 3
train_data_num.loc[train_data['S'] == 'D', 'S'] = 4
train_data_num.loc[train_data['S'] == 'E', 'S'] = 5
train_data_num.loc[train_data['M'] == 'A', 'M'] = 1
train_data_num.loc[train_data['M'] == 'B', 'M'] = 2
train_data_num.loc[train_data['M'] == 'C', 'M'] = 3
train_data_num.loc[train_data['M'] == 'D', 'M'] = 4
train_data_num.loc[train_data['M'] == 'E', 'M'] = 5
train_data_num.to_csv("./servo_train_60_num.csv", sep=",")

train_data = pd.read_csv("./servo_train_80.csv", sep=",")
train_data_num = train_data.copy()
train_data_num.loc[train_data['S'] == 'A', 'S'] = 1
train_data_num.loc[train_data['S'] == 'B', 'S'] = 2
train_data_num.loc[train_data['S'] == 'C', 'S'] = 3
train_data_num.loc[train_data['S'] == 'D', 'S'] = 4
train_data_num.loc[train_data['S'] == 'E', 'S'] = 5
train_data_num.loc[train_data['M'] == 'A', 'M'] = 1
train_data_num.loc[train_data['M'] == 'B', 'M'] = 2
train_data_num.loc[train_data['M'] == 'C', 'M'] = 3
train_data_num.loc[train_data['M'] == 'D', 'M'] = 4
train_data_num.loc[train_data['M'] == 'E', 'M'] = 5
train_data_num.to_csv("./servo_train_80_num.csv", sep=",")

train_data = pd.read_csv("./servo_test.csv", sep=",")
train_data_num = train_data.copy()
train_data_num.loc[train_data['S'] == 'A', 'S'] = 1
train_data_num.loc[train_data['S'] == 'B', 'S'] = 2
train_data_num.loc[train_data['S'] == 'C', 'S'] = 3
train_data_num.loc[train_data['S'] == 'D', 'S'] = 4
train_data_num.loc[train_data['S'] == 'E', 'S'] = 5
train_data_num.loc[train_data['M'] == 'A', 'M'] = 1
train_data_num.loc[train_data['M'] == 'B', 'M'] = 2
train_data_num.loc[train_data['M'] == 'C', 'M'] = 3
train_data_num.loc[train_data['M'] == 'D', 'M'] = 4
train_data_num.loc[train_data['M'] == 'E', 'M'] = 5
train_data_num.to_csv("./servo_test_num.csv", sep=",")

train_data = pd.read_csv("./servo_valid.csv", sep=",")
train_data_num = train_data.copy()
train_data_num.loc[train_data['S'] == 'A', 'S'] = 1
train_data_num.loc[train_data['S'] == 'B', 'S'] = 2
train_data_num.loc[train_data['S'] == 'C', 'S'] = 3
train_data_num.loc[train_data['S'] == 'D', 'S'] = 4
train_data_num.loc[train_data['S'] == 'E', 'S'] = 5
train_data_num.loc[train_data['M'] == 'A', 'M'] = 1
train_data_num.loc[train_data['M'] == 'B', 'M'] = 2
train_data_num.loc[train_data['M'] == 'C', 'M'] = 3
train_data_num.loc[train_data['M'] == 'D', 'M'] = 4
train_data_num.loc[train_data['M'] == 'E', 'M'] = 5
train_data_num.to_csv("./servo_valid_num.csv", sep=",")

