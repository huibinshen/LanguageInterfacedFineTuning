import pandas as pd
import numpy as np
import os, random, itertools, sys, time, json, openai
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sys.path.insert(1, '../../regression/utils')
from GPT3FineTuner import GPT3FineTuner
from GPTJFineTuner import GPTJFineTuner


class dataGenerate(object):
    """
    A class of functions for generating jsonl dataset. 
    """
    def __init__(self,data_dir):
        self.data_dir = data_dir

    def data2text(self, row, integer = False, mode = 'standard', label = True):
        # if mode == 'standard':
        #     prompt = 'x1 and x2 can be used to predict y. '
        # elif mode == 'correct':
        #     prompt = 'x1 plus x2 equals y. '
        # elif mode == 'contradict':
        #     prompt = 'x1 minus x2 equals y. '
        # else:
        #     NotImplementedError('Mode %s is not supported! Please select between standard, contradict, and correct.' % mode)
        prompt = "" 
        for i in range(len(row)-label):
            if integer:
                prompt += "x%d=%d, " % (i+1, row[i])
            else:
                prompt += "x%d=%.4f, " % (i+1, row[i]) 
        if mode == 'standard':
            prompt += "y="
        elif mode == 'contradict':
            prompt += 'x1-x2='
        elif mode == 'correct':
            prompt += 'x1+x2='
        else:
            NotImplementedError('Mode %s is not supported! Please select between standard, contradict, and correct.' % mode)
        prompt += 'what is y?'
        if not label:
            return "%s###" % prompt
        else:
            if integer:
                completion = "%d" % row['y']
            else:
                completion = "%.3f" % row['y']
            return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
    
    def df2jsonl(self, df, filename, integer = False, mode = 'standard'):
        jsonl = '\n'.join(df.apply(func = partial(self.data2text, integer = integer, mode = mode), axis = 1).tolist())
        with open(os.path.join(self.data_dir, filename), 'w') as f:
            f.write(jsonl)
        print("Save a file at:",os.path.join(self.data_dir, filename))
        return os.path.join(self.data_dir, filename)
            
    def array2prompts(self, X, integer = False, mode = 'standard'):
        return list(map(partial(self.data2text, integer = integer, label = False, mode = mode), X))

    def X_generate(self, n, p, integer = False, lb = -10, ub = 10, donut = False):
        if donut: 
            X = np.zeros(p).reshape(1, p)
            while X.shape[0] < n:
                X = np.random.uniform(lb, ub, n*2*p).reshape(n*2,p)
                q1 = lb + (ub-lb) / 3
                q2 = ub - (ub-lb) / 3
                idx = X.T[0] >= q1
                for i in range(p):
                    idx = idx & (X.T[i] >= q1) & (X.T[i] <= q2)
                X = X[~idx]
            X = X[:n]
        else:
            X = np.random.uniform(lb, ub, n*p).reshape(n,p)
            if integer: X = X.round()
        return X
    
    def gridX_generate(self, interval, p, integer, resolution = 100):
        lb, ub = interval
        X_grid = np.linspace(lb * np.ones(p), ub * np.ones(p), resolution).T
        X_grid = np.array(list(itertools.product(*X_grid)))
        grid_prompts = self.array2prompts(X_grid, integer)
        return X_grid, grid_prompts
        
    def data_split(self, X, y, n_train, n_valid):
        n = X.shape[0]
        idx = np.arange(n)
        random.shuffle(idx)
        train_idx, valid_idx = idx[:int(n_train)], idx[int(n_train):]
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        return X_train, X_valid, y_train, y_valid
    
    def linear(self, X, beta):
        return np.dot(X, beta)

    def generate(self, func, mode, n_train, n_valid, n_test, p, integer = False, 
                 noise_level = 0, test_int = None, 
               lb = -10, ub = 10,  beta = None
              ):
        """
        mode: "standard", "contradict", "correct"
        """
        np.random.seed(123)
        if beta is None: beta = np.ones(p) 
        if test_int is None: test_int = (lb, ub)
        
        # Generate x and y   
        n = n_train + n_valid
        X = self.X_generate(n, p, integer, lb, ub)
        X_test = self.X_generate(n_test, p, integer, test_int[0], test_int[1])
        
        y_true = self.linear(X, beta)
        y_test = self.linear(X_test, beta)
        
        y = (y_true + np.random.normal(0,noise_level*np.std(y_true),n)).reshape(n,1)

        if integer: y = y.round()
        
        # split into train, valid, test dataset
        X_train, X_valid, y_train, y_valid = self.data_split(X, y, n_train, n_valid)
  
        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test
        
        train_file = self.df2jsonl(train_df, '%s_%s_n_%d_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_train.jsonl'%(func, mode, n,p,integer,lb,ub,noise_level), integer, mode)
        valid_file = self.df2jsonl(valid_df, '%s_%s_n_%d_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_valid.jsonl'%(func, mode, n,p,integer,lb,ub,noise_level), integer, mode)
        valid_prompts = self.array2prompts(X_valid, integer = integer, mode = mode)
        test_prompts = self.array2prompts(X_test, integer = integer, mode = mode)
            
        return train_df, valid_df, test_df, test_prompts, valid_prompts, train_file, valid_file

def generate_data(data_dir, mode, n_train, n_valid, n_test, 
                 noise_level = 0, test_int = None, 
               lb = -10, ub = 10,  beta = None
              ):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    dataGen = dataGenerate(data_dir)
    return dataGen.generate('linear', mode, n_train, n_valid, n_test, 2, False, noise_level, test_int, lb, ub, beta)

def run_setting_gpt3(data_dir, mode_list = ['standard', 'correct', 'contradict'], n_train_list = [4, 8, 16, 32, 64, 125, 256, 512, 1024], n_valid = 50, n_test = 100,
    num_epochs = 10, batch_size = 5, lr_list = [0.05, 0.1, 0.2], openai_key = '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'):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    openai.api_key = openai_key
    # os.system('export OPENAI_API_KEY="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"')

    p = 2
    config = {'model_type':'ada',"num_epochs":num_epochs,"batch_size":batch_size, 'lr':lr_list}

    counter = 1

    # run exps
    for mode in mode_list:
        for n_train in n_train_list:
            print("------------------Runing group %d---------------------"%counter)
            counter += 1
            train_df, valid_df, test_df, test_prompts, valid_prompts,train_file,valid_file = generate_data(data_dir=data_dir,\
                mode = mode,n_train=n_train,n_test=n_test,n_valid=n_valid)
            
            print("train file saved at: "+train_file)
            print("validation file saved at: "+valid_file)
            
            gpt3_fine_tuner = GPT3FineTuner(config=config,train_jsonl=train_file,valid_jsonl=valid_file, openai_key = openai_key)
            gpt3_fine_tuner.fine_tune()

            file_name = valid_file.split('valid.')[0].replace(",","").replace("(","").replace(")","")+'ft_info.json'            
            y_test_outputs,_,_,_,_ = gpt3_fine_tuner.eval(test_prompts=test_prompts,n_train=n_train,test_df=test_df,training_csv_file_name = file_name, valid_df = valid_df, valid_prompts = valid_prompts,train_df = train_df)

            # save fine-tuned info and results
            with open(valid_file.split('valid.')[0]+'ft_info.json', 'w') as fp:
                json.dump(openai.FineTune.retrieve(id=gpt3_fine_tuner.ft_id).copy(), fp, indent=4)

            tr_ts_vl_json = {"train_x":train_df[train_df.columns[:p]].values.tolist(),"train_y":list(train_df['y']),"validation_x":valid_df[valid_df.columns[:p]].values.tolist(),"validation_y":list(valid_df['y']),
            "test_x":test_df[test_df.columns[:p]].values.tolist(),"test_y":list(test_df['y']),"gpt3_test_y":y_test_outputs, 'openai_key': openai_key, 'ft_id': gpt3_fine_tuner.ft_id,'model_id':gpt3_fine_tuner.ft_info}
            with open(valid_file.split('valid.')[0]+'all.json','w') as fp:
                json.dump(tr_ts_vl_json,fp)
                
def run_setting_gptj(data_dir_list, cuda_idx = 0, epochs = [2,6,10]):

    p = 2
    config = {'learning_rate': 1e-4, 'batch_size': 4, 'epochs':epochs,  'weight_decay': 0.01, 'warmup_steps': 6}
    counter = 1
    dg = dataGenerate('data')
    # run exps
    for data_dir in data_dir_list:
        for file in os.listdir(data_dir):
            if file.endswith('all.json'):
                print("------------------Runing group %d---------------------"%counter)
                counter += 1
            
                with open('%s/%s' % (data_dir, file), 'r') as f:
                    data_json = json.load(f)
                train_df = pd.DataFrame(data_json['train_x'])
                train_df['y'] = data_json['train_y']
                valid_df = pd.DataFrame(data_json['validation_x'])
                valid_df['y'] = data_json['validation_y']
                test_df = pd.DataFrame(data_json['test_x'])
                test_df['y'] = data_json['test_y']
                
                test_prompts = dg.array2prompts(np.array(data_json['test_x']))
                valid_prompts = dg.array2prompts(data_json['validation_x'])
                
                train_file = '%s/%s' % (data_dir, file.split('all.json')[0] + 'train.jsonl')
                valid_file = '%s/%s' % (data_dir, file.split('all.json')[0] + 'valid.jsonl')
                
                n_train = int(file.split('_n_')[1].split('_')[0]) - 50
                
                gptj_fine_tuner = GPTJFineTuner(config=config,train_jsonl=train_file,valid_jsonl=valid_file,cuda_idx=cuda_idx)
                gptj_fine_tuner.fine_tune()
                gptj_test_y, _, _, rmse, rmse_woo = gptj_fine_tuner.eval(test_prompts = test_prompts, 
                    n_train = n_train, 
                    test_df = test_df, 
                    valid_df = valid_df, 
                    valid_prompts = valid_prompts, 
                    plot = False, 
                    X_grid = None,
                    grid_prompts = None,
                    y_grid = None,
                    train_df = train_df
                )

                try:
                    with open(valid_file.split('valid.')[0]+'all.json','r') as fp:
                        tr_ts_vl_json = json.load(fp)
                    
                    tr_ts_vl_json['gptj_test_y'] = gptj_test_y
                except:
                    tr_ts_vl_json = {"train_x":train_df[train_df.columns[:p]].values.tolist(),"train_y":list(train_df['y']),"validation_x":valid_df[valid_df.columns[:p]].values.tolist(),"validation_y":list(valid_df['y']),
                    "test_x":test_df[test_df.columns[:p]].values.tolist(),"test_y":list(test_df['y']),"gptj_test_y":gptj_test_y}
                with open(valid_file.split('valid.')[0]+'all.json','w') as fp:
                    json.dump(tr_ts_vl_json,fp)
                

if __name__ == '__main__':
    data_dir = sys.argv[1]
    openai_key = sys.argv[2]
    run_setting_gpt3(data_dir, mode_list = ['standard', 'correct', 'contradict'], n_train_list = [1, 4, 8, 16, 32, 64, 125, 256, 512], n_valid = 50, n_test = 100,
        num_epochs = 10, batch_size = 5, lr_list = [0.05, 0.1, 0.2], openai_key = openai_key)
