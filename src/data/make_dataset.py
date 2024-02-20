# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib 
import yaml
import sys

# load data
def load_data(path):
    df=pd.read_csv(path)
    return df

# train test split
def split_data(df,test_split,seed):
    #splitting data into train and test
    train, test = train_test_split(df, test_size=test_split, random_state=seed)
    return train, test

# save data
def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)


def main():
    curr_dir =pathlib.Path(__file__)
    print(curr_dir)
    home_dir=curr_dir.parent.parent.parent
    param_file_path=home_dir.as_posix()+'/params.yaml'
    params=(yaml.safe_load(open(param_file_path))['make_dataset'])
    dvc_file_path=(home_dir.as_posix()+'/dvc.yaml')
    params_dvc=(yaml.safe_load(open(dvc_file_path))['stages']['make_dataset'])    
    input_file=params_dvc['deps'][0]
    output_path = home_dir.as_posix() + params_dvc['outs'][0]
    data_path=(home_dir.as_posix() + input_file)
    print(data_path)
    df=load_data(data_path)
    train,test=split_data(df,params['params']['test_split'],params['params']['seed'])
    save_data(train,test,output_path)

    
    
    
if __name__ == "__main__":
    main()
