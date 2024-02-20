# train_model.py
import pathlib
import sys
import yaml
import joblib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# fitting the model
def train_model(train_features, target, n_estimators, max_depth, seed):
    # Train your machine learning model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    model.fit(train_features, target)
    return model



# saving model
def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + '/model.joblib')
    

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))['train_model']
    dvc_file_path=(home_dir.as_posix()+'/dvc.yaml')
    params_dvc=(yaml.safe_load(open(dvc_file_path))['stages']['train_model'])
    input_file=params_dvc['deps'][0]
    output_path = params_dvc['outs'][0]
    print(output_path)
    train=pd.read_csv(input_file)
    X=train.drop(columns=['Class'],axis=1)
    y=train['Class']
    model=train_model(X,
                y,
                n_estimators=params['params']['n_estimators'],
                max_depth=params['params']['max_depth'],
                seed=params['params']['seed'])
    save_model(model,output_path)
    
    
    
    
if __name__ == "__main__":
    main()
