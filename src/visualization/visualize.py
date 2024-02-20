import pathlib
import joblib
import sys
import yaml
import pandas as pd
from sklearn import metrics
from sklearn import tree
from dvclive import Live
from matplotlib import pyplot as plt


def evaluate(model, X, y, split, live, save_path):
    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        X (pandas.DataFrame): Input DF.
        y (pamdas.Series): Target column.
        split (str): Dataset name.
        live (dvclive.Live): Dvclive instance.
        save_path (str): Path to save the metrics.
    """

    predictions_by_class = model.predict_proba(X)
    predictions = predictions_by_class[:, 1]

    # Use dvclive to log a few simple metrics...
    avg_prec = metrics.average_precision_score(y, predictions)
    roc_auc = metrics.roc_auc_score(y, predictions)
    


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    print(home_dir)
    # TODO - Optionally add visualization params as well
    # params_file = home_dir.as_posix() + '/params.yaml'
    # params = yaml.safe_load(open(params_file))["train_model"]

    dvc_file_path = home_dir.as_posix() + '/dvc.yaml'
    params_dvc=yaml.safe_load(open(dvc_file_path))['stages']['models']
    model_path=params_dvc['deps'][0]
    print(model_path)
    
    # Load the model.
    model = joblib.load(model_path)
    print(model)
    
    # Load the data.
    input_file = params_dvc['deps'][1]
    data_path = home_dir.as_posix() + input_file
    print(data_path)
    output_path = home_dir.as_posix() + '/dvclive'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    print(output_path)
    #TARGET = 'Class'
    train_features = pd.read_csv(data_path + 'train.csv')
    #X_train = train_features.drop(TARGET, axis=1)
    #y_train = train_features[TARGET]
    #feature_names = X_train.columns.to_list()

    #test_features = pd.read_csv(data_path + '/test.csv')
    #X_test = test_features.drop(TARGET, axis=1)
    #y_test = test_features[TARGET]

    # Evaluate train and test datasets.
    #with Live(output_path, dvcyaml=False) as live:
        #evaluate(model, X_train, y_train, "train", live, output_path)
        #evaluate(model, X_test, y_test, "test", live, output_path)

        # Dump feature importance plot.
        #save_importance_plot(live, model, feature_names)

if __name__ == "__main__":
    main()
