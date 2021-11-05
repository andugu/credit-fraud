# Train stage
import pandas as pd
import xgboost as xgb
import yaml
import json
import pickle
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score


def get_confusion_coefs(y_actual, y_pred):
    """  Returns the true positives rate & the true negatives rate for a set of y_real values and y_predicted values """
    cf = confusion_matrix(y_actual, y_pred)
    tnr = cf[0][0] / (cf[0][0]+cf[1][0])
    tpr = cf[1][1] / (cf[1][1]+cf[0][1])
    return tpr, tnr


def optimize_classifier(train, init_points, n_iter, cv, stratified, shuffle, num_boost_round, balanced,
                        scale_pos_weight) -> dict:
    """ Performs bayesian optimization over the most significant XGBoost parameters and returns the ones with
    the highest score """

    bounds = {'max_depth': (4, 10),
              'reg_alpha': (0.0, 1),
              'reg_lambda': (0.0, 1),
              'gamma': (0, 100),
              'min_child_weight': (1, 20),
              'max_delta_step': (0, 10),
              'subsample': (0.5, 1),
              'colsample_bytree': (0, 1),
              'learning_rate': (0.01, 0.3)
              }

    def get_score(max_depth, reg_alpha, reg_lambda, gamma, min_child_weight, max_delta_step,
                  subsample, colsample_bytree, learning_rate):
        params = {'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'eval_metric': 'aucpr',
                  'scale_pos_weight': scale_pos_weight,
                  'is_balanced': balanced,
                  'max_depth': int(max_depth),
                  'reg_alpha': reg_alpha,
                  'reg_lambda': reg_lambda,
                  'gamma': gamma,
                  'min_child_weight': int(min_child_weight),
                  'max_delta_step': max_delta_step,
                  'subsample': subsample,
                  'colsample_bytree': colsample_bytree,
                  'learning_rate': learning_rate,
                  'verbosity': 0,
                  'n_jobs': -1,
                  'use_label_encoder': False
                  }

        results = xgb.cv(params, train, num_boost_round=num_boost_round, nfold=cv, stratified=stratified,
                         shuffle=shuffle)
        return (results['train-aucpr-mean'].iloc[-1] * 0.4
                + results['test-aucpr-mean'].iloc[-1] * 0.6
                - abs(results['train-aucpr-mean'].iloc[-1] - results['test-aucpr-mean'].iloc[-1]) * 5)

    optimizer = BayesianOptimization(get_score, bounds)
    optimizer.probe({'max_depth': 6,
                     'reg_alpha': 0,
                     'reg_lambda': 1,
                     'gamma': 0,
                     'min_child_weight': 1,
                     'max_delta_step': 0,
                     'subsample': 1,
                     'colsample_bytree': 1,
                     'learning_rate': 0.3},
                    lazy=True)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    best_params = {'booster': 'gbtree',
                   'objective': 'binary:logistic',
                   'is_balanced': balanced,
                   'eval_metric': 'aucpr',
                   'scale_pos_weight': scale_pos_weight,
                   'max_depth': int(optimizer.max['params']['max_depth']),
                   'reg_alpha': optimizer.max['params']['reg_alpha'],
                   'reg_lambda': optimizer.max['params']['reg_lambda'],
                   'gamma': optimizer.max['params']['gamma'],
                   'min_child_weight': int(optimizer.max['params']['min_child_weight']),
                   'max_delta_step': optimizer.max['params']['max_delta_step'],
                   'subsample': optimizer.max['params']['subsample'],
                   'colsample_bytree': optimizer.max['params']['colsample_bytree'],
                   'learning_rate': optimizer.max['params']['learning_rate'],
                   'verbosity': 0,
                   'n_jobs': -1,
                   'use_label_encoder': False
                   }
    return best_params


def main():
    params = yaml.safe_load(open('params.yaml'))['train']
    test_size = params['test_size']
    val_size = params['val_size']
    n_iter = params['n_iter']
    init_points = params['init_points']
    cv = params['cv']
    stratified = params['stratified']
    shuffle = params['shuffle']
    balanced = params['balanced']
    num_boost_round = params['num_boost_round']
    scale_pos_weight = params['scale_pos_weight']

    # Load data
    features = pd.read_pickle('data/features.pkl')
    labels = pd.read_pickle('data/labels.pkl')

    # Manage default parameters values
    if scale_pos_weight is None:
        scale_pos_weight = labels.value_counts()[0]/labels.value_counts()[1]

    # Split data in 3 sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, stratify=labels)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=val_size, stratify=y_test)

    train_data = xgb.DMatrix(x_train, label=y_train)
    test_data = xgb.DMatrix(x_test, label=y_test)
    val_data = xgb.DMatrix(x_val, label=y_val)

    # Look for the best parameters with bayesian optimization
    best_parameters = optimize_classifier(train_data, init_points, n_iter, cv, stratified, shuffle,
                                          num_boost_round, balanced, scale_pos_weight)

    # Train the model
    model = xgb.train(best_parameters, train_data, num_boost_round=num_boost_round,
                      evals=[(train_data, 'train'), (test_data, 'test')], verbose_eval=False)
    # Save Model
    with open('pickles/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # AUC
    train_score = model.eval(train_data)
    test_score = model.eval(test_data)
    val_score = model.eval(val_data)

    # Get test tpr, tnr & recall
    test_predictions = model.predict(test_data)
    test_predictions = np.round(test_predictions)
    test_tpr, test_tnr = get_confusion_coefs(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)

    # Get val tpr, tnr & recall
    val_predictions = model.predict(val_data)
    val_predictions = np.round(val_predictions)
    val_tpr, val_tnr = get_confusion_coefs(y_val, val_predictions)
    val_recall = recall_score(y_val, val_predictions)

    # Dump Metrics
    results = {
        'train-aucpr': float(train_score[train_score.find(':')+1:]),
        'test-aucpr': float(test_score[test_score.find(':')+1:]),
        'test-tpr': test_tpr,
        'test-tnr': test_tnr,
        'test-recall': test_recall,
        'val-aucpr': float(val_score[val_score.find(':')+1:]),
        'val-tpr': val_tpr,
        'val-tnr': val_tnr,
        'val-recall': val_recall
    }
    with open('pickles/metrics.json', 'w') as file:
        json.dump(results, file, indent=4)


if __name__ == '__main__':
    main()
