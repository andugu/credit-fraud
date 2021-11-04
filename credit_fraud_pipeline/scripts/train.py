# Train stage
import pandas as pd
import xgboost as xgb
import yaml
import json
import pickle
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix


def optimize_classifier(train, init_points, n_iter, cv, stratified, shuffle, num_boost_round, balanced):

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
                  'is_balanced': balanced,
                  'eval_metric': 'auc',
                  'max_depth': int(max_depth),
                  'reg_alpha': reg_alpha,
                  'reg_lambda': reg_lambda,
                  'gamma': gamma,
                  'min_child_weight': int(min_child_weight),
                  'max_delta_step': max_delta_step,
                  'subsample': subsample,
                  'colsample_bytree': colsample_bytree,
                  'learning_rate': learning_rate,
                  'num_class': None,
                  'verbosity': 0,
                  'n_jobs': -1,
                  'use_label_encoder': False
                  }
        results = xgb.cv(params, train, num_boost_round=num_boost_round, nfold=cv, stratified=stratified,
                         shuffle=shuffle)
        return results['test-auc-mean'].iloc[-1]

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
                   'eval_metric': 'auc',
                   'max_depth': int(optimizer.max['params']['max_depth']),
                   'reg_alpha': optimizer.max['params']['reg_alpha'],
                   'reg_lambda': optimizer.max['params']['reg_lambda'],
                   'gamma': optimizer.max['params']['gamma'],
                   'min_child_weight': int(optimizer.max['params']['min_child_weight']),
                   'max_delta_step': optimizer.max['params']['max_delta_step'],
                   'subsample': optimizer.max['params']['subsample'],
                   'colsample_bytree': optimizer.max['params']['colsample_bytree'],
                   'learning_rate': optimizer.max['params']['learning_rate'],
                   'num_class': None,
                   'verbosity': 0,
                   'n_jobs': -1,
                   'use_label_encoder': False
                   }
    return best_params


def main():
    params = yaml.safe_load(open('params.yaml'))['train']
    test_size = params['test_size']
    n_iter = params['n_iter']
    init_points = params['init_points']
    cv = params['cv']
    stratified = params['stratified']
    shuffle = params['shuffle']
    balanced = params['balanced']
    num_boost_round = params['num_boost_round']

    # Load data
    features = pd.read_pickle('data/features.pkl')
    labels = pd.read_pickle('data/labels.pkl')

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, stratify=labels)
    train_data = xgb.DMatrix(x_train, label=y_train)
    test_data = xgb.DMatrix(x_test, label=y_test)

    # Look for the best parameters with bayesian optimization
    best_parameters = optimize_classifier(train_data, init_points, n_iter, cv, stratified, shuffle,
                                          num_boost_round, balanced)

    # Train the model
    model = xgb.train(best_parameters, train_data, num_boost_round=num_boost_round)

    # Save Model
    pickle.dump(model, open('models/model.pkl', 'wb'))

    # AUC
    train_score = model.eval(train_data)
    test_score = model.eval(test_data)

    # TPR and TNR of testing data
    y_prob = model.predict(test_data)
    # Get concrete prediction
    y_prob = np.round(y_prob)
    cf = confusion_matrix(y_test, y_prob)
    tnr = cf[0][0] / (cf[0][0]+cf[1][0])
    tpr = cf[1][1] / (cf[1][1]+cf[0][1])

    # Dump Metrics
    results = {
        'train-auc': float(train_score[train_score.find(':')+1:]),
        'test-auc': float(test_score[test_score.find(':')+1:]),
        'tpr': tpr,
        'tnr': tnr
    }
    json.dump(results, open('models/metrics.json', 'w'), indent=4)


if __name__ == '__main__':
    main()
