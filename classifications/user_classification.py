from __future__ import annotations, division
from numpy import mean
from numpy import std
from pathlib import Path
from sklearn import ensemble, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.settings import get_models, evaluate_model, selectUsers

def classification(df, users, mode):
    saveFolder = 'D:/Diplomadolgozat/Classification/'
    scores_filename = 'scores_strong_' + str(max(users)) + '.csv'
    boxplots_filename = 'boxplots_strong_' + str(max(users)) + '.png'
    if mode == 1:
        scores_filename = 'scores_' + str(max(users)) + '.csv'
        boxplots_filename = 'boxplots_' + str(max(users)) + '.png'
    Path(saveFolder).mkdir(parents=True, exist_ok=True)
    df = selectUsers(df, users)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    y = np.array(y)
    y = y.reshape(y.shape[0])

    # get the models to evaluate
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    scores_df = pd.DataFrame({})
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
        score_df = pd.DataFrame([[str(name), mean(scores), std(scores)]], columns=['Classifier', 'mean score', 'std score'])
        scores_df = scores_df.append(score_df, ignore_index=True)
    scores_df.to_csv(saveFolder + scores_filename, index=False)
    # plot model performance for comparison
    fig = plt.figure(figsize=(20,5))
    plt.boxplot(results, labels=names, showmeans=True)
    fig.savefig(saveFolder + boxplots_filename)
    plt.close(fig)

def classification2(df, users):
    df = selectUsers(df, users)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    y = np.array(y)
    y = y.reshape(y.shape[0])
 
    model = RandomForestClassifier(n_estimators=100)
    scoring = ['accuracy']
    num_folds = model_selection.StratifiedKFold(n_splits=10, shuffle=True)
    scores = cross_val_score(model, X, y, cv = num_folds)
    for i in range(0,10):
        print('\tFold '+str(i+1)+':' + str(scores[ i ]))    
    print("accuracy : %0.4f (%0.4f)" % (scores.mean() , scores.std()))

## https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
## Feature importance: https://machinelearningmastery.com/calculate-feature-importance-with-python/
## _logsumexp.py 110 line
## _logistic.py 301 line
def trainTest(train, test, users, mode):
    saveFolder = 'D:/Diplomadolgozat/Classification/'
    name = 'plus'
    scores_filename = 'scores_strong_' + name + str(max(users)) + '.csv'
    if mode == 1:
        scores_filename = 'scores_' + name + str(max(users)) + '.csv'
    Path(saveFolder).mkdir(parents=True, exist_ok=True)

    train = selectUsers(train, users)
    test = selectUsers(test, users)

    X_train = train.iloc[:,:-1]
    X_test = test.iloc[:,:-1]
    y_train = train.iloc[:,-1:]
    y_test = test.iloc[:,-1:]
    
    y_train = np.array(y_train)
    y_train = y_train.reshape(y_train.shape[0])
    y_test = np.array(y_test)
    y_test = y_test.reshape(y_test.shape[0])

    # get the models to evaluate
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    scores_df = pd.DataFrame({})
    for name, model in models.items():
        ## train
        model.fit(X_train, y_train)
        ## score
        scores = model.score(X_test, y_test)

        results.append(scores)
        names.append(name)
        print('>%s %.3f' % (name, scores))

        # try:
        #     # get importance
        #     importance = model.feature_importances_
        #     # summarize feature importance
        #     for i,v in enumerate(importance):
        #         print('Feature: %0d, Score: %.5f' % (i,v))
        # except:
        #     print('Exception: ', name)
        score_df = pd.DataFrame([[str(name), mean(scores)]], columns=['Osztályozó', 'Pontosság'])
        scores_df = scores_df.append(score_df, ignore_index=True)
    scores_df.to_csv(saveFolder + scores_filename, index=False)

# Reference: https://towardsdatascience.com/machine-learning-with-python-classification-complete-tutorial-d2c99dc524ec
def tt2(train, test, users):
    train = selectUsers(train, users)
    test = selectUsers(test, users)

    X_train = train.iloc[:,:-1]
    X_test = test.iloc[:,:-1]
    y_train = train.iloc[:,-1:]
    y_test = test.iloc[:,-1:]
    
    y_train = np.array(y_train)
    y_train = y_train.reshape(y_train.shape[0])
    y_test = np.array(y_test)
    y_test = y_test.reshape(y_test.shape[0])

    ## Gradient boosting
    # model = ensemble.GradientBoostingClassifier(subsample= 0.9, n_estimators= 250, min_samples_split= 2, min_samples_leaf = 1, max_features= 5, max_depth = 3, learning_rate = 0.15)
    model = ensemble.GradientBoostingClassifier(n_estimators = 250, random_state=8)
    # model = ensemble.RandomForestClassifier(n_estimators= 250, random_state=8,min_samples_split= 2, min_samples_leaf = 1, max_features= 5, max_depth = 3)
    # model = ensemble.RandomForestClassifier(n_estimators= 250, random_state=0)

    ## train
    model.fit(X_train, y_train)

    ## score
    score = model.score(X_test, y_test)

    ## test
    # predicted_prob = model.predict_proba(X_test)
    # predicted = model.predict(X_test)

    # ## Accuray e AUC
    # accuracy = metrics.accuracy_score(y_test, predicted)
    # auc = metrics.roc_auc_score(y_test, predicted_prob, multi_class='ovo')
    # print("Accuracy (overall correct predictions):",  round(accuracy,2))
    # print("Auc:", round(auc,2))
        
    # ## Precision e Recall
    # recall = metrics.recall_score(y_test, predicted, average='macro')
    # precision = metrics.precision_score(y_test, predicted, average='macro')
    # print("Recall (all 1s predicted right):", round(recall,2))
    # print("Precision (confidence when predicting a 1):", round(precision,2))
    # print("Detail:")
    # print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))

    print(score)
