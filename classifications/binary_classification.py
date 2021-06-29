from __future__ import annotations, division
from numpy import mean
from numpy import std

import numpy as np
import pandas as pd
from utils.settings import get_models, evaluate_model

def binaryClassification(mode=1, length=5052):
    path_for_features_1min = 'D:/Diplomadolgozat/Features/sapimouse_1min.csv'
    if mode == 1:
        path_for_features_gen_1min = 'D:/Diplomadolgozat/FeaturesGen/sapimouse_1min.csv'
        df_save_path = 'D:/Diplomadolgozat/BinaryClassification/binary_classification.csv'
        scores_save_path = 'D:/Diplomadolgozat/BinaryClassification/noncentral_f_timediffs.csv'
    elif mode == 2:
        path_for_features_gen_1min = 'D:/Diplomadolgozat/FeaturesSynth/sapimouse_1min.csv'
        df_save_path = 'D:/Diplomadolgozat/BinaryClassification/binary_classification2.csv'
        scores_save_path = 'D:/Diplomadolgozat/BinaryClassification/synthesized_timediffs2.csv'
    elif mode == 3:
        path_for_features_gen_1min = 'D:/Diplomadolgozat/FeaturesAEOneByOne/sapimouse_1min.csv'
        df_save_path = 'D:/Diplomadolgozat/BinaryClassification/binary_classification3OneByOne.csv'
        scores_save_path = 'D:/Diplomadolgozat/BinaryClassification/autoencoder_timediffsOneByOne.csv'

    features = pd.read_csv(path_for_features_1min)
    features_gen = pd.read_csv(path_for_features_gen_1min)

    features_gen = features_gen.drop(columns=['userid'])
    features = features.drop(columns=['userid'])
    rows, cols = features.shape
    features['userid'] = np.zeros(rows)
    features_gen['userid'] = np.ones(rows)

    if (length > rows):
        length = rows
    elif (length < 1):
        length = 900
    print('length = ' + str(length))
    
    human_like = features.sample(n=length, random_state=0)
    generated = features_gen.sample(n=length, random_state=0)

    df = pd.DataFrame({})
    df = df.append(pd.DataFrame(human_like), ignore_index=True)
    df = df.append(pd.DataFrame(generated), ignore_index=True)
    # print(df.shape)
    # print(df.describe())
    df.to_csv(df_save_path, index=False)
    # return

    X = df.drop(columns=['userid'])
    y = df['userid']
    # y = np.array(y)
    # y = y.reshape(y.shape[0])

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
        mean_std = '' + str(round(mean(scores), 3)) + ' (' + str(round(std(scores), 3)) + ')'
        score_df = pd.DataFrame([[mean_std]], columns=[str(name)])
        scores_df = pd.concat([scores_df, score_df], axis=1)
        # print(scores_df)
    
    scores_df.to_csv(scores_save_path, index=False)
