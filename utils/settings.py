import numpy as np
import pandas as pd
import pysynth as ps
from __future__ import annotations, division
from enum import Enum
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

generate_mode = Enum('Generate_mode', 'synth avg noncentral_f')
MIN_LENGTH = 10

# users - range of users e.g. [1, 11)
def selectUsers( df, users ):
    df =  df.loc[ df.iloc[:, -1].isin(users) ]
    return df

def similarTimeDifferences(t, mode):
    dt = np.diff(t)
    avg_dt = np.average(dt)
    data = pd.DataFrame({})
    data['dt'] = dt
    data['zeros'] = np.zeros(len(dt))
    if mode == generate_mode.synth:
        synthesized = ps.synthesize(data)
        _dt = synthesized['dt']
    elif mode == generate_mode.avg:
        _dt = np.random.randint( avg_dt/2, (3*avg_dt)/2, len(dt) )
    elif mode == generate_mode.noncentral_f:
        _dt = np.random.noncentral_f(0.05, 100, 0.01, len(dt)) + 17

    return _dt

def similarCoordinateDifferences(target):
    dtarget = np.diff(target)
    data = pd.DataFrame({})
    data['dtarget'] = dtarget
    data['zeros'] = np.zeros(len(dtarget))
    synthesized = ps.synthesize(data)
    _dtarget = synthesized['dtarget']

    return _dtarget

# get a list of models to evaluate
def get_models():
	models = dict()
	# models['LogisticR'] = LogisticRegression(max_iter=1000)
	models['RandomForestC'] = RandomForestClassifier(n_estimators=100)
	# models['rfr'] = RandomForestRegressor(n_estimators=100)
	models['GradientBoostingC'] = GradientBoostingClassifier(n_estimators=100)
	models['KNeighboursC'] = KNeighborsClassifier(n_neighbors=5)
	# models['knr'] = KNeighborsRegressor(n_neighbors=5)
	models['DecisionTreeC'] = DecisionTreeClassifier()
	models['SupportVectorC'] = SVC()
	models['GaussianNB'] = GaussianNB()
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores
