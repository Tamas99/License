from __future__ import annotations, division
from numpy import mean
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statistics as st
import sys
sys.path.insert(0,'D:\License\pyclick_master\pyclick')
import humanclicker as hcfile
from generate_datasets.synthesized_dataset import similarCoordinateDifferences

#################################################################
#                                                               #
#                       Testing functions                       #
#                                                               #
#################################################################

def extractRows(user_index, min, action_index):
    '''
        Returns timestamp, x and y coordinates for
        a given action.
        Params: (user_index, min, action_index);
        user_index: (1 - 120) from which user we want the action;
        min: (1 or 3) 1min or 3min dataset;
        action_index: starts from 0 and depends how many actions a user has.
    '''
    path_for_data = 'D:/Diplomadolgozat/Users/user' + str(user_index) + '/'
    path_for_actions = 'D:/Diplomadolgozat/Actions/User' + str(user_index) + '/' + str(min) + 'min.csv'


    for _, _, filenames in os.walk(path_for_data):
        for filename in filenames:
            try:
                filenameToSer = pd.Series([filename])
                minute = filenameToSer.str.findall('\dmin').iloc[0][0]
                if minute == (str(min) + 'min'):
                    path_for_data = path_for_data + filename
                    break
            except:
                continue
    
    data = pd.read_csv(path_for_data)
    actions = pd.read_csv(path_for_actions)

    starti = actions.iloc[action_index]['Start index']
    stopi = actions.iloc[action_index]['Stop index']

    t = data.iloc[starti:stopi+1]['client timestamp']
    x = data.iloc[starti:stopi+1]['x']
    y = data.iloc[starti:stopi+1]['y']

    return t, x, y

def dummies():
    ## Paths
    path_for_data = 'D:/Diplomadolgozat/Users/'
    path_for_actions = 'D:/Diplomadolgozat/Actions/'
    path_for_data_gen = 'D:/Diplomadolgozat/UsersGen/'
    dataSavePath = 'D:/Diplomadolgozat/UsersGen/'
    actionsSavePath = 'D:/Diplomadolgozat/ActionsGen/'
    path_for_features_1min = 'D:/Diplomadolgozat/Features/v1/sapimouse_1min_weak.csv'
    path_for_features_3min = 'D:/Diplomadolgozat/Features/v1/sapimouse_3min_weak.csv'

    Path(dataSavePath).mkdir(parents=True, exist_ok=True)
    Path(actionsSavePath).mkdir(parents=True, exist_ok=True)

    ## Initializing
    features_1min = pd.read_csv(path_for_features_1min)
    features_3min = pd.read_csv(path_for_features_3min)
    hc = hcfile.HumanClicker() # HumanClicker object
    count_users = 0
    folders = 0

    action_lengths_1min = features_1min['length']
    action_lengths_3min = features_3min['length']
    avg_dt_1min = features_1min['avg_dt']
    avg_dt_3min = features_3min['avg_dt']
    std_dt_1min = features_1min['std_dt']
    std_dt_3min = features_3min['std_dt']

    size1 = len(action_lengths_1min)
    size3 = len(action_lengths_3min)

    mean_lengths_1min = np.mean(action_lengths_1min)
    mean_lengths_3min = np.mean(action_lengths_3min)

    std_lengths_1min = st.stdev(action_lengths_1min)
    std_lengths_3min = st.stdev(action_lengths_3min)

    a_l1 = np.random.normal(mean_lengths_1min, std_lengths_1min, size1)
    a_l3 = np.random.normal(mean_lengths_3min, std_lengths_3min, size3)

    dt1 = np.random.normal(avg_dt_1min, std_dt_1min, size1)
    dt3 = np.random.normal(avg_dt_3min, std_dt_3min, size3)

    start = 1
    ind = 0
    time_stamp = pd.Series([])

    # print(sum(action_lengths_1min))
    # print(sum(a_l1))
    # print(a_l3)

    for i in dt1:
        start = start + i
        time_stamp[ind] = start
        ind = ind + 1
        # print(time_stamp[time_stamp.last_valid_index()])

    sorted_t = time_stamp.sort_values(ignore_index=True)

    diff_t = sorted_t.diff()
    first_action_dt = diff_t[1:action_lengths_1min[0]]

    
    ## a selected action
    t, _, _ = extractRows(1, 1, 1)

    dt = np.diff(t)

    x = np.random.noncentral_f(0.05, 100, 0.01, len(dt)) + 17
    x2 = np.copy(x)

    for i in range(len(x)):
        x2[i] = np.random.noncentral_f(0.05, 150, 0.01, 1) + 17

    print(mean(dt))
    print(mean(first_action_dt))
    print(mean(x))
    print(mean(x2))

    print(st.stdev(dt))
    print(st.stdev(x))
    print(st.stdev(x2))

    plt.figure(1)
    plt.plot(first_action_dt)
    # plt.show()

    # scaler = preprocessing.MinMaxScaler()
    # first_action_dt = first_action_dt.reshape(-1,1)
    # scaled = scaler.fit_transform(first_action_dt)
    plt.figure(2)
    plt.plot(x)

    plt.figure(3)
    plt.plot(x2)

    plt.figure(4)
    plt.plot(dt)

    plt.show()

    return

    ## Count the folders so we will know where we've
    ## been last time to continue
    ## Comment these two lines if the files need to
    ## be regenerated
    for _, dirnames, _ in os.walk(path_for_data_gen):
        folders += len(dirnames)

    for dirname, _, filenames in os.walk(path_for_data):
        if count_users <= folders - 1:
            count_users += 1
            continue

        dirnameToSer = pd.Series([dirname])
        try:
            user = dirnameToSer.str.findall('user\d{1,3}').iloc[0][0]
        except IndexError:
            continue

        Path(dataSavePath + user + '/').mkdir(parents=True, exist_ok=True)
        Path(actionsSavePath + user.capitalize() + '/').mkdir(parents=True, exist_ok=True)
        
        for filename in filenames:
            ## Show progress percentage
            progress = (count_users * 100)/120
            progress = "{:.2f}".format(progress)
            print('\rProgress: ' + str(progress) + '%', end='')
            
            ## Dataframes
            data_gen = pd.DataFrame({})
            actions_gen = pd.DataFrame({})

            filenameToSer = pd.Series([filename])
            minute = filenameToSer.str.findall('\dmin').iloc[0][0]
            dataPath = os.path.join(dirname, filename)
            data = pd.read_csv(dataPath)

            actionsPath = path_for_actions + user.capitalize() + '/' + minute + '.csv'
            actions = pd.read_csv(actionsPath)

            data_gen.to_csv(dataSavePath + user + '/' +
                filename, index = False)

            actions_gen.to_csv(actionsSavePath + user.capitalize() + '/' +
                minute + '.csv', index = False)

        count_users = count_users + 1

## Reference: https://pypi.org/project/pysynth/
def similarActionTest():
    t, x, y = extractRows(1, 1, 3)
    x = np.array(x)
    dx = np.diff(x)
    _dx = similarCoordinateDifferences(x)
    _x = np.array(np.zeros_like(x))
    _x[0] = x[0]
    for i in range(len(_dx)):
        _x[i+1] = _x[i] + _dx[i]

    plt.figure('x')
    plt.plot(x)
    plt.figure('_x')
    plt.plot(_x)
    plt.figure('dx')
    plt.plot(dx)
    plt.figure('_dx')
    plt.plot(_dx)
    plt.show()