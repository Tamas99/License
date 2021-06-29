from __future__ import annotations, division
from pathlib import Path
import numpy as np
import os
import pandas as pd
from utils.settings import MIN_LENGTH

def breakDownToMouseEvents():
    Path('D:/Diplomadolgozat/Actions').mkdir(parents=True, exist_ok=True)
    indexing = 0
    for dirname, _, filenames in os.walk('D:/Diplomadolgozat/Users'):
        dirnameToSer = pd.Series([dirname])
        try:
            user = dirnameToSer.str.findall('user\d{1,3}').iloc[0][0]
        except IndexError:
            continue
        for filename in filenames:
            Path('D:/Diplomadolgozat/Actions/' + user.capitalize()).mkdir(parents=True, exist_ok=True)
            path = os.path.join(dirname, filename)
            data = pd.read_csv(path)
            actions = pd.DataFrame({})
            filenameToSer = pd.Series([filename])
            minute = filenameToSer.str.findall('\dmin').iloc[0][0]

            for i in range(data.last_valid_index() + 1):
                if (data.iloc[i]['state'] == 'Released'):
                    # Move on to the next index if action is short
                    if (i - indexing < MIN_LENGTH):
                        indexing = i + 1
                        continue
                    if (data.iloc[i-1]['state'] == 'Drag'):
                        actions = actions.append(pd.DataFrame([['DD', indexing, i]], columns = ['Action type', 'Start index', 'Stop index']), ignore_index = True)
                        indexing = i + 1
                        
                    else:
                        actions = actions.append(pd.DataFrame([['PC', indexing, i]], columns = ['Action type', 'Start index', 'Stop index']), ignore_index = True)
                        indexing = i + 1
                # Filter the MM events
                elif (data.iloc[i]['state'] == 'Pressed' and i < data.last_valid_index()):
                    try:
                        if (data.iloc[i+1]['state'] == 'Drag'):
                            indexing = i
                    except IndexError:
                        print(dirname + filename)
                        return

            actions.to_csv('D:/Diplomadolgozat/Actions/' + user.capitalize() + '/' + minute + '.csv', header = True, index=False)

def mergeData():
    ## Paths
    path_for_data = 'D:/Diplomadolgozat/Users/'
    path_for_actions = 'D:/Diplomadolgozat/Actions/'

    dataSavePath = 'D:/Diplomadolgozat/UsersMerged/'
    actionsSavePath = 'D:/Diplomadolgozat/ActionsMerged/'

    Path(dataSavePath).mkdir(parents=True, exist_ok=True)
    Path(actionsSavePath).mkdir(parents=True, exist_ok=True)

    ## Initializing
    count_users = 0
    folders = 0
    user_data = {
        '1min' : pd.DataFrame({}),
        '3min' : pd.DataFrame({})
    }
    user_actions = {
        '1min' : pd.DataFrame({}),
        '3min' : pd.DataFrame({})
    }


    for dirname, _, filenames in os.walk(path_for_data):
        if count_users <= folders - 1:
            count_users += 1
            continue

        dirnameToSer = pd.Series([dirname])
        try:
            user = dirnameToSer.str.findall('user\d{1,3}').iloc[0][0]
            userid = user[4:]
        except IndexError:
            continue

        for filename in filenames:
            ## Show progress percentage
            progress = (count_users * 100)/120
            progress = "{:.2f}".format(progress)
            print('\rProgress: ' + str(progress) + '%', end='')

            filenameToSer = pd.Series([filename])
            minute = filenameToSer.str.findall('\dmin').iloc[0][0]
            dataPath = os.path.join(dirname, filename)
            data = pd.read_csv(dataPath)

            actionsPath = path_for_actions + user.capitalize() + '/' + minute + '.csv'
            actions = pd.read_csv(actionsPath)

            rows, _ = data.shape
            user_col = np.full((rows, 1), userid)
            data['user'] = user_col

            rows, _ = actions.shape
            user_col = np.full((rows, 1), userid)
            actions['user'] = user_col

            user_data[minute] = user_data[minute].append(data, ignore_index=True)
            user_actions[minute] = user_actions[minute].append(actions, ignore_index=True)

    user_data['1min'].to_csv(dataSavePath + 'user_data_1min.csv', index = False)
    user_data['3min'].to_csv(dataSavePath + 'user_data_3min.csv', index = False)
    user_actions['1min'].to_csv(actionsSavePath + 'user_actions_1min.csv', index = False)
    user_actions['3min'].to_csv(actionsSavePath + 'user_actions_3min.csv', index = False)