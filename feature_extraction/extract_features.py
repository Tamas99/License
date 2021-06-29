from __future__ import annotations, division
from pathlib import Path
import numpy as np
import os
import pandas as pd
import statistics as st

#################################################################
#                                                               #
#                       Feature extraction                      #
#                                                               #
#################################################################

def extractFeatures(dataset=0):
    ## Paths
    # scaler_name = 'MinMaxScaler'
    # scaler_name = 'StandardScaler'
    # path_for_data = 'D:/Diplomadolgozat/NormalizedData/' + scaler_name + '/'
    path_for_data = 'D:/Diplomadolgozat/Users/'
    path_for_actions = 'D:/Diplomadolgozat/Actions/'
    savePath = 'D:/Diplomadolgozat/Features/'
    if (dataset == 1):
        # scaler_name = 'MinMaxScaler'
        # scaler_name = 'StandardScaler'
        # path_for_data = 'D:/Diplomadolgozat/NormalizedData/' + scaler_name + '/'
        path_for_data = 'D:/Diplomadolgozat/UsersGen/'
        path_for_actions = 'D:/Diplomadolgozat/ActionsGen/'
        savePath = 'D:/Diplomadolgozat/FeaturesGen/'
    elif dataset == 2:
        path_for_data = 'D:/Diplomadolgozat/UsersSynth/'
        path_for_actions = 'D:/Diplomadolgozat/ActionsSynth/'
        savePath = 'D:/Diplomadolgozat/FeaturesSynth/'
    elif dataset == 3:
        path_for_data = 'D:/Diplomadolgozat/UsersGenAEOneByOne/'
        path_for_actions = 'D:/Diplomadolgozat/ActionsGenAEOneByOne/'
        savePath = 'D:/Diplomadolgozat/FeaturesAEOneByOne/'
    
    Path(savePath).mkdir(parents=True, exist_ok=True)
    features = {
        '1min' : pd.DataFrame({}),
        '3min' : pd.DataFrame({})
    }
    count_users = 0
    for dirname, _, filenames in os.walk(path_for_data):
        dirnameToSer = pd.Series([dirname])
        try:
            user = dirnameToSer.str.findall('user\d{1,3}').iloc[0][0]
            userid = user[4:]
        except IndexError:
            continue
        
        for filename in filenames:
            # if( filename.find('1min') == -1 ):  # Skipping the 3 min sections
            #     continue                        #
            
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

            max_x = max(data['x'])
            max_y = max(data['y'])
            for i in range(actions.last_valid_index() + 1):
                starti = actions.iloc[i]['Start index']
                stopi = actions.iloc[i]['Stop index']
                x = np.array(data.iloc[starti:stopi + 1]['x'])
                dx = np.diff(x)
                y = np.array(data.iloc[starti:stopi + 1]['y'])
                dy = np.diff(y)
                t = np.array(data.iloc[starti:stopi + 1]['client timestamp'])
                dt = np.diff(t)

                ## Average
                avg_dx = np.average(dx)
                avg_dy = np.average(dy)
                avg_dt = np.average(dt)
                
                ## Standard deviation
                std_dx = st.stdev(dx)
                std_dy = st.stdev(dy)
                std_dt = st.stdev(dt)

                ## Maximum
                max_dx = np.percentile(dx, 100)
                max_dy = np.percentile(dy, 100)
                max_dt = np.percentile(dt, 100)

                ## Median
                median_dx = np.percentile(dx, 50)
                median_dy = np.percentile(dy, 50)
                median_dt = np.percentile(dt, 50)

                ## Percentile 25
                perc25_dx = np.percentile(dx, 25)
                perc25_dy = np.percentile(dy, 25)
                perc25_dt = np.percentile(dt, 25)

                ## Percentile 75
                perc75_dx = np.percentile(dx, 75)
                perc75_dy = np.percentile(dy, 75)
                perc75_dt = np.percentile(dt, 75)

                ## Length
                length = stopi - starti

                ## Speed
                ## Do not divide by 0
                out1 = np.ones_like(dt)
                out2 = np.ones_like(dt)
                out1 = out1.astype(float)
                out2 = out2.astype(float)
                v_x = np.divide(dx, dt, out = out1, where = dt != 0)
                v_y = np.divide(dy, dt, out = out2, where = dt != 0)
                v = np.sqrt( np.square(v_x) + np.square(v_y) )
                
                ## Stats
                avg_v = np.average(v)
                avg_v_x = np.average(v_x)
                avg_v_y = np.average(v_y)

                std_v = st.stdev(v)
                std_v_x = st.stdev(v_x)
                std_v_y = st.stdev(v_y)

                max_v = np.percentile(v, 100)
                max_v_x = np.percentile(v_x, 100)
                max_v_y = np.percentile(v_y, 100)

                median_v = np.percentile(v, 50)
                median_v_x = np.percentile(v_x, 50)
                median_v_y = np.percentile(v_y, 50)

                perc25_v = np.percentile(v, 25)
                perc25_v_x = np.percentile(v_x, 25)
                perc25_v_y = np.percentile(v_y, 25)

                perc75_v = np.percentile(v, 75)
                perc75_v_x = np.percentile(v_x, 75)
                perc75_v_y = np.percentile(v_y, 75)

                ## Acceleration
                ddt = np.diff(dt)
                dv_x = np.diff(v_x)
                dv_y = np.diff(v_y)
                out1 = np.ones_like(dv_x)
                out2 = np.ones_like(dv_y)
                out1 = out1.astype(float)
                out2 = out2.astype(float)
                a_x = np.divide(dv_x, ddt, out=out1, where = ddt != 0)
                a_y = np.divide(dv_y, ddt, out=out2, where = ddt != 0)
                a = np.sqrt( np.square(a_x) + np.square(a_y) )
                
                ## Stats
                avg_a = np.average(a)
                avg_a_x = np.average(a_x)
                avg_a_y = np.average(a_y)

                std_a = st.stdev(a)
                std_a_x = st.stdev(a_x)
                std_a_y = st.stdev(a_y)

                max_a = np.percentile(a, 100)
                max_a_x = np.percentile(a_x, 100)
                max_a_y = np.percentile(a_y, 100)

                median_a = np.percentile(a, 50)
                median_a_x = np.percentile(a_x, 50)
                median_a_y = np.percentile(a_y, 50)

                perc25_a = np.percentile(a, 25)
                perc25_a_x = np.percentile(a_x, 25)
                perc25_a_y = np.percentile(a_y, 25)

                perc75_a = np.percentile(a, 75)
                perc75_a_x = np.percentile(a_x, 75)
                perc75_a_y = np.percentile(a_y, 75)

                ## Estimated time
                et = t[-1] - t[0]

                currentFeatures = pd.DataFrame([[
                                avg_dx, avg_dy, avg_dt,
                                std_dx, std_dy, std_dt,
                                max_dx, max_dy, max_dt,
                                median_dx, median_dy, median_dt,
                                perc25_dx, perc25_dy, perc25_dt,
                                perc75_dx, perc75_dy, perc75_dt,
                                avg_v, avg_v_x, avg_v_y,
                                std_v, std_v_x, std_v_y,
                                max_v, max_v_x, max_v_y,
                                median_v, median_v_x, median_v_y,
                                perc25_v, perc25_v_x, perc25_v_y,
                                perc75_v, perc75_v_x, perc75_v_y,
                                avg_a, avg_a_x, avg_a_y,
                                std_a, std_a_x, std_a_y,
                                max_a, max_a_x, max_a_y,
                                median_a, median_a_x, median_a_y,
                                perc25_a, perc25_a_x, perc25_a_y,
                                perc75_a, perc75_a_x, perc75_a_y,
                                length, et,
                                max_x, max_y,
                                userid
                                ]], 
                            columns=[
                                    'avg_dx', 'avg_dy', 'avg_dt',
                                    'std_dx', 'std_dy', 'std_dt', 
                                    'max_dx', 'max_dy', 'max_dt',
                                    'median_dx', 'median_dy', 'median_dt',
                                    'perc25_dx', 'perc25_dy', 'perc25_dt',
                                    'perc75_dx', 'perc75_dy', 'perc75_dt',
                                    'avg_v', 'avg_v_x', 'avg_v_y',
                                    'std_v', 'std_v_x', 'std_v_y',
                                    'max_v', 'max_v_x', 'max_v_y',
                                    'median_v', 'median_v_x', 'median_v_y',
                                    'perc25_v', 'perc25_v_x', 'perc25_v_y',
                                    'perc75_v', 'perc75_v_x', 'perc75_v_y',
                                    'avg_a', 'avg_a_x', 'avg_a_y',
                                    'std_a', 'std_a_x', 'std_a_y',
                                    'max_a', 'max_a_x', 'max_a_y',
                                    'median_a', 'median_a_x', 'median_a_y',
                                    'perc25_a', 'perc25_a_x', 'perc25_a_y',
                                    'perc75_a', 'perc75_a_x', 'perc75_a_y',
                                    'length', 'et',
                                    'max_x', 'max_y',
                                    'userid'
                                    ])
                
                features[minute] = features[minute].append(currentFeatures, ignore_index = True)

        count_users = count_users + 1
    
    features['1min'].to_csv(savePath + 'sapimouse_1min_strong.csv', 
                            index = False)
    # features['3min'].to_csv(savePath + 'sapimouse_3min_strong.csv', 
    #                         index = False)

    # features['1min'].drop(columns=['max_x', 'max_y']).to_csv(savePath + 'sapimouse_1min.csv', 
    #                         index = False)
    # features['3min'].drop(columns=['max_x', 'max_y']).to_csv(savePath + 'sapimouse_3min.csv', 
    #                         index = False)

def dataToDiffs():
    dataFolder = 'D:/Diplomadolgozat/Users/'
    actionsFolder = 'D:/Diplomadolgozat/Actions/'
    savePath = 'D:/Diplomadolgozat/Diffs/'
    diffs = {
        '1min' : pd.DataFrame({}),
        '3min' : pd.DataFrame({})
    }
    count_users = 0
    Path(savePath).mkdir(parents=True, exist_ok=True)
    for dirname, _, filenames in os.walk(dataFolder):
        dirnameToSer = pd.Series([dirname])
        try:
            user = dirnameToSer.str.findall('user\d{1,3}').iloc[0][0]
            userid = user[4:]
        except IndexError:
            continue
        
        for filename in filenames:
            # if( filename.find('1min') == -1 ):  # Skipping the 3 min sections
            #     continue                        #
            
            ## Show progress percentage
            progress = (count_users * 100)/120
            progress = "{:.2f}".format(progress)
            print('\rProgress: ' + str(progress) + '%', end='')
            
            filenameToSer = pd.Series([filename])
            minute = filenameToSer.str.findall('\dmin').iloc[0][0]

            dataPath = os.path.join(dirname, filename)
            data = pd.read_csv(dataPath)
            actionsPath = actionsFolder + user.capitalize() + '/' + minute + '.csv'
            actions = pd.read_csv(actionsPath)
            for i in range(actions.last_valid_index() + 1):
                starti = actions.iloc[i]['Start index']
                stopi = actions.iloc[i]['Stop index']
                x = np.array(data.iloc[starti:stopi + 1]['x'])
                dx = np.diff(x)
                y = np.array(data.iloc[starti:stopi + 1]['y'])
                dy = np.diff(y)
                t = np.array(data.iloc[starti:stopi + 1]['client timestamp'])
                dt = np.diff(t)

                if len(dx) < 128:
                    zeros = np.zeros(128 - len(dx), dtype=int)
                    dx = np.concatenate((dx, zeros))
                    dy = np.concatenate((dy, zeros))
                    dt = np.concatenate((dt, zeros))
                else:
                    dx = dx[:128]
                    dy = dy[:128]
                    dt = dt[:128]

                diff = np.concatenate((dx, dy, dt))
                id = np.full(1, userid)
                diff = np.concatenate((diff, id))
                diffs[minute] = diffs[minute].append(pd.DataFrame(diff.reshape(1,-1)), ignore_index=True)
            
            # print(diffs[minute])
            # return
        count_users += 1
    
    diffs['1min'].to_csv(savePath + 'diffs_1min.csv', header=False, index=False)
    diffs['3min'].to_csv(savePath + 'diffs_3min.csv', header=False, index=False)
