from __future__ import annotations, division
from pathlib import Path
import numpy as np
import os
import pandas as pd
import sys
sys.path.insert(0,'D:\License\pyclick_master\pyclick')
import humanclicker as hcfile
from utils.settings import generate_mode
from utils.settings import similarTimeDifferences

def generateMouseMovementsHC():
    ## Paths
    path_for_data = 'D:/Diplomadolgozat/Users/'
    path_for_actions = 'D:/Diplomadolgozat/Actions/'
    dataSavePath = 'D:/Diplomadolgozat/UsersGen/'
    actionsSavePath = 'D:/Diplomadolgozat/ActionsGen/'

    Path(dataSavePath).mkdir(parents=True, exist_ok=True)
    Path(actionsSavePath).mkdir(parents=True, exist_ok=True)

    ## Initializing
    hc = hcfile.HumanClicker() # HumanClicker object
    count_users = 0
    folders = 0

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
        
        user_data = {
            '1min' : pd.DataFrame({}),
            '3min' : pd.DataFrame({})
        }
        user_actions = {
            '1min' : pd.DataFrame({}),
            '3min' : pd.DataFrame({})
        }

        for filename in filenames:
            if( filename.find('1min') == -1 ):  # Skipping the 3 min sections
                continue                        #
            
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

            time_recording = 1
            starti = 0
            stopi = 0
            for i in range(len(actions)):
                user_starti = actions.iloc[i]['Start index']
                user_stopi = actions.iloc[i]['Stop index']
                length = int(user_stopi - user_starti) + 1
                t = np.array(data.iloc[user_starti:user_stopi+1]['client timestamp'])
                
                _dt_synth = similarTimeDifferences(t, generate_mode.noncentral_f)
                
                ## Build up timestamp from synthesized dt array
                start_t = time_recording
                ind = 0
                _time_stamp = pd.Series([])
                _time_stamp[ind] = start_t
                ind += 1
                for i in _dt_synth:
                    start_t = start_t + np.abs(i)
                    _time_stamp[ind] = start_t
                    ind = ind + 1

                # time_stamp = _time_stamp.sort_values(ignore_index=True)
                time_stamp = _time_stamp.round()
                time_stamp = time_stamp.astype(int)
                time_recording = time_stamp.iloc[-1]

                ## Start- and endpoints
                from_x = data.iloc[user_starti]['x']
                from_y = data.iloc[user_starti]['y']
                to_x = data.iloc[user_stopi]['x']
                to_y = data.iloc[user_stopi]['y']
                from_point = (from_x, from_y)
                to_point = (to_x, to_y)

                ## Generate points from_point to_point
                coordinates = hc.getPoints(from_point, to_point, length)
                
                ## Split up coordinate tuples
                x = pd.Series([])
                y = pd.Series([])
                ind = 0
                for i in coordinates:
                    x[ind] = i[0]
                    y[ind] = i[1]
                    ind += 1

                ## Appending to raw data and to actions
                df = pd.DataFrame({})
                df['client timestamp'] = time_stamp
                df['x'] = x
                df['y'] = y
                user_data[minute] = user_data[minute].append(df, ignore_index=True)
                stopi = user_data[minute].last_valid_index()
                user_actions[minute] = user_actions[minute].append(pd.DataFrame([[starti, stopi]], columns=['Start index', 'Stop index']), ignore_index=True)
                starti = stopi + 1
                
            ## Save generated raw data and actions
            user_data[minute].to_csv(dataSavePath + user + '/' +
                filename, index = False)
            user_actions[minute].to_csv(actionsSavePath + user.capitalize() + '/' +
                minute + '.csv', index = False)

        # return
        count_users = count_users + 1