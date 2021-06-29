from __future__ import annotations, division
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from utils.settings import selectUsers

def subplotActions(dataset=0):
    if dataset == 0:
        dataFolder = 'D:/Diplomadolgozat/Users/'
        actionsFolder = 'D:/Diplomadolgozat/Actions/'
        savePath = 'D:/Diplomadolgozat/SubplotActions/'
    elif dataset == 1:
        dataFolder = 'D:/Diplomadolgozat/UsersGen/'
        actionsFolder = 'D:/Diplomadolgozat/ActionsGen/'
        savePath = 'D:/Diplomadolgozat/SubplotActionsGen/'
    elif dataset == 2:
        dataFolder = 'D:/Diplomadolgozat/UsersGenAEOneByOne/'
        actionsFolder = 'D:/Diplomadolgozat/ActionsGenAEOneByOne/'
        savePath = 'D:/Diplomadolgozat/SubplotActionsGenAEOneByOne/'
    Path(savePath).mkdir(parents=True, exist_ok=True)
    for dirname, _, filenames in os.walk(dataFolder):
        dirnameToSer = pd.Series([dirname])
        try:
            user = dirnameToSer.str.findall('user\d{1,3}').iloc[0][0]
        except IndexError:
            continue
        
        Path(savePath + user.capitalize()).mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            print(dirname + ' ' + filename)
            filenameToSer = pd.Series([filename])
            minute = filenameToSer.str.findall('\dmin').iloc[0][0]
            Path(savePath + user.capitalize() + '/' + minute).mkdir(parents=True, exist_ok=True)
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

                fig, axs = plt.subplots(2, 2)
                try:
                    fig.suptitle(str(i) + ', ' + actions.iloc[i]['Action type'])
                except:
                    fig.suptitle(str(i))
                axs[0, 0].set_title('dx')
                axs[0, 0].set(xlabel='time', ylabel='value')
                axs[0, 0].plot(dx, '-o')
                axs[0, 1].set_title('dy')
                axs[0, 1].set(xlabel='time', ylabel='value')
                axs[0, 1].plot(dy, '-o')
                axs[1, 0].set_title('dt')
                axs[1, 0].set(xlabel='time', ylabel='value')
                axs[1, 0].plot(dt, '-o')
                axs[1, 1].set_title('Trajectory x, y')
                axs[1, 1].set(xlabel='X', ylabel='Y')
                max_y = np.max(y)
                axs[1, 1].plot(x,max_y - y, '-o')
                axs[1, 1].plot(x[0:1],max_y - y[0:1], 'ro', label='Starting')
                leg = axs[1, 1].legend()
                fig.tight_layout()
                try:
                    fig.savefig(savePath + user.capitalize() + '/' + 
                            minute + '/' + str(i) + ', ' + actions.iloc[i]['Action type'] + '.png')
                except:
                    fig.savefig(savePath + user.capitalize() + '/' + 
                            minute + '/' + str(i) + '.png')
                plt.close(fig)

def subplotActionsFixedSize():
    Path('D:/Diplomadolgozat/SubplotActionsFixedSize/').mkdir(parents=True, exist_ok=True)
    dataPath_1min = 'D:/Diplomadolgozat/UsersMerged/user_data_1min.csv'
    dataPath_3min = 'D:/Diplomadolgozat/UsersMerged/user_data_3min.csv'
    actionsPath_1min = 'D:/Diplomadolgozat/ActionsMerged/user_actions_1min.csv'
    actionsPath_3min = 'D:/Diplomadolgozat/ActionsMerged/user_actions_3min.csv'
    data_1min = pd.read_csv(dataPath_1min)
    data_3min = pd.read_csv(dataPath_3min)
    actions_1min = pd.read_csv(actionsPath_1min)
    actions_3min = pd.read_csv(actionsPath_3min)

    # 1 min
    # Data
    arrayT_1min = data_1min['client timestamp']
    arrayX_1min = data_1min['x']
    arrayY_1min = data_1min['y']

    # 3 min
    # Data
    arrayT_3min = data_3min['client timestamp']
    arrayX_3min = data_3min['x']
    arrayY_3min = data_3min['y']

    # 1 min
    # Diffs
    dt_1min = np.diff(arrayT_1min)
    dx_1min = np.diff(arrayX_1min)
    dy_1min = np.diff(arrayY_1min)
    
    # 3 min
    # Diffs
    dt_3min = np.diff(arrayT_3min)
    dx_3min = np.diff(arrayX_3min)
    dy_3min = np.diff(arrayY_3min)

    # 1 min
    # Maxes
    maxX_1min = np.max(arrayX_1min)
    maxY_1min = np.max(arrayY_1min)
    maxDt_1min = np.max(dt_1min)
    maxDx_1min = np.max(dx_1min)
    maxDy_1min = np.max(dy_1min)
    maxes_1min = {
        'x' : maxX_1min,
        'y' : maxY_1min,
        'dt' : maxDt_1min,
        'dx' : maxDx_1min,
        'dy' : maxDy_1min
    }

    # 1 min
    # Mins
    minDt_1min = np.min(dt_1min)
    minDx_1min = np.min(dx_1min)
    minDy_1min = np.min(dy_1min)
    mins_1min = {
        'dt' : minDt_1min,
        'dx' : minDx_1min,
        'dy' : minDy_1min
    }

    # 3 min
    # Maxes
    maxX_3min = np.max(arrayX_3min)
    maxY_3min = np.max(arrayY_3min)
    maxDt_3min = np.max(dt_3min)
    maxDx_3min = np.max(dx_3min)
    maxDy_3min = np.max(dy_3min)
    maxes_3min = {
        'x' : maxX_3min,
        'y' : maxY_3min,
        'dt' : maxDt_3min,
        'dx' : maxDx_3min,
        'dy' : maxDy_3min
    }
    
    # 3 min
    # Mins
    minDt_3min = np.min(dt_3min)
    minDx_3min = np.min(dx_3min)
    minDy_3min = np.min(dy_3min)
    mins_3min = {
        'dt' : minDt_3min,
        'dx' : minDx_3min,
        'dy' : minDy_3min
    }

    print('Maxes')
    print(maxes_1min)
    print(maxes_3min)
    print('Mins')
    print(mins_1min)
    print(mins_3min)

    for i in range(1,121):
        currentData_1min = selectUsers(data_1min, [i])
        currentData_3min = selectUsers(data_3min, [i])
        currentActions_1min = selectUsers(actions_1min, [i])
        currentActions_3min = selectUsers(actions_3min, [i])
        Path('D:/Diplomadolgozat/SubplotActionsFixedSize/User' + str(i)).mkdir(parents=True, exist_ok=True)
        Path('D:/Diplomadolgozat/SubplotActionsFixedSize/User' + str(i) + '/1min').mkdir(parents=True, exist_ok=True)
        Path('D:/Diplomadolgozat/SubplotActionsFixedSize/User' + str(i) + '/3min').mkdir(parents=True, exist_ok=True)

        subplotCurrentActions(currentData_1min, currentActions_1min, i, maxes_1min, mins_1min, '1min')
        subplotCurrentActions(currentData_3min, currentActions_3min, i, maxes_3min, mins_3min, '3min')

def subplotCurrentActions(currentData, currentActions, userid, maxes, mins, minute):
    for i in range(currentActions.last_valid_index() + 1):
        starti = currentActions.iloc[i]['Start index']
        stopi = currentActions.iloc[i]['Stop index']
        x = np.array(currentData.iloc[starti:stopi + 1]['x'])
        dx = np.diff(x)
        y = np.array(currentData.iloc[starti:stopi + 1]['y'])
        dy = np.diff(y)
        t = np.array(currentData.iloc[starti:stopi + 1]['client timestamp'])
        dt = np.diff(t)

        fig, axs = plt.subplots(2, 2)
        fig.suptitle(str(i) + ', ' + currentActions.iloc[i]['Action type'])
        axs[0, 0].set_title('dx')
        axs[0, 0].set(xlabel='time', ylabel='value')
        axs[0, 0].set_ylim([mins['dx'], maxes['dx']])
        axs[0, 0].plot(dx, '.')
        axs[0, 1].set_title('dy')
        axs[0, 1].set(xlabel='time', ylabel='value')
        axs[0, 1].set_ylim([mins['dy'], maxes['dy']])
        axs[0, 1].plot(dy, '.')
        axs[1, 0].set_title('dt')
        axs[1, 0].set(xlabel='time', ylabel='value')
        axs[1, 0].set_ylim([mins['dt'], maxes['dt']])
        axs[1, 0].plot(dt, '.')
        axs[1, 1].set_title('Trajectory x, y')
        axs[1, 1].set_xlim([0, maxes['x']])
        axs[1, 1].set_ylim([0, maxes['y']])
        axs[1, 1].set(xlabel='X', ylabel='Y')
        max_y = np.max(y)
        axs[1, 1].plot(x,max_y - y, '.')
        axs[1, 1].plot(x[0:1],max_y - y[0:1], 'r.', label='Starting')
        leg = axs[1, 1].legend()
        fig.tight_layout()
        fig.savefig('D:/Diplomadolgozat/SubplotActionsFixedSize/User' + str(userid) + '/' + 
                    minute + '/' + str(i) + ', ' + currentActions.iloc[i]['Action type'] + '.png')
        plt.close(fig)
