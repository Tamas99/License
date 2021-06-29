from __future__ import annotations, division
from enum import Enum
from keras import layers
from keras.callbacks import TensorBoard
from keras.layers import Conv1D, Conv1DTranspose, Conv2D, Conv2DTranspose
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from numpy import mean
from numpy import std
from numpy.lib.function_base import diff
from numpy.lib.npyio import save
from pathlib import Path
from sklearn import ensemble, model_selection
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# from tensorflow.python.ops.gen_array_ops import shape

import keras
import matplotlib
# matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.spines as spn
import numpy as np
import os
import pandas as pd
import pysynth as ps
import random as rn
import statistics as st
# import tensorflow as tf
import sys
sys.path.insert(0,'D:\License\pyclick_master\pyclick')
import humanclicker as hcfile

#################################################################
#                                                               #
#                       Data Manipulation                       #
#                                                               #
#################################################################

def breakDownToUnits():
    Path('D:/Diplomadolgozat/Actions').mkdir(parents=True, exist_ok=True)
    MIN_LENGTH = 10
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

#################################################################
#                                                               #
#                       Plots                                   #
#                                                               #
#################################################################

def boxplots():
    dataFolder = 'D:/Diplomadolgozat/UsersMerged/user_data_3min.csv'
    saveFolder = 'D:/Diplomadolgozat/Boxplots/'
    data = pd.read_csv(dataFolder)
    maxY = np.max(data['y'])
    # minY = np.min(data['y'])
    # print(minY, maxY)
    for i in range(1,121):
        currentData = selectUsers(data, [i])
        fig = plt.figure()
        boxplot = currentData.boxplot(column=['x', 'y'])
        # plt.xlim([minX, maxX])
        plt.ylim([0, maxY + (50 * maxY)/100])
        plt.title('user' + str(i))
        fig.savefig(saveFolder + 'user' + str(i) + '.png')
        plt.close(fig)
        print('user' + str(i))

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

def plotTrajectoriesUser10():
    ## Paths
    dataFolder = 'D:/Diplomadolgozat/UsersMerged/user_data_1min.csv'
    actionsFolder = 'D:/Diplomadolgozat/ActionsMerged/user_actions_1min.csv'

    data = pd.read_csv(dataFolder)
    actions = pd.read_csv(actionsFolder)

    user_data = selectUsers(data, [10])
    user_actions = selectUsers(actions, [10])

    savePath = 'D:/Diplomadolgozat/Trajectories/'
    Path(savePath).mkdir(parents=True, exist_ok=True)

    plt.figure()
    for i in range(0,25):
        user_starti = user_actions.iloc[i]['Start index']
        user_stopi = user_actions.iloc[i]['Stop index']

        color = 'r'
        if user_actions.iloc[i]['Action type'] == 'DD':
            color = 'b'
        
        plt.plot(user_data[user_starti:user_stopi+1]['x'], user_data[user_starti:user_stopi+1]['y'], color)
        plt.plot(user_data.iloc[user_starti]['x'], user_data.iloc[user_starti]['y'], color='green', marker='o', linewidth=2, label='Start')
        plt.plot(user_data.iloc[user_stopi]['x'], user_data.iloc[user_stopi]['y'], color='black', marker='x', linewidth=2, label='Stop')
    
    legend_elements = [
        Line2D([0], [0], color='r', label='PC'),
                          Line2D([0], [0], color='b', label='DD'),
                          Line2D([0], [0], marker='o', color='g', label='Start'),
                          Line2D([0], [0], marker='x', color='k', label='Stop')
                   ]

    plt.legend(handles=legend_elements, loc='best')
    plt.show()

def plotTrajectories(len, actionsPerFig):
    '''
    From first user to 'len'-th user plots actions, each figure 
    containing a specified amount ('actionsPerFigure' - int) of actions
    '''
    ## Paths
    dataFolder = 'D:/Diplomadolgozat/UsersMerged/user_data_1min.csv'
    actionsFolder = 'D:/Diplomadolgozat/ActionsMerged/user_actions_1min.csv'

    data = pd.read_csv(dataFolder)
    actions = pd.read_csv(actionsFolder)

    saveFolder = 'D:/Diplomadolgozat/Trajectories/' + str(actionsPerFig) +'/'
    Path(saveFolder).mkdir(parents=True, exist_ok=True)

    from_action = 0
    to_action = actionsPerFig
    # From first user to len-th
    for j in range(1,len+1):
        print(j)
        savePath = saveFolder + 'user' + str(j) + '/'
        Path(savePath).mkdir(parents=True, exist_ok=True)

        # Get the right user from dataset
        user_data = selectUsers(data, [j])
        user_actions = selectUsers(actions, [j])
        
        # Do not plot anymore if to_action is greater than the length of actions
        while (to_action < (user_actions.last_valid_index() - user_actions.first_valid_index())):
            fig = plt.figure(1)
            fig2 = plt.figure(2)
            count_PCs = 0
            count_DDs = 0
            # Plotting figures that contain 'actionsPerFigure' actions
            for i in range(from_action,to_action):
                user_starti = user_actions.iloc[i]['Start index']
                user_stopi = user_actions.iloc[i]['Stop index']

                color = 'r'
                if user_actions.iloc[i]['Action type'] == 'DD':
                    color = 'b'
                    count_DDs += 1
                else:
                    count_PCs += 1
                
                plt.figure(1)
                plt.plot(user_data[user_starti:user_stopi+1]['x'], user_data[user_starti:user_stopi+1]['y'], color)
                plt.plot(user_data.iloc[user_starti]['x'], user_data.iloc[user_starti]['y'], color='green', marker='o', linewidth=2)
                plt.plot(user_data.iloc[user_stopi]['x'], user_data.iloc[user_stopi]['y'], color='black', marker='x', linewidth=2)

                plt.figure(2)
                plt.plot(user_data[user_starti:user_stopi+1]['x'], user_data[user_starti:user_stopi+1]['y'])
                plt.plot(user_data.iloc[user_starti]['x'], user_data.iloc[user_starti]['y'], color='green', marker='o', linewidth=2)
                plt.plot(user_data.iloc[user_stopi]['x'], user_data.iloc[user_stopi]['y'], color='black', marker='x', linewidth=2)

            plt.figure(1)
            legend_elements = [
                Line2D([0], [0], color='r', label= str(count_PCs) + '-PC'),
                                Line2D([0], [0], color='b', label= str(count_DDs) + '-DD'),
                                Line2D([0], [0], marker='o', color='g', label='Start'),
                                Line2D([0], [0], marker='x', color='k', label='Stop')
                        ]

            plt.legend(handles=legend_elements, loc='best')
            fig.savefig(savePath + str(from_action) + '-' + str(to_action) + '.png')
            plt.close(fig)

            plt.figure(2)
            legend_elements = [
                                Line2D([0], [0], marker='o', color='g', label='Start'),
                                Line2D([0], [0], marker='x', color='k', label='Stop')
                        ]

            plt.legend(handles=legend_elements, loc='best')
            fig2.savefig(savePath + str(from_action) + '-' + str(to_action) + 'mix.png')
            plt.close(fig2)

            from_action = to_action
            to_action += actionsPerFig
        
        # Start over again if a user doesn't have anymore action
        from_action = 0
        to_action = actionsPerFig

def testHistograms():
    x = [30,30,30,35,35,40,45,45,50,55]

    plt.hist(x)
    plt.show()

def countActionsPerUser():
    ## Paths
    actionsFolder1 = 'D:/Diplomadolgozat/ActionsMerged/user_actions_1min.csv'
    actionsFolder3 = 'D:/Diplomadolgozat/ActionsMerged/user_actions_3min.csv'

    actions1 = pd.read_csv(actionsFolder1)
    actions3 = pd.read_csv(actionsFolder3)

    length1 = pd.Series([])
    length3 = pd.Series([])
    for i in range(1,121):
        df = selectUsers(actions1, [i])
        length1 = length1.append( pd.Series([ df.last_valid_index() - df.first_valid_index() + 1 ]), ignore_index=True )
        df = selectUsers(actions3, [i])
        length3 = length3.append( pd.Series([ df.last_valid_index() - df.first_valid_index() + 1 ]), ignore_index=True )
    
    return length1, length3

def sessionHistograms():
    savePath = 'D:/Diplomadolgozat/ActionsPerUserHistogram/'
    Path(savePath).mkdir(parents=True, exist_ok=True)
    actions_per_user1, actions_per_user3 = countActionsPerUser()

    fig = plt.figure()
    plt.hist(actions_per_user1)
    fig.savefig(savePath + 'actions_per_user_1min.png')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(actions_per_user3)
    fig.savefig(savePath + 'actions_per_user_3min.png')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)

    # N is the count in each bin, bins is the lower-limit of the bin
    N, _, patches = ax.hist(actions_per_user1)

    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = matplotlib.colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    fig.savefig(savePath + 'actions_per_user_1min_colored.png')
    plt.close(fig)

    ##################### 3 min #################################
    fig, ax = plt.subplots(1, 1)
    
    # N is the count in each bin, bins is the lower-limit of the bin
    N, _, patches = ax.hist(actions_per_user3)

    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = matplotlib.colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    fig.savefig(savePath + 'actions_per_user_3min_colored.png')
    plt.close(fig)

def plotSingleAction(user, nr_action, min):
    dataFolder = 'D:/Diplomadolgozat/UsersMerged/user_data_' + str(min) + 'min.csv'
    actionsFolder = 'D:/Diplomadolgozat/ActionsMerged/user_actions_' + str(min) + 'min.csv'
    savePath = 'D:/Diplomadolgozat/SingleTrajectories/'
    Path(savePath).mkdir(parents=True, exist_ok=True)
    
    data = pd.read_csv(dataFolder)
    actions = pd.read_csv(actionsFolder)

    user_data = selectUsers(data, [user])
    user_actions = selectUsers(actions, [user])

    user_starti = user_actions.iloc[nr_action - 1 + user_actions.first_valid_index()]['Start index']
    user_stopi = user_actions.iloc[nr_action - 1 + user_actions.first_valid_index()]['Stop index']

    action = user_data[user_starti:user_stopi+1]
    x = action['x']
    y = action['y']
    fig = plt.figure()
    plt.plot(x,y,marker='o')
    var = -1
    plotted_last = 0
    max_x = max(x)
    max_y = max(y)
    min_x = np.min(x)
    min_y = np.min(y)
    distance_x = max_x - min_x
    distance_y = max_y - min_y
    limit_percent = 1
    for i in range(len(x)):
        if i > 0:
            distance_between_points_x = abs(x.iloc[i] - x.iloc[plotted_last])
            distance_between_points_y = abs(y.iloc[i] - y.iloc[plotted_last])
            distance_in_percent_x = (distance_between_points_x * 100)/distance_x
            distance_in_percent_y = (distance_between_points_y * 100)/distance_y
            # Just plot that points where the distance between it and the
            # previous plotted point is greater than 'limit_percent'
            if distance_in_percent_x > limit_percent and distance_in_percent_y > limit_percent:
                plotted_last = i
                if i <= 20:
                    plt.annotate('P' + str(i+1),
                                (x.iloc[i], y.iloc[i]),
                                textcoords='offset points',
                                xytext=(0,5)
                                )
                elif i > 20 and i < 26:
                    plt.annotate('P' + str(i+1),
                                (x.iloc[i], y.iloc[i]),
                                textcoords='offset points',
                                xytext=(0,10 * var)
                                )
                else:
                    plt.annotate('P' + str(i+1),
                                (x.iloc[i], y.iloc[i]),
                                textcoords='offset points',
                                xytext=(10 * var,0)
                                )
                var = var * (-1)
        elif i == (len(x) - 1):
            plt.annotate('P' + str(i+1),
                                (x.iloc[i], y.iloc[i]),
                                textcoords='offset points',
                                xytext=(0,5)
                                )
        else:
            plt.annotate('P' + str(i+1),
                                (x.iloc[i], y.iloc[i]),
                                textcoords='offset points',
                                xytext=(0,5)
                                )
    
    # plt.show()
    fig.savefig(savePath + 'user' + str(user) + '_' + str(nr_action) + '_' + str(min) + 'min_distantiated.png')
    plt.close(fig)

def plotForUI(dataset=0):
    if dataset == 0:
        dataFolder = 'D:/Diplomadolgozat/Users/'
        actionsFolder = 'D:/Diplomadolgozat/Actions/'
    saveFolder = 'D:/Diplomadolgozat/JustTrajDxDy/'

    Path(saveFolder).mkdir(parents=True, exist_ok=True)

    for dirname, _, filenames in os.walk(dataFolder):
        dirnameToSer = pd.Series([dirname])
        try:
            user = dirnameToSer.str.findall('user\d{1,3}').iloc[0][0]
        except IndexError:
            continue
        
        Path(saveFolder + user.capitalize()).mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            print(dirname + ' ' + filename)
            filenameToSer = pd.Series([filename])
            minute = filenameToSer.str.findall('\dmin').iloc[0][0]
            Path(saveFolder + user.capitalize() + '/' + minute).mkdir(parents=True, exist_ok=True)
            Path(saveFolder + user.capitalize() + '/' + minute + '/tr').mkdir(parents=True, exist_ok=True)
            Path(saveFolder + user.capitalize() + '/' + minute + '/dx').mkdir(parents=True, exist_ok=True)
            Path(saveFolder + user.capitalize() + '/' + minute + '/dy').mkdir(parents=True, exist_ok=True)
            dataPath = os.path.join(dirname, filename)
            data = pd.read_csv(dataPath)

            actionsPath = actionsFolder + user.capitalize() + '/' + minute + '.csv'
            actions = pd.read_csv(actionsPath)

            for i in range(actions.last_valid_index() + 1):
                if i > 14:
                    break
                starti = actions.iloc[i]['Start index']
                stopi = actions.iloc[i]['Stop index']
                x = np.array(data.iloc[starti:stopi + 1]['x'])
                dx = np.diff(x)
                y = np.array(data.iloc[starti:stopi + 1]['y'])
                dy = np.diff(y)

                # fig = plt.figure(1)
                # ax = fig.add_subplot(1,1,1)
                # ax.plot(dx)
                # ax.spines['left'].set_position('zero')
                # ax.spines['bottom'].set_position('zero')
                # ax.spines['top'].set_position('zero')
                # ax.spines['right'].set_position('zero')
                # plt.show()

                fig = plt.figure()
                max_y = np.max(y)
                plt.title(str(i))
                plt.ylabel('y')
                plt.xlabel('x')
                plt.plot(x, max_y - y, marker='.')
                plt.plot(x[:1], max_y - y[:1], 'r.', label='Starting')
                plt.legend()
                fig.savefig(saveFolder + user.capitalize() + '/' + minute +
                    '/tr/' + str(i) + '.png')
                plt.close(fig)

                fig = plt.figure()
                plt.title(str(i))
                plt.ylabel('dx')
                plt.xlabel('time')
                plt.plot(dx)
                fig.savefig(saveFolder + user.capitalize() + '/' + minute +
                    '/dx/' + str(i) + '.png')
                plt.close(fig)

                fig = plt.figure()
                plt.title(str(i))
                plt.ylabel('dy')
                plt.xlabel('time')
                plt.plot(dy)
                fig.savefig(saveFolder + user.capitalize() + '/' + minute +
                    '/dy/' + str(i) + '.png')
                plt.close(fig)
                
                # plt.show()

                # return

def plotBezierForUI(dataset=0):
    if dataset == 0:
        dataFolder = 'D:/Diplomadolgozat/Users/'
        actionsFolder = 'D:/Diplomadolgozat/Actions/'
        bezDataFolder = 'D:/Diplomadolgozat/UsersGen/'
        bezActionsFolder = 'D:/Diplomadolgozat/ActionsGen/'
    saveFolder = 'D:/Diplomadolgozat/BezierTrajDxDy/'

    Path(saveFolder).mkdir(parents=True, exist_ok=True)

    for dirname, _, filenames in os.walk(dataFolder):
        dirnameToSer = pd.Series([dirname])
        try:
            user = dirnameToSer.str.findall('user\d{1,3}').iloc[0][0]
        except IndexError:
            continue
        
        Path(saveFolder + user.capitalize()).mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            if( filename.find('1min') == -1 ):  # Skipping the 3 min sections
                continue

            print(dirname + ' ' + filename)
            filenameToSer = pd.Series([filename])
            minute = filenameToSer.str.findall('\dmin').iloc[0][0]
            Path(saveFolder + user.capitalize() + '/' + minute).mkdir(parents=True, exist_ok=True)
            Path(saveFolder + user.capitalize() + '/' + minute + '/tr').mkdir(parents=True, exist_ok=True)
            Path(saveFolder + user.capitalize() + '/' + minute + '/dx').mkdir(parents=True, exist_ok=True)
            Path(saveFolder + user.capitalize() + '/' + minute + '/dy').mkdir(parents=True, exist_ok=True)

            dataPath = os.path.join(dirname, filename)
            data = pd.read_csv(dataPath)
            bezDataPath = bezDataFolder + user + '/' + filename
            bezData = pd.read_csv(bezDataPath)
            actionsPath = actionsFolder + user.capitalize() + '/' + minute + '.csv'
            actions = pd.read_csv(actionsPath)
            bezActionsPath = bezActionsFolder + user.capitalize() + '/' + minute + '.csv'
            bezActions = pd.read_csv(bezActionsPath)

            for i in range(actions.last_valid_index() + 1):
                if i > 14:
                    break
                starti = actions.iloc[i]['Start index']
                stopi = actions.iloc[i]['Stop index']
                x = np.array(data.iloc[starti:stopi + 1]['x'])
                dx = np.diff(x)
                y = np.array(data.iloc[starti:stopi + 1]['y'])
                dy = np.diff(y)

                bstarti = bezActions.iloc[i]['Start index']
                bstopi = bezActions.iloc[i]['Stop index']
                bx = np.array(bezData.iloc[bstarti:bstopi + 1]['x'])
                bdx = np.diff(bx)
                by = np.array(bezData.iloc[bstarti:bstopi + 1]['y'])
                bdy = np.diff(by)

                # fig = plt.figure(1)
                # ax = fig.add_subplot(1,1,1)
                # ax.plot(dx)
                # ax.spines['left'].set_position('zero')
                # ax.spines['bottom'].set_position('zero')
                # ax.spines['top'].set_position('zero')
                # ax.spines['right'].set_position('zero')
                # plt.show()

                fig = plt.figure()
                max_y = np.max(y)
                max_by = np.max(by)
                plt.title(str(i))
                plt.ylabel('y')
                plt.xlabel('x')
                plt.plot(x, max_y - y, marker='.', label='human')
                plt.plot(bx, max_by - by, '--', label='Bézier')
                plt.plot(x[:1], max_y - y[:1], 'r.', label='Starting')
                plt.plot(bx[:1], max_by - by[:1], 'r.')
                plt.legend()
                fig.savefig(saveFolder + user.capitalize() + '/' + minute +
                    '/tr/' + str(i) + '.png')
                plt.close(fig)

                fig = plt.figure()
                plt.title(str(i))
                plt.ylabel('dx')
                plt.xlabel('time')
                plt.plot(dx, label='human')
                plt.plot(bdx, '--', label='Bézier')
                plt.legend()
                fig.savefig(saveFolder + user.capitalize() + '/' + minute +
                    '/dx/' + str(i) + '.png')
                plt.close(fig)

                fig = plt.figure()
                plt.title(str(i))
                plt.ylabel('dy')
                plt.xlabel('time')
                plt.plot(dy, label='human')
                plt.plot(bdy, '--', label='Bézier')
                plt.legend()
                fig.savefig(saveFolder + user.capitalize() + '/' + minute +
                    '/dy/' + str(i) + '.png')
                plt.close(fig)
                
                # plt.show()

                # return
                
def printNumOfActionsPerUser():
    length1, length3 = countActionsPerUser()
    print(np.average(length1))
    print(np.average(length3))

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

#################################################################
#                                                               #
#                       Classification                          #
#                                                               #
#################################################################

# users - range of users e.g. [1, 11)
def selectUsers( df, users ):
    df =  df.loc[ df.iloc[:, -1].isin(users) ]
    return df

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
    scores_filename = 'scores_strong_' + str(max(users)) + '.csv'
    if mode == 1:
        scores_filename = 'scores_' + str(max(users)) + '.csv'
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

#################################################################
#                                                               #
#                       Generate mouse movements                #
#                                                               #
#################################################################

generate_mode = Enum('Generate_mode', 'synth avg noncentral_f')
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

def generateMouseMovementsSynth():
    ## Paths
    path_for_data = 'D:/Diplomadolgozat/Users/'
    path_for_actions = 'D:/Diplomadolgozat/Actions/'
    dataSavePath = 'D:/Diplomadolgozat/UsersSynth/'
    actionsSavePath = 'D:/Diplomadolgozat/ActionsSynth/'

    Path(dataSavePath).mkdir(parents=True, exist_ok=True)
    Path(actionsSavePath).mkdir(parents=True, exist_ok=True)

    ## Initializing
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
                x = np.array(data.iloc[user_starti:user_stopi+1]['x'])
                y = np.array(data.iloc[user_starti:user_stopi+1]['y'])
                
                _dt = similarTimeDifferences(t, generate_mode.synth)
                
                ## Build up timestamp from synthesized dt array
                start_t = time_recording
                ind = 0
                _time_stamp = pd.Series([])
                _time_stamp[ind] = start_t
                ind += 1
                for i in _dt:
                    start_t = start_t + np.abs(i)
                    _time_stamp[ind] = start_t
                    ind = ind + 1

                time_stamp = _time_stamp.round()
                time_stamp = time_stamp.astype(int)
                time_recording = time_stamp.iloc[-1]

                _dx = similarCoordinateDifferences(x)
                _dy = similarCoordinateDifferences(y)

                _x = np.array(np.zeros_like(x))
                _x[0] = x[0]
                for i in range(len(_dx)):
                    _x[i+1] = _x[i] + _dx[i]

                _y = np.array(np.zeros_like(y))
                _y[0] = y[0]
                for i in range(len(_dy)):
                    _y[i+1] = _y[i] + _dy[i]

                ## Appending to raw data and to actions
                df = pd.DataFrame({})
                df['client timestamp'] = time_stamp
                df['x'] = _x
                df['y'] = _y
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


# pd.set_option('display.max_colwidth', 500)
np.set_printoptions(threshold=sys.maxsize)
def generateMouseMovementsAE():
    ## Paths
    diffsFolder_1min = 'D:/Diplomadolgozat/Diffs/diffs_1min.csv'
    diffsFolder_3min = 'D:/Diplomadolgozat/Diffs/diffs_3min.csv'
    diffsSavePath = 'D:/Diplomadolgozat/DiffsGenAEConv/'

    diffs_1min = pd.read_csv(diffsFolder_1min)
    diffs_3min = pd.read_csv(diffsFolder_3min)

    Path(diffsSavePath).mkdir(parents=True, exist_ok=True)

    # df = selectUsers(diffs_3min, [1])
    # df2 = selectUsers(diffs_1min, [1])
    X_train = diffs_3min.iloc[:2, :128]
    X_test = diffs_1min.iloc[:2, :128]

    df = similarActionsWithAEConv(X_train, X_test)

def generateMouseMovementsAE2():
    ## Paths
    path_for_data = 'D:/Diplomadolgozat/Users/'
    path_for_actions = 'D:/Diplomadolgozat/Actions/'
    dataSavePath = 'D:/Diplomadolgozat/UsersGenAEOneByOne/'
    actionsSavePath = 'D:/Diplomadolgozat/ActionsGenAEOneByOne/'

    Path(dataSavePath).mkdir(parents=True, exist_ok=True)
    Path(actionsSavePath).mkdir(parents=True, exist_ok=True)

    ## Initializing
    count_users = 0
    folders = 0

    # Iterating through user folders
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

        read_data = {
            '1min' : pd.DataFrame({}),
            '3min' : pd.DataFrame({})
        }

        read_actions = {
            '1min' : pd.DataFrame({}),
            '3min' : pd.DataFrame({})
        }

        # Iterating through files in user's folder
        for filename in filenames:
            ## Show progress percentage
            # progress = (count_users * 100)/120
            # progress = "{:.2f}".format(progress)
            # print('\rProgress: ' + str(progress) + '%', end='')
            print(dirname, filename)

            filenameToSer = pd.Series([filename])
            minute = filenameToSer.str.findall('\dmin').iloc[0][0]
            dataPath = os.path.join(dirname, filename)
            read_data[minute] = pd.read_csv(dataPath)

            actionsPath = path_for_actions + user.capitalize() + '/' + minute + '.csv'
            read_actions[minute] = pd.read_csv(actionsPath)

        # After both files were read into DataFrames
        train = read_data['3min']
        test = read_data['1min']

        train = train.drop(columns=['button', 'state'])
        test = test.drop(columns=['button', 'state'])
        df = pd.DataFrame({})

        for column in train.columns:
            x_train = train[column]
            x_test = test[column]
            x_train = np.array(x_train)
            x_test = np.array(x_test)
            x_train = x_train.reshape(-1, 1)
            x_test = x_test.reshape(-1, 1)

            generated_data, loss = similarActionsWithAE(x_train, x_test)

            while (loss > 0.001):
                print('############################################')
                print('############ Loss was too high #############')
                print(loss, '--->', 0.001)
                print('############################################')
                generated_data, loss = similarActionsWithAE(x_train, x_test)

            df[column] = generated_data[:,0]

        # t = df['client timestamp']
        # t = np.sort(t)
        # x = df['x']
        # # y = df.iloc[:,2]

        # # Gen t
        # dt = np.diff(t)

        # # Real t
        # rt = test['client timestamp']
        # drt = np.diff(rt)

        # plt.figure()
        # plt.plot(drt)

        # plt.figure()
        # plt.plot(dt)

        # dx = np.diff(x)
        # rx = test['x']
        # drx = np.diff(rx)

        # plt.figure()
        # plt.plot(drx)

        # plt.figure()
        # plt.plot(dx)


        # plt.show()

        # return
        
        ## Save generated raw data and actions
        df = df.astype(int)
        df.to_csv(dataSavePath + user + '/1min.csv', index = False)
        read_actions['1min'].to_csv(actionsSavePath + user.capitalize()
            + '/1min.csv', index = False)
        
        return

def similarActionsWithAEConv(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    # scaler = MinMaxScaler(feature_range=(0,1))
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.fit_transform(x_test)
    # print(x_test)
    # plt.figure()
    # plt.plot(x_test[0])
    # plt.show()
    # return
    # x_train = x_train[0]
    # x_test = x_test[0]
    # x_train = np.reshape(x_train, (2, 128))
    # x_test = np.reshape(x_test, (2, 128))
    # x_train = x_train.reshape(-1, 2)
    # x_test = x_test.reshape(-1, 2)

    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # print(x_train.shape)
    # print(x_test.shape)

    rows, _ = x_train.shape
    # BATCH_SIZE = int(rows/50)
    BATCH_SIZE = 1
    EPOCHS = 25
    # input_shape = keras.Input(shape=(256, 1), batch_size=(256))
    input_shape = keras.Input(shape=(128, 1))
    fcn_filters = 128
    conv1 = Conv1D(filters = fcn_filters, kernel_size=3, padding='same', activation='relu')(input_shape)
    conv2 = Conv1D(filters = 2*fcn_filters, kernel_size=3, padding='same', activation='relu')(conv1)
    conv3 = Conv1D(filters = fcn_filters, kernel_size=3, padding='same', activation='relu')(conv2)

    conv3 = Conv1DTranspose( filters = fcn_filters, kernel_size=3, padding='same', activation='relu')( conv3 )
    conv2 = Conv1DTranspose( filters = 2*fcn_filters, kernel_size=3, padding='same', activation='relu')( conv3 )
    conv1 = Conv1DTranspose( filters = fcn_filters, kernel_size=3, padding='same', activation='relu')( conv2 )
    # decoded = Conv1D(filters=1, kernel_size=4, padding='same', activation='relu')(conv1)

    autoencoder = keras.Model(input_shape, conv1)
    autoencoder.summary()
    autoencoder.compile(optimizer='rmsprop', loss=keras.losses.mean_absolute_error)

    autoencoder.fit(x_train, x_train,
                    epochs=EPOCHS,
                    # validation_data=(x_test, x_test),
                    batch_size=BATCH_SIZE,
                    shuffle=True
                    # callbacks=[TensorBoard(log_dir='/tmp/convAutoencoder')]
                    )

    decoded_data = autoencoder.predict(x_test)
    # decoded_data = np.array(decoded_data[0,:])
    # decoded_data = decoded_data.reshape(1,-1)
    # print(decoded_data)
    print(x_test.shape)
    print(x_test[0].shape)
    print(decoded_data.shape)
    print(decoded_data[:,:,0].shape)
    # decoded_data = scaler.inverse_transform(decoded_data)
    # x_test = x_test.reshape(-1, 1)
    # plt.figure()
    # plt.plot(x_test)
    # plt.figure()
    # plt.plot(decoded_data[0])
    # plt.show()
    decoded_data = decoded_data[:,:,0]
    print(decoded_data[0,:])

def similarActionsWithAE(x_train, x_test):
    standardScaler = MinMaxScaler(feature_range=(0,1))
    x_train = standardScaler.fit_transform(x_train)
    x_test = standardScaler.fit_transform(x_test)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    # BATCH_SIZE = 128
    rows, _ = x_train.shape
    BATCH_SIZE = int(rows/50)
    print('BATCH_SIZE: ', BATCH_SIZE)
    EPOCHS = 10
    # This is the size of our encoded representations
    encoding_dim = 4
    # The data dimension is 1, because 1 row * 1 col
    data_dim = 1
    # This is our input image
    input_img = keras.Input(shape=(data_dim,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # encoded = layers.Dense(16, activation='relu')(encoded)
    # encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    # "decoded" is the lossy reconstruction of the input
    # decoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    # decoded = layers.Dense(16, activation='relu')(decoded)
    decoded = layers.Dense(data_dim, activation='relu')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)

    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    
    autoencoder.compile(optimizer='rmsprop', loss='mse')
    autoencoder.summary()
    history = autoencoder.fit(x_train, x_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    plot_history(history)

    loss = autoencoder.evaluate(x_train, x_train,
                    batch_size=BATCH_SIZE)

    # Encode and decode some digits
    # Note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    decoded_imgs = standardScaler.inverse_transform(decoded_imgs)
    return decoded_imgs, loss

def plot_history( history ):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Autoencoder Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # plt.savefig(stt.TRAINING_CURVES_PATH+'/' + model_name +  '.png', format='png')

# def similarActionsWithAE2(train, test):
#     VALIDATE_SIZE = 0.2
#     RANDOM_SEED = 42
#     EPOCHS = 100
#     BATCH_SIZE = 256

#     # setting random seeds for libraries to ensure reproducibility
#     np.random.seed(RANDOM_SEED)
#     rn.seed(RANDOM_SEED)
#     tf.compat.v1.set_random_seed(RANDOM_SEED)

#     X_train = train.drop(columns=['button', 'state'])
#     X_test = test.drop(columns=['button', 'state'])

#     X_train, X_validate = train_test_split(X_train, 
#                                        test_size=VALIDATE_SIZE, 
#                                        random_state=RANDOM_SEED)



#     # configure our pipeline
#     pipeline = Pipeline([('normalizer', Normalizer()),
#                         ('scaler', MinMaxScaler())])
    
#     # get normalization parameters by fitting to the training data
#     pipeline.fit(X_train)

#     # transform the training and validation data with these parameters
#     X_train_transformed = pipeline.transform(X_train)
#     X_validate_transformed = pipeline.transform(X_validate)

#     input_dim = 3

#     autoencoder = tf.keras.models.Sequential([
    
#         # deconstruct / encode
#         tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim, )), 
#         tf.keras.layers.Dense(3, activation='elu'),
#         tf.keras.layers.Dense(3, activation='elu'),
#         tf.keras.layers.Dense(3, activation='elu'),
#         tf.keras.layers.Dense(3, activation='elu'),
        
#         # reconstruction / decode
#         tf.keras.layers.Dense(3, activation='elu'),
#         tf.keras.layers.Dense(3, activation='elu'),
#         tf.keras.layers.Dense(3, activation='elu'),
#         tf.keras.layers.Dense(input_dim, activation='elu')
        
#     ])

#     # https://keras.io/api/models/model_training_apis/
#     autoencoder.compile(optimizer="adam", 
#                         loss="mse",
#                         metrics=["acc"])

#     history = autoencoder.fit(
#         X_train_transformed, X_train_transformed,
#         shuffle=False,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         validation_data=(X_validate_transformed, X_validate_transformed)
#     )

#     # transform the test set with the pipeline fitted to the training set
#     X_test_transformed = pipeline.transform(X_test)

#     # pass the transformed test set through the autoencoder to get the reconstructed result
#     reconstructions = autoencoder.predict(X_test_transformed)
#     return reconstructions

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

#################################################################
#                                                               #
#                       Binary classification                   #
#                                                               #
#################################################################

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

#################################################################
#                                                               #
#                       X directional dynamics                  #
#                                                               #
#################################################################

## Reference:
## Hide ticks when plot: https://stackoverflow.com/questions/29988241/python-hide-ticks-but-show-tick-labels
## Walk dir without digging in: https://stackoverflow.com/questions/229186/os-walk-without-digging-into-directories-below
def plotXDirectionalDynamics(dataset=0):
    data_folder = 'D:/Diplomadolgozat/Users/'
    actions_folder = 'D:/Diplomadolgozat/Actions/'

    data_gen_folder = 'D:/Diplomadolgozat/UsersGen/'
    actions_gen_folder = 'D:/Diplomadolgozat/ActionsGen/'
    x_directions_folder = 'D:/Diplomadolgozat/XDirectionalDynamics/'
    if dataset == 1:
        data_gen_folder = 'D:/Diplomadolgozat/UsersSynth/'
        actions_gen_folder = 'D:/Diplomadolgozat/ActionsSynth/'
        x_directions_folder = 'D:/Diplomadolgozat/XDirectionalDynamicsSynth/'
    elif dataset == 2:
        data_gen_folder = 'D:/Diplomadolgozat/UsersGenAEOneByOne/'
        actions_gen_folder = 'D:/Diplomadolgozat/ActionsGenAEOneByOne/'
        x_directions_folder = 'D:/Diplomadolgozat/XDirectionalDynamicsAE/'
    Path(x_directions_folder).mkdir(parents=True, exist_ok=True)

    count_users = 0
    folders = dict()

    ## Count the folders so we will know where we've
    ## been last time to continue
    ## Comment these two lines if the files need to
    ## be regenerated
    # try:
    #     for item in os.listdir(x_directions_folder):
    #         folders[str(item)] = 1
    # except:
    #     pass

    for dirname, _, filenames in os.walk(data_folder):
        dirnameToSer = pd.Series([dirname])
        try:
            user = dirnameToSer.str.findall('user\d{1,3}').iloc[0][0]
        except IndexError:
            continue

        try:
            if folders[user.capitalize()]:
                count_users += 1
                continue
        except:
            folders[str(user.capitalize())] = 1
        
        Path(x_directions_folder + user.capitalize()).mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            if( filename.find('1min') == -1 ):  # Skipping the 3 min sections
                continue                        #
            
            ## Show progress percentage
            progress = (count_users * 100)/120
            progress = "{:.2f}".format(progress)
            print('\rProgress: ' + str(progress) + '%', end='')

            filenameToSer = pd.Series([filename])
            minute = filenameToSer.str.findall('\dmin').iloc[0][0]
            Path(x_directions_folder + user.capitalize() + '/' + minute).mkdir(parents=True, exist_ok=True)

            path_for_data = os.path.join(dirname, filename)
            data = pd.read_csv(path_for_data)

            path_for_actions = actions_folder + user.capitalize() + '/' + minute + '.csv'
            actions = pd.read_csv(path_for_actions)

            # This for other generated datas
            path_for_data_gen = data_gen_folder + user + '/' + filename
            # This for AE data
            if dataset == 2:
                path_for_data_gen = data_gen_folder + user + '/1min.csv'
            data_gen = pd.read_csv(path_for_data_gen)

            path_for_actions_gen = actions_gen_folder + user.capitalize() + '/' + minute + '.csv'
            actions_gen = pd.read_csv(path_for_actions_gen)

            for i in range(len(actions)):
                ## Start-End indices
                user_starti = actions.iloc[i]['Start index']
                user_stopi = actions.iloc[i]['Stop index']
                gen_starti = actions_gen.iloc[i]['Start index']
                gen_stopi = actions_gen.iloc[i]['Stop index']

                ## Raw data from real
                x = data[user_starti:user_stopi+1]['x']
                t = data[user_starti:user_stopi+1]['client timestamp']

                ## Raw data from generated
                x_gen = data_gen[gen_starti:gen_stopi+1]['x']
                t_gen = data_gen[gen_starti:gen_stopi+1]['client timestamp']

                ## Diffs
                dx = np.diff(x)
                dt = np.diff(t)
                dx_gen = np.diff(x_gen)
                dt_gen = np.diff(t_gen)

                ## Velocities
                out1 = np.ones_like(dx)
                out1 = out1.astype(float)
                out2 = np.ones_like(dx_gen)
                out2 = out2.astype(float)
                v_x = np.divide(dx, dt, out=out1, where = dt != 0)
                v_x_gen = np.divide(dx_gen, dt_gen, out=out2, where = dt_gen != 0)

                ## Diffs
                dv_x = np.diff(v_x)
                ddt = np.diff(dt)
                dv_x_gen = np.diff(v_x_gen)
                ddt_gen = np.diff(dt_gen)

                ## Accelerations
                out1 = np.ones_like(dv_x)
                out2 = np.ones_like(dv_x_gen)
                a_x = np.divide(dv_x, ddt, out=out1, where = ddt != 0)
                a_x_gen = np.divide(dv_x_gen, ddt_gen, out=out2, where = ddt_gen != 0)

                ## Subplots
                fig, axs = plt.subplots(3, 2)
                fig.suptitle(str(i) + ', ' + actions.iloc[i]['Action type'])

                ## X position
                maximums = np.maximum(dx, dx_gen)
                dx_max = np.max(maximums)
                minimums = np.minimum(dx, dx_gen)
                dx_min = np.min(minimums)
                dist = dx_max - dx_min

                axs[0, 0].set_title('Ember')
                axs[0, 0].plot(dx, '-o')
                axs[0, 0].set_ylim(dx_min - (5*dist)/100, dx_max + (5*dist)/100)
                plt.setp(axs[0, 0].get_xticklabels(), visible=False)

                axs[0, 1].set_title('Bot')
                axs[0, 1].set(ylabel='dx')
                axs[0, 1].plot(dx_gen, '-o')
                axs[0, 1].set_ylim(dx_min - (5*dist)/100, dx_max + (5*dist)/100)
                plt.setp(axs[0, 1].get_xticklabels(), visible=False)
                plt.setp(axs[0, 1].get_yticklabels(), visible=False)
                
                ## X Speed
                maximums = np.maximum(v_x, v_x_gen)
                v_x_max = np.max(maximums)
                minimums = np.minimum(v_x, v_x_gen)
                v_x_min = np.min(minimums)
                dist = v_x_max - v_x_min

                axs[1, 0].plot(v_x, '-o')
                axs[1, 0].set_ylim(v_x_min - (5*dist)/100, v_x_max + (5*dist)/100)
                plt.setp(axs[1, 0].get_xticklabels(), visible=False)
                
                axs[1, 1].set(ylabel='X sebesség')
                axs[1, 1].plot(v_x_gen, '-o')
                axs[1, 1].set_ylim(v_x_min - (5*dist)/100, v_x_max + (5*dist)/100)
                plt.setp(axs[1, 1].get_xticklabels(), visible=False)
                plt.setp(axs[1, 1].get_yticklabels(), visible=False)
                
                ## X Acceleration
                maximums = np.maximum(a_x, a_x_gen)
                a_x_max = np.max(maximums)
                minimums = np.minimum(a_x, a_x_gen)
                a_x_min = np.min(minimums)
                dist = a_x_max - a_x_min

                axs[2, 0].plot(a_x, '-o')
                axs[2, 0].set_ylim(a_x_min - (5*dist)/100, a_x_max + (5*dist)/100)
                
                axs[2, 1].set(ylabel='X gyorsulás')
                axs[2, 1].plot(a_x_gen, '-o')
                axs[2, 1].set_ylim(a_x_min - (5*dist)/100, a_x_max + (5*dist)/100)
                plt.setp(axs[2, 1].get_yticklabels(), visible=False)
                
                for ax in fig.get_axes():
                    ax.yaxis.set_label_position("right")
                
                fig.savefig(x_directions_folder + user.capitalize() + '/' + 
                            minute + '/' + str(i) + ', ' + actions.iloc[i]['Action type'] + '.png')
                plt.close(fig)
        # return
        count_users += 1

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