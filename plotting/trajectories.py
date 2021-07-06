from __future__ import annotations, division
from matplotlib.lines import Line2D
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from utils.settings import selectUsers

def plotTrajectoriesUser10OnSingleDiagram():
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

def plotTrajectoriesOnSingleDiagram(len, actionsPerFig):
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
                plt.ylabel("y")
                plt.xlabel("x")
                plt.plot(user_data[user_starti:user_stopi+1]['x'], user_data[user_starti:user_stopi+1]['y'], color)
                plt.plot(user_data.iloc[user_starti]['x'], user_data.iloc[user_starti]['y'], color='green', marker='o', linewidth=2)
                plt.plot(user_data.iloc[user_stopi]['x'], user_data.iloc[user_stopi]['y'], color='black', marker='x', linewidth=2)

            #     plt.figure(2)
            #     plt.ylabel("y")
            #     plt.xlabel("x")
            #     plt.plot(user_data[user_starti:user_stopi+1]['x'], user_data[user_starti:user_stopi+1]['y'])
            #     plt.plot(user_data.iloc[user_starti]['x'], user_data.iloc[user_starti]['y'], color='green', marker='o', linewidth=2)
            #     plt.plot(user_data.iloc[user_stopi]['x'], user_data.iloc[user_stopi]['y'], color='black', marker='x', linewidth=2)
            #     plt.gca().invert_yaxis()

            # plt.figure(1)
            legend_elements = [
                Line2D([0], [0], color='r', label= str(count_PCs) + '-PC'),
                                Line2D([0], [0], color='b', label= str(count_DDs) + '-DD'),
                                Line2D([0], [0], marker='o', color='g', label='Start'),
                                Line2D([0], [0], marker='x', color='k', label='Stop')
                        ]

            plt.legend(handles=legend_elements, loc='best')
            plt.gca().invert_yaxis()
            fig.savefig(savePath + str(from_action) + '-' + str(to_action) + '.png')
            plt.close(fig)

            # plt.figure(2)
            # legend_elements = [
            #                     Line2D([0], [0], marker='o', color='g', label='Start'),
            #                     Line2D([0], [0], marker='x', color='k', label='Stop')
            #             ]

            # plt.legend(handles=legend_elements, loc='best')
            # fig2.savefig(savePath + str(from_action) + '-' + str(to_action) + 'mix.png')
            # plt.close(fig2)

            from_action = to_action
            to_action += actionsPerFig
        
        # Start over again if a user doesn't have anymore action
        from_action = 0
        to_action = actionsPerFig

def plotSingleActionOnSingleDiagram(user, nr_action, min):
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
    plt.ylabel('y')
    plt.xlabel('x')
    plt.plot(x,y,marker='.')
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
    
    plt.gca().invert_yaxis()
    # plt.show()
    fig.savefig(savePath + 'user' + str(user) + '_' + str(nr_action) + '_' + str(min) + 'min_distantiated.png')
    plt.close(fig)

def plotTrajDxDy(dataset=0):
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
                # Not just the first 15
                # if i > 14:
                #     break
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
                plt.title(str(i))
                plt.ylabel('y')
                plt.xlabel('x')
                plt.plot(x, y, marker='.')
                plt.plot(x[:1], y[:1], 'r.', label='Starting')
                plt.legend()
                plt.gca().invert_yaxis()
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

def plotBezierTrajDxDy(dataset=0):
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
            # if( filename.find('1min') == -1 ):  # Skipping the 3 min sections
            #     continue

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
                # Not just the first 15 actions
                # if i > 14:
                #     break
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
                plt.title(str(i))
                plt.ylabel('y')
                plt.xlabel('x')
                plt.plot(x, y, marker='.', label='human')
                plt.plot(bx, by, '--', label='Bézier')
                plt.plot(x[:1], y[:1], 'r.', label='Starting')
                plt.plot(bx[:1], by[:1], 'r.')
                plt.legend()
                plt.gca().invert_yaxis()
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
                