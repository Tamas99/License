from __future__ import annotations, division
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.settings import selectUsers

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

def saveNumOfActionsToFile():
    savePath = 'D:/Diplomadolgozat/NumOfActions/'
    Path(savePath).mkdir(parents=True, exist_ok=True)
    l1, l3 = countActionsPerUser()
    l1.to_csv(savePath + 'lengths_1min.csv', header=False, index=False)
    l3.to_csv(savePath + 'lengths_3min.csv', header=False, index=False)

def NumOfActionsToString():
    savePath = 'D:/Diplomadolgozat/NumOfActions/'
    Path(savePath).mkdir(parents=True, exist_ok=True)
    l1, l3 = countActionsPerUser()
    string = ''
    index = 1
    for i in l1:
        string = string + 'User' + str(index) + ': ' + str(i) + ', '
        index += 1

    string2 = ''
    index = 1
    for i in l3:
        string2 = string2 + 'User' + str(index) + ': ' + str(i) + ', '
        index += 1

    print(string)
    print(string2)

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

def printNumOfActionsPerUser():
    length1, length3 = countActionsPerUser()
    print(np.average(length1))
    print(np.average(length3))