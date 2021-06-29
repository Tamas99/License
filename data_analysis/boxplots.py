from __future__ import annotations, division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.settings import selectUsers

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
