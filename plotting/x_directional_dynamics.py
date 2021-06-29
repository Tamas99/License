from __future__ import annotations, division
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

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
