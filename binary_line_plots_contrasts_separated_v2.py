# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:13:03 2017

@author: Tapster
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:35:40 2017

@author: Tapster
"""

import csv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import matplotlib
import os
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import scipy as scipy
from scipy import signal
from scipy import stats 
import json 

from collections import deque

#If we are using python 2.7 or under
if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkFileDialog as filedialog
    import tkMessageBox as messagebox
      
#If we are using python 3.0 or above
elif sys.version_info[0] >= 3:
    import tkinter as tk
    import tkinter.filedialog as filedialog
    import tkinter.messagebox as messagebox

#Each 'Block' of data consists of 46 lines
def read_JAABA_csv_block(csv_reader):
    current_group = deque(maxlen=1)
    
    line_count = 0
    
    groups = []
    parsed_groups = []
    df_list = []

    grab_xdata = None

    grab_ydata = None
    
    header_read = False
    condition_label = None

    while True:
        line = csv_reader.next()
        
        if "end" in line:
           break
        
        if line:
            
            if line[0].startswith('% type'):
                df_type = line
            if line[0].startswith('% title'):
                df_title = line
            if line[0].startswith('% xlabel'):
                df_xlabel = line
            if line[0].startswith('% ylabel'):
                df_ylabel = line
                print(' ')
                print('df_ylabel is: ' + str(df_ylabel[0]))
                condition_label = df_ylabel[0]
                header_read = True
            
            if header_read and '% group' in line[0]:
                line = filter(None, line)
                groups.append(line)
                parsed_groups.append(line[0].lstrip('% group '))
                #save the group name to use when generating dataframes
                current_group.append(line[0].lstrip('% group '))
            
            if grab_xdata:
                line = filter(None, line)
                x_key = current_group[0]
                # convert read in "string version of line into actual numbers"
                # The JAABA datafiles always have a lingering ',' after each data group
                # this results in a ' ' character that float() cannot convert. 
                # (So remember to strip the last element from the list)
                xdata = {x_key:map(float, line[:-1])}
                xdata_df = pd.DataFrame(xdata)
                df_list.append(xdata_df)
                grab_xdata = False
                
            if grab_ydata: 
                line = filter (None, line)
                y_key = '{} binary'.format(current_group[0])
                ydata = {y_key:map(float, line[:-1])}
                ydata_df = pd.DataFrame(ydata)
                df_list.append(ydata_df)
                grab_ydata = False
        
            if '% xdata' in line:
                grab_xdata = True
            if line[0].startswith('% experiment'):
                grab_ydata = True
            
        line_count += 1
        
    output_df = pd.concat(df_list, axis=1)    

    return output_df, condition_label
    
#load our frame cutoffs from file
def load_frame_cutoffs(path):
    with open (path, 'rb') as f:
        csv_reader = csv.reader (f, delimiter = ',')
        dictionary = {row[0] : map(int,row[1:7]) for indx, row in enumerate(csv_reader) if indx != 0}
    return dictionary
    
path_to_cutoff_csv = 'C:\Users\Tapster\Desktop\Sophia\Cutoff Frames_Dictionary.csv'

cutoff_dictionary = load_frame_cutoffs(path_to_cutoff_csv)    

#non-hard coded paths
#root = tk.Tk()
#path_to_T1 = filedialog.askopenfilename(parent=root, title="Select CSV for T1") 
#path_to_T2 = filedialog.askopenfilename(parent=root, title="Select CSV for T2") 
#path_to_T3 = filedialog.askopenfilename(parent=root, title="Select CSV for T3")     

#Advancing Day 1
path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM1_Trial1_Lanes1-6.csv'
path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM1_Trial2_Lanes1-6.csv'
path_to_T3 ='E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM1_Trial3_Lanes1-6.csv'

#Advancing Day 2
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-17_Day2_PM1_Trial1_Lanes1-6.csv'
#path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-17_Day2_PM1_Trial2_Lanes1-6.csv'
#path_to_T3 ='E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-17_Day2_PM1_Trial3_Lanes1-6.csv'

#Retreating Day 1
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Retreating_2015-09-16_Day1_PM1_Trial1_Lanes1-6.csv'
#path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Retreating_2015-09-16_Day1_PM1_Trial2_Lanes1-6.csv'
#path_to_T3 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Retreating_2015-09-16_Day1_PM1_Trial3_Lanes1-6.csv'

#Retreating Day 2
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Retreating_2015-09-17_Day2_PM1_Trial1_Lanes1-6.csv'
#path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Retreating_2015-09-17_Day2_PM1_Trial2_Lanes1-6.csv'
#path_to_T3 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Retreating_2015-09-17_Day2_PM1_Trial3_Lanes1-6.csv'

#Pausing Day 1
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Pausing_2015-09-16_Day1_PM1_Trial1_Lanes1-6.csv'
#path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Pausing_2015-09-16_Day1_PM1_Trial2_Lanes1-6.csv'
#path_to_T3 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Pausing_2015-09-16_Day1_PM1_Trial3_Lanes1-6.csv'

#Pausing Day 2
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Pausing_2015-09-17_Day2_PM1_Trial1_Lanes1-6.csv'
#path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Pausing_2015-09-17_Day2_PM1_Trial2_Lanes1-6.csv'
#path_to_T3 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Pausing_2015-09-17_Day2_PM1_Trial3_Lanes1-6.csv'

list_of_paths_to_load = [path_to_T1, path_to_T2, path_to_T3]

results = {}

for indx, path in enumerate(list_of_paths_to_load):
    with open(path, 'rb') as f:
        csv_reader = csv.reader(f, delimiter=',', skipinitialspace=True)
        classifier_df, condition_label = read_JAABA_csv_block(csv_reader)
        classifier_label = condition_label.lstrip('% ylabel=')
        classifier_label = classifier_label.rstrip(' (%)')
        results['classifier'] = classifier_label
        trial_label = 'Trial {}'.format(indx+1)
        results[trial_label] = classifier_df
        #figure out the key to enter into the cutoff_dictionary
        split_path = path.split("\\") [-1].split("_") [1:-1]
        joined_path = "_".join(split_path)
        cutoffs = cutoff_dictionary[joined_path]

        cutoff_key = "{} cutoffs".format(trial_label)        
        results[cutoff_key] = cutoffs
               
def get_values (group, cutoff):
    for indx, contrast in enumerate(group):

        index = contrast.rstrip(' binary')

        xdata = trial_data[index]
        ydata = trial_data[contrast]
        #============ Code to apply frame cutoffs per lane =================
        #Need to change the FPS to the actual value for that trial if possible
        FPS = 15.
        #convert cutoff time to minutes
        cutoff_time = (cutoff/FPS)/60.
        cutoff_dataframe_index = trial_data[(trial_data[contrast] < cutoff_time)].last_valid_index()
        
        x = xdata[0:-1]

        return x, cutoff_dataframe_index

def get_y_values (group, cutoff): 
    
    for indx, contrast in enumerate(group):

        #print "indx: %d, contrast: %s" % (indx, contrast)
        index = contrast.rstrip(' binary')

        #Load in the x data for our given contrast (contrasts can be lanes, IAA, EtOH, or any other condition)
        xdata = trial_data[index]
        ydata = trial_data[contrast]
        
        #============ Code to apply frame cutoffs per lane =================
        #Need to change the FPS to the actual value for that trial if possible
        FPS = 15.
        #convert cutoff time to minutes
        cutoff_time = (cutoff/FPS)/60.
                     
        x = xdata[0:-1]
        y = ydata[0:-1]
        
        return y

def plot (fig, ax, group, group_cutoffs, trial_cutoffs, cutoff_dataframe_index, x_values, y_values_1, y_values_2, y_values_3, runway, baic, alcohol, title, *colors): 
        
        plot_color = colors[0]
        fill_color = colors[1]
        
        array_list = []
        array_sem= []
        array_plus_sem = []
        array_minus_sem = []

        max_cutoff = float(max(trial_cutoffs)) #finish runway line, start alcohol line
        max_cutoff_time = max_cutoff/15/60
        max_cutoff_time_p1 = (max_cutoff+1)/15/60
        max_cutoff_time_m1 = (max_cutoff-1)/15/60
        end_time = float(x_values.iget(-1))
        min_cutoff = float(min(group_cutoffs)) #start baic line
        min_cutoff_time = min_cutoff/15/60
        
#####################################
#   Determine standard error        #
#####################################
        average_y = (y_values_1 + y_values_2 + y_values_3)/3 
        
        x_values_runway = x_values[0:int(max_cutoff)]
        average_y_runway = average_y[0:int(max_cutoff)]
        x_values_baic = x_values[int(min_cutoff):int(max_cutoff)]
        average_y_baic = average_y[int(min_cutoff):int(max_cutoff)]
        x_values_alcohol = x_values[int(max_cutoff+1):cutoff_dataframe_index]
        average_y_alcohol = average_y[int(max_cutoff+1):cutoff_dataframe_index]
        
        array_list.append([])
        array_list.append([])
        array_list.append([])
        for number in range(cutoff_dataframe_index): 
            array_list[0].append(y_values_1[number])
            array_list[1].append(y_values_2[number])
            array_list[2].append(y_values_3[number])
        array_sem.append(scipy.stats.sem(array_list))
        array_sem_transposed = map (list, zip(*array_sem))
        
        for number2 in range(cutoff_dataframe_index):
            plus_sem = array_sem_transposed[number2] + average_y[number2]
            array_plus_sem.append(float(plus_sem))
            minus_sem = average_y[number2] - array_sem_transposed[number2]
            array_minus_sem.append(float(minus_sem))

#####################################
#       Determine bin values        #
#####################################
        bin_input = 70
        bins = np.linspace(0, end_time, bin_input)
        index = cutoff_dataframe_index
        number_per_bin = index/bin_input
        min_per_bin = number_per_bin/15/60
        
        digitized_x = np.digitize(x_values, bins)
        digitized_avg = np.digitize(average_y, bins)
        x_values_binned = [x_values[digitized_x == i].mean() for i in range(1, len(bins))]
        y_average_binned = [average_y[digitized_x==i].mean()for i in range(1, len(bins))]
        
        runway_cutoff = int(max_cutoff)/number_per_bin
        baic_cutoff = int(min_cutoff)/number_per_bin
        alcohol_end = int(cutoff_dataframe_index)/number_per_bin
                         
        x_runway_binned = x_values_binned[0:runway_cutoff]
        y_runway_binned = y_average_binned[0:runway_cutoff]
        x_baic_binned = x_values_binned[baic_cutoff:runway_cutoff]
        y_baic_binned = y_average_binned[baic_cutoff:runway_cutoff]
        x_alcohol_binned = x_values_binned[runway_cutoff:alcohol_end]
        y_alcohol_binned = y_average_binned[runway_cutoff:alcohol_end]
        
        array_plus_sem = np.array(array_plus_sem)
        digitized_p_sem = [array_plus_sem[digitized_x == i].mean() for i in range(1, len(bins))]
        p_sem_runway = digitized_p_sem[0:runway_cutoff]
        p_sem_baic = digitized_p_sem[baic_cutoff:runway_cutoff]
        p_sem_alcohol = digitized_p_sem[runway_cutoff:alcohol_end]

        array_minus_sem = np.array(array_minus_sem)
        digitized_m_sem = [array_minus_sem[digitized_x == i].mean() for i in range(1, len(bins))]   
        m_sem_runway = digitized_m_sem[0:runway_cutoff]
        m_sem_baic = digitized_m_sem[baic_cutoff:runway_cutoff]
        m_sem_alcohol = digitized_m_sem[runway_cutoff:alcohol_end]
######################################
##          Plot runway              #
######################################     
        if runway: 
#Plot normally            
    ##       ax.plot(x_values_runway, average_y_runway, color = plot_color)
    #        #ax.fill_between(x_values_1, array_plus_sem, array_minus_sem, color = fill_color, alpha = 0.25)

#Plot binned            
 
            ax.plot(x_runway_binned, y_runway_binned, color = plot_color)
            ax.fill_between (x_runway_binned, p_sem_runway, m_sem_runway, color = fill_color, alpha = 0.25)

#Formats plot               
 
            ax.set_title(title, fontsize = 10)
            ax.set_xlabel ('Time (min)', fontsize=10, x = 0.95)
            ax.set_ylabel ('Average total ' + classifier.lower(), fontsize = 10)
            figure_title = ' '.join(split_path).rstrip ('Trial3')
######################################
## Plots end chamber pre-reward      #
######################################               
        if baic: 
#Plot normally            
    #        ax.plot(x_values_baic, average_y_baic, color = plot_color)
    #        #ax.fill_between(x_values_1, array_plus_sem, array_minus_sem, color = fill_color, alpha = 0.25)
    
#Plot binned             
            ax.plot(x_baic_binned, y_baic_binned, color = plot_color)
            ax.fill_between(x_baic_binned, p_sem_baic, m_sem_baic, color = fill_color, alpha = 0.25)
# Formats plot
            ax.set_title(title, fontsize = 10)
            ax.set_xlabel ('Time (min)', fontsize=10, x = 0.95)
            ax.set_ylabel ('Average total ' + classifier.lower(), fontsize = 10)
            figure_title = ' '.join(split_path).rstrip ('Trial3')
            fig1.suptitle ('Plot of ' + figure_title + classifier, x = 0.136, y = 1)
            fig1.patch.set_facecolor('white')
            
        if alcohol: 
#Plot normally          
    #       ax.plot(x_values_alcohol, average_y_alcohol, color = plot_color)
    #       ax.fill_between(x_values_1, array_plus_sem, array_minus_sem, color = fill_color, alpha = 0.25)

#Plot binned             
            ax.plot(x_alcohol_binned, y_alcohol_binned, color = plot_color)
            ax.fill_between(x_alcohol_binned, p_sem_alcohol, m_sem_alcohol, color = fill_color, alpha = 0.25)
    
#Formats plot             
            ax.set_title(title, fontsize = 10)
            ax.set_xlabel ('Time (min)', fontsize=10, x = 0.95)
            ax.set_ylabel ('Average total ' + classifier.lower(), fontsize = 10)
            figure_title = ' '.join(split_path).rstrip ('Trial3')

fig1, ax1 = plt.subplots(3,3, sharex = True, sharey = True) 
fig1 = matplotlib.pyplot.gcf()

#fig1.set_figwidth(28)
#fig1.set_figheight(6)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

fig1.tight_layout()

#This for loop is iterating through different trials in the experiment.
index = 0
for key in sorted(results.keys()):
    if 'cutoffs' in key or 'classifier' in key:
        pass
    else:
        trial_data = results[key]

        trial_cutoffs = results[key + ' cutoffs']

        classifier = results['classifier']

        column_names = trial_data.columns
        
        contrasts = sorted([column_name for column_name in column_names if 'binary' in column_name])
    
        num_contrasts = len(contrasts)
        
        contrast_dict = {}
        
        lane_1 = contrasts [0:1] 
        lane_2 = contrasts[1:2]
        lane_3 = contrasts [2:3]
        lane_4 = contrasts [3:4]
        lane_5 = contrasts [4:5]
        lane_6 = contrasts [5:6]
        control_lanes = contrasts[0:3]
        control_cutoffs = trial_cutoffs[0:3]
        experimental_lanes = contrasts [3:6]
        experimental_cutoffs = trial_cutoffs[3:6]
        
    if key == 'Trial 1':
        x_values, cutoff_dataframe_index1 = get_values (lane_1, trial_cutoffs[0])
        y_values_1 = get_y_values (lane_1, trial_cutoffs [0])
        y_values_2 = get_y_values (lane_2, trial_cutoffs[1])
        y_values_3 = get_y_values (lane_3, trial_cutoffs[2])
        y_values_4 = get_y_values (lane_4, trial_cutoffs[3])
        y_values_5 = get_y_values (lane_5, trial_cutoffs[4])
        y_values_6 = get_y_values (lane_6, trial_cutoffs[5])
               
        plot (fig1, ax1[0,0], control_lanes, control_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_1, y_values_2, y_values_3, True, False, False, 'Trial 1 Runway',  '#2f4f4f', '#778899')
        plot (fig1, ax1[0,0], experimental_lanes, experimental_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_4, y_values_5, y_values_6, True, False, False, 'Trial 1 Runway',  '#000080', '#56A3DC')
        plot (fig1, ax1[1,0], control_lanes, control_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_1, y_values_2, y_values_3, False, True, False, 'Trial 1 End Chamber (Pre-Reward)', '#2f4f4f', '#778899')
        plot (fig1, ax1[1,0], experimental_lanes, experimental_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_4, y_values_5, y_values_6, False, True, False, 'Trial 1 End Chamber (Pre-Reward)', '#000080', '#56A3DC')
        plot (fig1, ax1[2,0], control_lanes, control_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_1, y_values_2, y_values_3, False, False, True, 'Trial 1 End Chamber (Post-Reward)', '#2f4f4f', '#778899')
        plot (fig1, ax1[2,0], experimental_lanes, experimental_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_4, y_values_5, y_values_6, False, False, True, 'Trial 1 End Chamber (Post-Reward)','#000080', '#56A3DC')
        
    if key == 'Trial 2':
        x_values, cutoff_dataframe_index1 = get_values(lane_1, trial_cutoffs[0])
        y_values_1 = get_y_values (lane_1, trial_cutoffs [0])
        y_values_2 = get_y_values (lane_2, trial_cutoffs[1])
        y_values_3 = get_y_values (lane_3, trial_cutoffs[2])
        y_values_4 = get_y_values (lane_4, trial_cutoffs[3])
        y_values_5 = get_y_values (lane_5, trial_cutoffs[4])
        y_values_6 = get_y_values (lane_6, trial_cutoffs[5])
        
        
        plot (fig1, ax1[0,1], control_lanes, control_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_1, y_values_2, y_values_3, True, False, False, 'Trial 2 Runway', '#2f4f4f', '#778899')
        plot (fig1, ax1[0,1], experimental_lanes, experimental_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_4, y_values_5, y_values_6, True, False, False, 'Trial 2 Runway',  '#000080', '#56A3DC')
        plot (fig1, ax1[1,1], control_lanes, control_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_1, y_values_2, y_values_3, False, True, False, 'Trial 2 End Chamber (Pre-Reward)',  '#2f4f4f', '#778899')
        plot (fig1, ax1[1,1], experimental_lanes, experimental_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_4, y_values_5, y_values_6, False, True, False, 'Trial 2 End Chamber (Pre-Reward)', '#000080', '#56A3DC')
        plot (fig1, ax1[2,1], control_lanes, control_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_1, y_values_2, y_values_3, False, False, True, 'Trial 2 End Chamber (Post-Reward)',  '#2f4f4f', '#778899')
        plot (fig1, ax1[2,1], experimental_lanes, experimental_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_4, y_values_5, y_values_6, False, False, True, 'Trial 2 End Chamber (Post-Reward)', '#000080', '#56A3DC')
    
    if key == 'Trial 3': 
        x_values, cutoff_dataframe_index1 = get_values(lane_1, trial_cutoffs[0])
        y_values_1 = get_y_values (lane_1, trial_cutoffs [0])
        y_values_2 = get_y_values (lane_2, trial_cutoffs[1])
        y_values_3 = get_y_values (lane_3, trial_cutoffs[2])
        y_values_4 = get_y_values (lane_4, trial_cutoffs[3])
        y_values_5 = get_y_values (lane_5, trial_cutoffs[4])
        y_values_6 = get_y_values (lane_6, trial_cutoffs[5])
        
        plot (fig1, ax1[0,2], control_lanes, control_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_1, y_values_2, y_values_3, True, False, False, 'Trial 3 Runway', '#2f4f4f', '#778899')
        plot (fig1, ax1[0,2], experimental_lanes, experimental_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_4, y_values_5, y_values_6, True, False, False, 'Trial 3 Runway',  '#000080', '#56A3DC')
        plot (fig1, ax1[1,2], control_lanes, control_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_1, y_values_2, y_values_3, False, True, False, 'Trial 3 End Chamber (Pre-Reward)',  '#2f4f4f', '#778899')
        plot (fig1, ax1[1,2], experimental_lanes, experimental_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_4, y_values_5, y_values_6, False, True, False, 'Trial 3 End Chamber (Pre-Reward)','#000080', '#56A3DC')
        plot (fig1, ax1[2,2], control_lanes, control_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_1, y_values_2, y_values_3, False, False, True, 'Trial 3 End Chamber (Post-Reward)', '#2f4f4f', '#778899')
        plot (fig1, ax1[2,2], experimental_lanes, experimental_cutoffs, trial_cutoffs, cutoff_dataframe_index1, x_values, y_values_4, y_values_5, y_values_6, False, False, True, 'Trial 3 End Chamber (Post-Reward)','#000080', '#56A3DC')
        

plt.show()

