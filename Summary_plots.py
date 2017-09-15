# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 13:56:42 2017

@author: Tapster
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 11:15:24 2017

@author: Tapster
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:17:16 2017

@author: sophiasong
"""

##Plots comparison of experimental and control groups (T1, T2, and T3) in the runway, before alcohol administration in the end chamber, and in the end chamber. 

import csv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import matplotlib
import os
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

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
    grab_ydata_mean_p_stdev = None
    grab_ydata_mean_m_stdev = None
    
    header_read = False
    condition_label = None

    while True:
        line = csv_reader.next()
        
        #currently we don't care about the raw data
        if "% raw data" in line:
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
                x_key = '{} xdata'.format(current_group[0])
                # convert read in "string version of line into actual numbers"
                # The JAABA datafiles always have a lingering ',' after each data group
                # this results in a ' ' character that float() cannot convert. 
                # (So remember to strip the last element from the list)
                xdata = {x_key:map(float, line[:-1])}
                xdata_df = pd.DataFrame(xdata)
                df_list.append(xdata_df)
                grab_xdata = False
                
            if grab_ydata_mean_p_stdev:
                line = filter(None, line)
                ypstd_key = '{} mean + std'.format(current_group[0])
                ydata_mean_p_std = {ypstd_key:map(float, line[:-1])}      
                ydata_mean_p_std_df = pd.DataFrame(ydata_mean_p_std)
                df_list.append(ydata_mean_p_std_df)
                grab_ydata_mean_p_stdev = False
                
            if grab_ydata_mean_m_stdev:
                line = filter(None, line)
                ymstd_key ='{} mean - std'.format(current_group[0])
                ydata_mean_m_std = {ymstd_key:map(float, line[:-1])}
                ydata_mean_m_std_df = pd.DataFrame(ydata_mean_m_std)
                df_list.append(ydata_mean_m_std_df)
                grab_ydata_mean_m_stdev = False
            
            
            if '% xdata' in line:
                grab_xdata = True
            if '% ydata' in line and 'mean + std dev' in line:
                grab_ydata_mean_p_stdev = True
            if '% ydata' in line and 'mean - std dev' in line:
                grab_ydata_mean_m_stdev = True

        line_count += 1
    
    #output_df = pd.concat([xdata_1_df, ydata_1_mean_pstd_df, ydata_1_mean_mstd_df, xdata_2_df, ydata_2_mean_pstd_df, ydata_2_mean_mstd_df],axis=1)
    output_df = pd.concat(df_list, axis=1)    
    return output_df, condition_label
    
#load our frame cutoffs from file
def load_frame_cutoffs(path):
    with open (path, 'rb') as f:
        csv_reader = csv.reader (f, delimiter = ',')
        dictionary = {row[0] : map(int,row[1:7]) for indx, row in enumerate(csv_reader) if indx != 0}
    return dictionary
    
    
 
root = tk.Tk()
path_to_T1 = filedialog.askopenfilename(parent=root, title="Select CSV for T1") 
path_to_T2 = filedialog.askopenfilename(parent=root, title="Select CSV for T2") 
path_to_T3 = filedialog.askopenfilename(parent=root, title="Select CSV for T3")     

list_of_paths_to_load = [path_to_T1, path_to_T2, path_to_T3]


path_to_cutoff_csv = 'C:\Users\Tapster\Desktop\Sophia\Cutoff Frames_Dictionary.csv'

cutoff_dictionary = load_frame_cutoffs(path_to_cutoff_csv)    

#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM1_Trial1_Lanes1-6.csv'
#path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM1_Trial2_Lanes1-6.csv'
#path_to_T3 ='E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM1_Trial3_Lanes1-6.csv'
 

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

def plot(fig, ax, group, max_cutoff, loc1, loc2, key, color, color_alcohol):
    x_data = pd.DataFrame()
    x_data_alcohol = pd.DataFrame()
    y_data = pd.DataFrame()
    y_data_alcohol = pd.DataFrame()

    for indx, contrast in enumerate(group):
        #print "indx: %d, contrast: %s" % (indx, contrast)

        cutoff_frame = trial_cutoffs[indx]
        index = contrast.rstrip(' xdatal')
        mean_plus_std_index = index + ' mean + std'
        mean_min_std_index = index + ' mean - std'

        #Load in the x data for our given contrast (contrasts can be lanes, IAA, EtOH, or any other condition)
        xdata = trial_data[contrast]

        #load in the classifier mean and stdev
        classifier_mean = (trial_data[mean_plus_std_index] + trial_data[mean_min_std_index])/2
        classifier_stdev = (trial_data[mean_plus_std_index] - classifier_mean)
            
        contrast_dict[index] = (xdata, classifier_mean, classifier_stdev)
            
        #============ Code to apply frame cutoffs per lane =================
        #Need to change the FPS to the actual value for that trial if possible
        FPS = 15.
        #convert cutoff time to minutes
        cutoff_time = (cutoff_frame/FPS)/60.
        max_cutoff_time = (max(trial_cutoffs)/FPS)/60.

        cutoff_dataframe_index = trial_data[(trial_data[contrast] < cutoff_time)].last_valid_index()
        
        x_index = xdata
        y_index = classifier_mean
        
        x_index_alcohol = xdata[max_cutoff+1:-1]
        y_index_alcohol = classifier_mean[max_cutoff+1:-1]
        
        x_data [indx] = x_index
        y_data [indx] = y_index
        x_data_alcohol [indx] = x_index_alcohol
        y_data_alcohol [indx] = y_index_alcohol 

        y_means = y_data.mean(axis=1)
        y_means_alcohol = y_data_alcohol.mean(axis = 1)
        y_sem = y_data.sem(axis=1)
        y_sem_alcohol = y_data_alcohol.sem(axis=1)

    ax.set_title(key, fontsize = 10)
    ax.plot(x_data [0], y_means, color=color)
    ax.plot(x_data_alcohol[0], y_means_alcohol, color=color_alcohol)

    ax.fill_between(x_data[indx], y_means + y_sem, y_means-y_sem, color=color, alpha=0.25,  linewidth=0.0)
    ax.fill_between(x_data_alcohol[indx], y_means_alcohol + y_sem_alcohol, y_means_alcohol-y_sem_alcohol, color = color_alcohol, alpha = 0.25, linewidth = 0.0)
    ax.set_ylim([0,100])
    ax.set_xlim([0,15.5])
    ax.set_xlabel ('Time (min)', fontsize = 7, x = 0.95)
    ax.set_ylabel (classifier + ' +/- Std', fontsize = 7.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize = 7)
    
    fig1.suptitle (classifier + ' Behavior for ' + ' '.join(split_path).rstrip ('Trial3'), fontsize = 10, x = 0.16)
    fig1.tight_layout()
    
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    
    

#Saves the figures to the name "classifier date day PM# trial# group lane#" 
    figure_title = ' '.join(split_path).rstrip ('Trial3')
       
#fig.savefig('C:\Users\Tapster\Desktop\Sophia\Summary Graphs\ ' + classifier + ' ' + figure_title + '.jpg')
#    
    #closes the figure after the images are saved 
    
    plt.close()
    
fig1, ax1 = plt.subplots(3)
fig1.set_figwidth(14)
fig1.set_figheight(8)        
fig1.patch.set_facecolor('white')  



#%%
index = 0
for key in sorted(results.keys()):
    if 'cutoffs' in key or 'classifier' in key:
        pass
    else:
        trial_data = results[key]
        #print trial_data
        trial_cutoffs = results[key + ' cutoffs']
        
        classifier = results['classifier']
        
        column_names = trial_data.columns
        
        contrasts = sorted([column_name for column_name in column_names if 'xdata' in column_name])
        #print contrasts
        num_contrasts = len(contrasts)
        
        contrast_dict = {}
        
        lanes = contrasts [0:6]
        control_lanes = contrasts[0:3]     
        control_cutoffs = trial_cutoffs[0:3]
        experimental_lanes = contrasts [3:6]
        experimental_cutoffs = trial_cutoffs[3:6]
        max_cutoff = max(trial_cutoffs)
        #print max_cutoff

        #print "key: %s, index: %d" % (key, index)
        ax11 = ax1[index]

        index += 1

    if key == "Trial 1":
         plot(fig1, ax11, control_lanes, max_cutoff, 0, 2 , ' Trial 1', '#b2b2b2', '#778899')       
         plot(fig1, ax11, experimental_lanes, max_cutoff, 0,2, 'Trial 1', '#56A3DC','#4169e1')
                     
    if key == "Trial 2": 
        plot(fig1, ax11, lanes, max_cutoff, 1,2, ' Trial 2', '#b2b2b2', '#778899')
        plot(fig1, ax11, experimental_lanes, max_cutoff, 1,2, ' Trial 2', '#56A3DC', '#4169e1')
  
    if key == "Trial 3":

        plot(fig1, ax11, control_lanes, max_cutoff, 2,2, ' Trial 3', '#b2b2b2', '#778899')
        plot(fig1, ax11, experimental_lanes, max_cutoff, 2,2, ' Trial 3', '#56A3DC', '#4169e1')    
        
plt.show()