# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:27:29 2017

@author: Tapster
"""

#Plots %classifier behavior of Lanes 1-3 and Lanes 4-6 separately of 
#Trial 1, Trial 2, and Trial 3 before end chamber, after end chamber, and after alcohol administration

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:01:57 2016

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
    
path_to_cutoff_csv = 'C:\Users\Tapster\Desktop\Sophia\Cutoff Frames_Dictionary.csv'

cutoff_dictionary = load_frame_cutoffs(path_to_cutoff_csv)    
    
#non-hard coded paths
#root = tk.Tk()
#path_to_T1 = filedialog.askopenfilename(parent=root, title="Select CSV for T1") 
#path_to_T2 = filedialog.askopenfilename(parent=root, title="Select CSV for T2") 
#path_to_T3 = filedialog.askopenfilename(parent=root, title="Select CSV for T3")     

#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-17_Day2_PM1_Trial1_Lanes1-6.csv'
#path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-17_Day2_PM1_Trial2_Lanes1-6.csv'
#path_to_T3 ='E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-17_Day2_PM1_Trial3_Lanes1-6.csv'

path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM1_Trial1_Lanes1-6.csv'
path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM1_Trial2_Lanes1-6.csv'
path_to_T3 ='E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM1_Trial3_Lanes1-6.csv'

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


#%%

def plot(fig, ax, group, trial_cutoffs, max_cutoffs, key, runway_color, normal_color, alcohol_color): 
  
    #gs = gridspec.GridSpec(3,3)

    fig.patch.set_facecolor('white')
    
    leg_patches = [] 
    
    for indx, contrast in enumerate(group):
        cutoff_frame = trial_cutoffs[indx]
        
        #print "indx: %d, contrast: %s" % (indx, contrast)
        index = contrast.rstrip(' xdatal')
        print index
        print contrast

        mean_plus_std_index = index + ' mean + std'
        mean_min_std_index = index + ' mean - std'
        
        #Load in the x data for our given contrast (contrasts can be lanes, IAA, EtOH, or any other condition)
        xdata = trial_data[contrast]
        #print xdata
        end_time = xdata.iget(-1)
        #load in the classifier mean and stdev
        classifier_mean = (trial_data[mean_plus_std_index] + trial_data[mean_min_std_index])/2
        classifier_stdev = (trial_data[mean_plus_std_index] - classifier_mean)
        
        contrast_dict[index] = (xdata, classifier_mean, classifier_stdev)
        
        #============ Code to apply frame cutoffs per lane =================
        #Need to change the FPS to the actual value for that trial if possible
        FPS = 15.
        #convert cutoff time to minutes
        cutoff_time = (cutoff_frame/FPS)/60.
        #print cutoff_time
        cutoff_time_list = [0.73, 0.795555555556, 0.722222222222, 0.864444444444, 0.895555555556, 1.52111111111, 0.742222222222, 3.88222222222, 1.98222222222, 0.354444444444, 0.542222222222, 1.28222222222, 5.17444444444, 5.15333333333, 5.10333333333, 1.80555555556, 1.29777777778, 4.3]
        #max_cutoff_time = (max(trial_cutoffs)/FPS)/60.
        #print cutoff_time_list[indx]
        
        cutoff_dataframe_index = trial_data[(trial_data[contrast] < cutoff_time)].last_valid_index()
        #print cutoff_dataframe_index
        
        max_cutoff_dataframe_index = max_cutoff 
        max_cutoff_time = (max_cutoff/FPS)/60.
        
#*********************************
        #==============Plots complete plot
        #plt.subplot(gs [loc,:])
        #normal axis (from after runway to end)
        x1 = xdata[cutoff_dataframe_index:-1]
        y1 = classifier_mean[cutoff_dataframe_index:-1]
        #runway axis
        x2 = xdata[0:cutoff_dataframe_index]
        y2 = classifier_mean[0:cutoff_dataframe_index]
        #alcohol axis (after alcohol administraation to end) 
        x3 =  xdata[max_cutoff+1:-1]
        y3 = classifier_mean[max_cutoff+1:-1]
#       y1_stdev = classifier_stdev
            
       
    
        #ax1.fill_between(x1, y1-y1_stdev, y1+y1_stdev, color=colors[indx], alpha=0.25,  linewidth=0.0)
        #separate_lines = 100
        #for index in group:
        #    ax.plot([0,cutoff_time_list[indx]], [separate_lines, separate_lines], color = runway_color, linewidth = 2)
        #    separate_lines -= 10
        
        ax.plot(x1, y1,color=normal_color)

        
      
       
        ax.plot(x2, y2, color = runway_color)
        #ax.plot(x3, y3, color = alcohol_color)
        
        ax.plot([max_cutoff_time, end_time],[99,99], color = alcohol_color , linewidth = 3)

        ax.set_ylim([0,103])
        ax.tick_params(labelsize = 5)

        #other conditions
    ax.set_title(key, fontsize = 7)
    ax.set_xlabel ('Time (min)', fontsize = 5, x = 0.95)
    ax.set_ylabel (classifier + ' +/- Std', fontsize = 6)
    ax = plt.gca()
    #ax.xaxis.set_label_coords(0.97, -0.09)
    
    
    plt.suptitle(classifier + ' Plot of ' + ' '.join(split_path).rstrip ('Trial3'), fontsize = 8, x = 0.2)
    plt.tight_layout()
    
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

#Saves the figures to the name "classifier date day PM# trial# group lane#" 
    figure_title = ' '.join(split_path).rstrip ('Trial3')
    figure_string = str(contrast.rstrip('xdata'))
    
    fig.savefig('C:\Users\Tapster\Desktop\Sophia\Summary Graphs_SeparateLanes\ ' + classifier + ' ' + figure_title + 'jpg')
    
    #closes the figure after the images are saved 
    #plt.close()
fig1, ax1 = plt.subplots(3) 

trial_means = {}

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
        
        contrasts = sorted([column_name for column_name in column_names if 'xdata' in column_name])
        #print contrasts
        num_contrasts = len(contrasts)
        
        contrast_dict = {}
        
        control_lanes = contrasts [0:3]     
        control_cutoffs = trial_cutoffs[0:3]
        experimental_lanes = contrasts [3:6]
        experimental_cutoffs = trial_cutoffs[3:6]
        max_cutoff = max(trial_cutoffs)
        #print max_cutoff

        #print "key: %s, index: %d" % (key, index)
        ax11 = ax1[index]
        index += 1
#different colors for runway, before alcohol in chamber, and after alcohol
#    if key == 'Trial 1':
#        plot (fig1, ax11, control_lanes, control_cutoffs, max_cutoff, 0,0, key, '#cdc9c9', '#b2b2b2', '#778899')
#
#        plot (fig1, ax11, experimental_lanes, experimental_cutoffs, max_cutoff, 1,2, key, '#87cefa', '#56A3DC', '#4169e1')
#            
#    if key == 'Trial 2':
#        plot (fig1, ax11, control_lanes, control_cutoffs, max_cutoff, 1,2, key, '#cdc9c9', '#b2b2b2', '#778899')
#        plot (fig1, ax11, experimental_lanes, experimental_cutoffs, max_cutoff, 1,2, key, '#87cefa', '#56A3DC', '#4169e1') 
#    if key == 'Trial 3': 
#        plot (fig1, ax11, control_lanes, control_cutoffs, max_cutoff, 2,2, key, '#cdc9c9', '#b2b2b2', '#778899')
#        plot (fig1, ax11, experimental_lanes, experimental_cutoffs, max_cutoff, 2,2, key, '#87cefa', '#56A3DC', '#4169e1')
#For when alcohol is shaded
    if key == 'Trial 1':
        plot (fig1, ax11, control_lanes, control_cutoffs, max_cutoff, key, '#b2b2b2', '#2f4f4f','#2f4f4f')

        plot (fig1, ax11, experimental_lanes, experimental_cutoffs, max_cutoff, key, '#56A3DC', '#000080', '#000080')
            
    if key == 'Trial 2':
        plot (fig1, ax11, control_lanes, control_cutoffs, max_cutoff, key, '#b2b2b2','#2f4f4f','#2f4f4f')
        plot (fig1, ax11, experimental_lanes, experimental_cutoffs, max_cutoff, key, '#56A3DC', '#000080', '#000080') 
    if key == 'Trial 3': 
        plot (fig1, ax11, control_lanes, control_cutoffs, max_cutoff, key, '#b2b2b2', '#2f4f4f','#2f4f4f')
        plot (fig1, ax11, experimental_lanes, experimental_cutoffs, max_cutoff, key, '#56A3DC', '#000080', '#000080')


plt.show()
