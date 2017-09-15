# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:24:05 2017

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
from matplotlib.path import Path

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
        try:
            line = csv_reader.next()
        except StopIteration:
            print 'AT THE END OF THE CSV FILE!!!!!!!!'
            break

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
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-22_Day1_PM2_Trial1_Lanes3456.csv'
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Retreating_2015-09-22_Day1_PM2_Trial1_Lanes3456.csv'
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Pausing_2015-09-22_Day1_PM2_Trial1_Lanes3456.csv'
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Thrashing_2015-09-22_Day1_PM2_Trial1_Lanes3456.csv'
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Pacing_2015-09-22_Day1_PM2_Trial1_Lanes3456.csv'
#path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Thrashing_2015-09-23_Day2_PM2_Trial2_Lanes3456.csv'
#path_to_T3 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Thrashing_2015-09-23_Day2_PM2_Trial3_Lanes3456.csv'

#Advancing Day 1
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-22_Day1_PM1_Trial1_Lanes1-6.csv'
#path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-22_Day1_PM1_Trial1_Lanes1-6.csv'
#path_to_T3 ='E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-22_Day1_PM1_Trial1_Lanes1-6.csv'

#Advancing Day 2
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-17_Day2_PM1_Trial1_Lanes1-6.csv'
#path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-17_Day2_PM1_Trial2_Lanes1-6.csv'
#path_to_T3 ='E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-17_Day2_PM1_Trial3_Lanes1-6.csv'

#Retreating Day 1
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Retreating_2015-09-16_Day1_PM1_Trial1_Lanes1-6.csv'
#path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Retreating_2015-09-16_Day1_PM1_Trial2_Lanes1-6.csv'
#path_to_T3 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Retreating_2015-09-16_Day1_PM1_Trial3_Lanes1-6.csv'

#Retreating Day 2
#==============================================================================
# path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Retreating_2015-09-17_Day2_PM1_Trial1_Lanes1-6.csv'
# path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Retreating_2015-09-17_Day2_PM1_Trial2_Lanes1-6.csv'
# path_to_T3 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Retreating_2015-09-17_Day2_PM1_Trial3_Lanes1-6.csv'
#==============================================================================

#Pausing Day 1
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Pausing_2015-09-16_Day1_PM1_Trial1_Lanes1-6.csv'
#path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Pausing_2015-09-16_Day1_PM1_Trial2_Lanes1-6.csv'
#path_to_T3 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Pausing_2015-09-16_Day1_PM1_Trial3_Lanes1-6.csv'

#Pausing Day 2
#path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Pausing_2015-09-17_Day2_PM1_Trial1_Lanes1-6.csv'
#path_to_T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Pausing_2015-09-17_Day2_PM1_Trial2_Lanes1-6.csv'
#path_to_T3 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Pausing_2015-09-17_Day2_PM1_Trial3_Lanes1-6.csv'


#list_of_paths_to_load = [path_to_T1, path_to_T2, path_to_T3]
list_of_paths_to_load = [path_to_T1]

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

def plot(fig, ax, group, trial_cutoff, max_cutoff,  title, lane_title, *colors): 
    runway_color =  colors [0] 
    baic_color = colors [1]
    alcohol_color = colors[2] 

    #print trial_cutoff
    
    fig.patch.set_facecolor('white')
    
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
        cutoff_time = (trial_cutoff/FPS)/60.
        cutoff_dataframe_index = trial_data[(trial_data[contrast] < cutoff_time)].last_valid_index()        
        max_cutoff_dataframe_index = max_cutoff 
        max_cutoff_time = (max_cutoff/FPS)/60.
        end_time = float(xdata.iget(-1))
        
        x = xdata[0:-1]
        y = ydata[0:-1]
        
        x_runway = x[0:trial_cutoff]
        y_runway = y[0:trial_cutoff]
        x_baic = x[trial_cutoff+1:max_cutoff]
        y_baic = y[trial_cutoff+1 : max_cutoff]
        x_alcohol = x[max_cutoff:-1]
        y_alcohol = y[max_cutoff:-1]

        #ax.plot (x, y, color = color, linewidth = 0.3)
        
        ax.plot(x_runway, y_runway, color = runway_color, linewidth = 0.1, alpha = 0.5)
        ax.fill_between (x_runway, y_runway, color = runway_color, alpha = 0.5)
        ax.plot(x_baic, y_baic, color = baic_color, linewidth = 0.1, alpha = 0.5)
        ax.fill_between (x_baic, y_baic, color = baic_color, alpha = 0.5)
        ax.plot(x_alcohol, y_alcohol, color = alcohol_color, linewidth = 0.1, alpha = 0.5)
        ax.fill_between(x_alcohol, y_alcohol, color = alcohol_color, alpha = 0.5)

#
#        ax.annotate('', xy = (xdata[trial_cutoff], -0.22), xytext = (xdata[trial_cutoff], -0.38), arrowprops = dict(facecolor = baic_color, edgecolor = baic_color, headwidth = 6.8, headlength = 6.8, width = 0.001, shrink = 0.1), annotation_clip = False)    
#        ax.annotate('', xy = (xdata[max_cutoff], -0.22), xytext = (xdata[max_cutoff], -0.38), arrowprops = dict(facecolor = alcohol_color, edgecolor = alcohol_color, headwidth = 6.8, headlength = 6.8, width = 0.001, shrink = 0.1), annotation_clip = False)#       

       #ax.scatter(xdata[trial_cutoff], 0, marker = '^', s = 100, color = runway_marker_color)
#
#        ax.scatter(xdata[max_cutoff], 0, marker = '^', s = 100, color = alcohol_marker_color)
#==============================================================================
# For making large raster plot   
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
#        ax.spines['bottom'].set_visible(False)
        
        #ax.scatter(xdata[cutoffs], 1.1, marker = '.', s = 100, color = runway_marker_color)
#        ax.plot([0, cutoff_time], [1.1, 1.1], linewidth = 2, color = runway_color, alpha = 0.7)
#        ax.plot([cutoff_time, max_cutoff_time], [1.1,1.1], linewidth = 2, color = baic_color, alpha = 0.7)
#        ax.plot([max_cutoff_time, end_time], [1.1, 1.1], linewidth = 2,  color = alcohol_color, alpha = 0.7)
   
        ax.plot([end_time, end_time], [1.05, 1.3], linewidth = 2, color = 'black', alpha = 0.5)
        #ax.scatter(xdata[max_cutoff], 1.1, marker = '.', s = 100, color = alcohol_marker_color)
        ax.text(6.5, 1.4, title)
    ax.set_title(lane_title, rotation='horizontal', fontsize = 10,x=-0.2, y = 0.3)   
    ax.set_yticklabels([])
 
    ax.set_title(title, fontsize = 10)
    ax.set_ylim([0,1])

    #ax.set_ylabel (classifier, fontsize = 8)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 1))
#    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 


    ax.spines['bottom'].set_visible(True)
    ax.xaxis.set_ticks_position('bottom') 
    ax.tick_params(labelsize=7)
 
    ax.xaxis.set_ticks(np.arange(0, 20, 5))
    ax.set_xlim([0,15.5])
        #ax.set_xlabel ('Time (min)', x = 0.95)
    ax.set_ylim([-0.3, 1])
    #else: 
    #    ax.set_xticklabels([])


#==============================================================================
# Plots individual lanes, with T1, T2, and T3 on separate pages

figure_title = ' '.join(split_path).rstrip ('Trial3')

fig1, ax1 = plt.subplots(6, sharex = True) 
fig1.suptitle ('Plot of ' + figure_title)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
 
fig2, ax2 = plt.subplots(6, sharex = True)
fig2.suptitle ('Plot of ' + figure_title) 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
 
fig3, ax3 = plt.subplots(6, sharex = True)
fig3.suptitle ('Plot of ' + figure_title) 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
#
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
figure_title = ' '.join(split_path).rstrip ('Trial3')
#==============================================================================

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
#plt.savefig('C:\Users\Tapster\Google Drive\Kaun Lab\Sophia\ ' + figure_title + 'Trial 1.eps', format = 'eps')
#
#fig1, ax1 = plt.subplots (6, 3, figsize = (18,18), sharex = True, sharey = True)
#plt.subplots_adjust(left = 0.1, right = 0.5, top=0.95, bottom=.6)


#This for loop is iterating through different trials in the experiment.
index = 0
for key in sorted(results.keys()):
    if 'cutoffs' in key or 'classifier' in key:
        pass
    else:
        trial_data = results[key]

        trial_cutoffs = results[key + ' cutoffs']
        #print trial_cutoffs

        classifier = results['classifier']
        print classifier
        
        column_names = trial_data.columns
        
        contrasts = sorted([column_name for column_name in column_names if 'binary' in column_name])
    
        num_contrasts = len(contrasts)
        
        contrast_dict = {}
        
#        lane_1 = contrasts [0:1] #lane 1
#        lane_2 = contrasts[1:2] #lane 2
#        lane_3 = contrasts [2:3] #lane4
#        lane_4 = contrasts [3:4] #lane5
#        lane_5 = contrasts [4:5] #lane 6
#        lane_6 = contrasts [5:6]
       
        #gives the right index to the contrasts
        for index, value in enumerate (contrasts): 
            if "Lane 1" in value: 

                lane_1 = contrasts[index:index+1]
            if "Lane 2" in value: 

                lane_2 = contrasts[index:index+1]
            if "Lane 3" in value: 

                lane_3 = contrasts[index:index+1]
            if "Lane 4" in value: 

                lane_4 = contrasts[index:index+1]
            if "Lane 5" in value: 

                lane_5 = contrasts[index:index+1]
            if "Lane 6" in value: 

                lane_6 = contrasts[index:index+1]

        

        control_cutoffs = trial_cutoffs[0:3]
        experimental_lanes = contrasts [3:6]
        experimental_cutoffs = trial_cutoffs[3:6]
        max_cutoff = max(trial_cutoffs)
        
        #print "key: %s, index: %d" % (key, index)


#==============================================================================
#Plots all trials in one figure 
#    if key == 'Trial 1':
#        plot (fig1, ax1[0, 0], lane_1, trial_cutoffs[0], max_cutoff, 'Trial 1', '1','#cdcdc1', '#b2b2b2', '#696969')
#        plot (fig1, ax1[1, 0], lane_2, trial_cutoffs[1], max_cutoff, '', '2', '#cdcdc1', '#b2b2b2', '#696969')
#        plot (fig1, ax1[2, 0], lane_3, trial_cutoffs[2], max_cutoff, '', '3', '#cdcdc1', '#b2b2b2', '#696969')
#        plot (fig1, ax1[3, 0], lane_4, trial_cutoffs[3], max_cutoff, '', '4', '#87ceeb', '#56A3DC', '#4682b4')
#        plot (fig1, ax1[4,0], lane_5, trial_cutoffs[4], max_cutoff, '', '5','#87ceeb', '#56A3DC', '#4682b4')
#        plot (fig1, ax1[5,0], lane_6, trial_cutoffs[5], max_cutoff, '', '6', '#87ceeb', '#56A3DC', '#4682b4')
#    if key == 'Trial 2':
#        plot (fig1, ax1[0, 1], lane_1, trial_cutoffs[0], max_cutoff,   'Trial 2', '', '#cdcdc1', '#b2b2b2', '#696969')
#        plot (fig1, ax1[1, 1], lane_2, trial_cutoffs[1], max_cutoff, '', '','#cdcdc1', '#b2b2b2', '#696969')
#        plot (fig1, ax1[2, 1], lane_3, trial_cutoffs[2], max_cutoff, '', '','#cdcdc1', '#b2b2b2', '#696969')
#        plot (fig1, ax1[3, 1], lane_4, trial_cutoffs[3], max_cutoff,'', '',  '#87ceeb', '#56A3DC', '#4682b4')
#        plot (fig1, ax1[4,1], lane_5, trial_cutoffs[4], max_cutoff, '', '', '#87ceeb', '#56A3DC', '#4682b4')
#        plot (fig1, ax1[5,1], lane_6, trial_cutoffs[5], max_cutoff, '','',  '#87ceeb', '#56A3DC', '#4682b4')
#    if key == 'Trial 3':
#        plot (fig1, ax1[0, 2], lane_1, trial_cutoffs[0], max_cutoff, 'Trial 3', '', '#cdcdc1', '#b2b2b2', '#696969')
#        plot (fig1, ax1[1, 2], lane_2, trial_cutoffs[1], max_cutoff, '', '','#cdcdc1', '#b2b2b2', '#696969')
#        plot (fig1, ax1[2, 2], lane_3, trial_cutoffs[2], max_cutoff, '','', '#cdcdc1', '#b2b2b2', '#696969')
#        plot (fig1, ax1[3, 2], lane_4, trial_cutoffs[3], max_cutoff,'','','#87ceeb', '#56A3DC', '#4682b4')
#        plot (fig1, ax1[4, 2], lane_5, trial_cutoffs[4], max_cutoff, '','','#87ceeb', '#56A3DC', '#4682b4')
#        plot (fig1, ax1[5, 2], lane_6, trial_cutoffs[5], max_cutoff,'','',  '#87ceeb', '#56A3DC', '#4682b4')
#==============================================================================
#==============================================================================
#  Plots trials on separate figures
#If some lanes are missing on the csv, comment them out on the plot commands below! 

        if key == 'Trial 1':
#            plot (fig1, ax1[0], lane_1, trial_cutoffs[0], max_cutoff, 'Trial 1', '', '#cdcdc1', '#b2b2b2', '#696969')
#            plot (fig1, ax1[1], lane_2, trial_cutoffs[1], max_cutoff, '', '', '#cdcdc1', '#b2b2b2', '#696969')
            plot (fig1, ax1[2], lane_3, trial_cutoffs[2], max_cutoff, 'Trial 1', '', '#cdcdc1', '#b2b2b2', '#696969')
            plot (fig1, ax1[3], lane_4, trial_cutoffs[3], max_cutoff, '', '',  '#87ceeb', '#56A3DC', '#4682b4')
            plot (fig1, ax1[4], lane_5, trial_cutoffs[4], max_cutoff, '','', '#87ceeb', '#56A3DC',  '#4682b4')
            plot (fig1, ax1[5], lane_6, trial_cutoffs[5], max_cutoff, '', '', '#87ceeb', '#56A3DC', '#4682b4')
 
        if key == 'Trial 2':
#            plot (fig2, ax2[0], lane_1, trial_cutoffs[0], max_cutoff, 'Trial 2', '', '#cdcdc1', '#b2b2b2', '#696969')
#            plot (fig2, ax2[1], lane_2, trial_cutoffs[1], max_cutoff, '', '', '#cdcdc1', '#b2b2b2', '#696969')
            plot (fig2, ax2[2], lane_3, trial_cutoffs[2], max_cutoff, 'Trial 2','', '#cdcdc1', '#b2b2b2', '#696969')
            plot (fig2, ax2[3], lane_4, trial_cutoffs[3], max_cutoff, '','',  '#87ceeb', '#56A3DC', '#4682b4')
            plot (fig2, ax2[4], lane_5, trial_cutoffs[4], max_cutoff, '','',  '#87ceeb', '#56A3DC', '#4682b4')
            plot (fig2, ax2[5], lane_6, trial_cutoffs[5], max_cutoff, '','',  '#87ceeb', '#56A3DC', '#4682b4')
        if key == 'Trial 3': 
#            plot (fig3, ax3[0], lane_1, trial_cutoffs[0], max_cutoff, 'Trial 3','', '#cdcdc1', '#b2b2b2', '#696969')
#            plot (fig3, ax3[1], lane_2, trial_cutoffs[1], max_cutoff, '','', '#cdcdc1', '#b2b2b2', '#696969')
            plot (fig3, ax3[2], lane_3, trial_cutoffs[2], max_cutoff, 'Trial 3','', '#cdcdc1', '#b2b2b2', '#696969')
            plot (fig3, ax3[3], lane_4, trial_cutoffs[3], max_cutoff, '','',  '#87ceeb', '#56A3DC', '#4682b4')
            plot (fig3, ax3[4], lane_5, trial_cutoffs[4], max_cutoff, '','',  '#87ceeb', '#56A3DC', '#4682b4')
            plot (fig3, ax3[5], lane_6, trial_cutoffs[5], max_cutoff, '','',  '#87ceeb', '#56A3DC', '#4682b4')
#==============================================================================

plt.show()