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

def plot(fig, ax, group, cutoffs, max_cutoffs, classifier, title, lane_title, color, runway_marker_color, baic_marker_color, alcohol_marker_color): 
    fig.patch.set_facecolor('white')
    
    for indx, contrast in enumerate(group):
        #cutoff_frame = trial_cutoffs[indx]
        
        #print "indx: %d, contrast: %s" % (indx, contrast)
        index = contrast.rstrip(' binary')
    
        #print contrast
#        
        #Load in the x data for our given contrast (contrasts can be lanes, IAA, EtOH, or any other condition)
        xdata = trial_data[index]
        #print xdata

        ydata = trial_data[contrast]
        #print ydata     
        #contrast_dict[index] = (xdata, classifier_mean, classifier_stdev)
        
        #============ Code to apply frame cutoffs per lane =================
        #Need to change the FPS to the actual value for that trial if possible
        FPS = 15.
        #convert cutoff time to minutes
        cutoff_time = (cutoffs/FPS)/60.
        #print cutoff_time
       
        #max_cutoff_time = (max(trial_cutoffs)/FPS)/60.
        #print cutoff_time_list[indx]
        cutoff_dataframe_index = trial_data[(trial_data[contrast] < cutoff_time)].last_valid_index()
        #print cutoff_dataframe_index
        
        max_cutoff_dataframe_index = max_cutoff 
        max_cutoff_time = (max_cutoff/FPS)/60.
        end_time = float(xdata.iget(-1))
        
        x = xdata[0:-1]
        y = ydata[0:-1]
        
        #y_runway = y_data[0:cutoff_frame]
        #y_baic = y_data[cutoff_frame+1 : max_cutoff]
        #y_alcohol = y_data[max_cutoff:-1]
  
        
        
        ax.plot (x, y, color = color, linewidth = 0.3)
       
        ax.fill_between(x, y, color = color, alpha = 0.5)

        #ax.scatter(xdata[cutoffs], 0, marker = '^', s = 300, color = runway_marker_color)

        #ax.scatter(xdata[max_cutoff], 0, marker = '^', s = 300, color = alcohol_marker_color)
 
    #runway_patch = mpatches.Patch(color=runway_marker_color, label='Runway')
    
    #alcohol_patch = mpatches.Patch(color=alcohol_marker_color, label='Alcohol/Odor Administration')
    
#==============================================================================
# For making large raster plot   
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.scatter(xdata[cutoffs], 1.1, marker = '.', s = 100, color = runway_marker_color)
        ax.plot([0, cutoff_time], [1.1, 1.1], linewidth = 2, color = runway_marker_color, alpha = 0.7)
        ax.plot([cutoff_time, max_cutoff_time], [1.1,1.1], linewidth = 2, color = baic_marker_color, alpha = 0.7)
        ax.plot([max_cutoff_time, end_time], [1.1, 1.1], linewidth = 2,  color = alcohol_marker_color, alpha = 0.7)
   
        #ax.plot([end_time, end_time], [1.05, 1.3], linewidth = 2, color = 'black', alpha = 0.5)
        #ax.scatter(xdata[max_cutoff], 1.1, marker = '.', s = 100, color = alcohol_marker_color)
        ax.text(6.5, 1.4, title)
    ax.set_title(lane_title, rotation='horizontal', fontsize = 10,x=-0.2, y = 0.3)

#==============================================================================

        
    #ax.set_title(title, fontsize = 10)
    
    
    ax.set_ylim([0,1.2])
    ax.set_xlabel ('Time (min)', fontsize=8, x = 0.95)
    ax.set_ylabel (classifier, fontsize = 8)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 1))
    
    figure_title = ' '.join(split_path).rstrip ('Trial3')
    fig1.suptitle ('Plot of ' + figure_title + classifier, y = .95)
    
    
#fig1.suptitle ('Plot of ' + figure_title + 'Trial 1 '+ classifier, x = 0.136, y = 1)



#==============================================================================
# Plots individual lanes, with T1, T2, and T3 on separate pages
#     fig1.suptitle ('Plot of ' + figure_title + 'Trial 1 '+ classifier, x = 0.136, y = 1)
#     fig1.set_figwidth(14)
#     fig1.set_figheight(6)
#     fig1.patch.set_facecolor('white')
#     
#     fig2.suptitle ('Plot of ' + figure_title + 'Trial 2 '+ classifier, x = 0.136, y = 1)
#     fig2.set_figwidth(14)
#     fig2.set_figheight(6)
#     fig2.patch.set_facecolor('white')
#     
#     fig3.suptitle ('Plot of ' + figure_title + 'Trial 3 '+ classifier, x = 0.136, y = 1)
#     fig3.set_figwidth(14)
#     fig3.set_figheight(6)
#     fig3.patch.set_facecolor('white')
#     
# fig1, ax1 = plt.subplots(6) 
# 
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# 
# fig2, ax2 = plt.subplots(6)
# 
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# 
# 
# fig3, ax3 = plt.subplots(6)
# 
# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# 
fig1.tight_layout()
# fig2.tight_layout()
# fig3.tight_layout()
#==============================================================================

fig1, ax1 = plt.subplots (6, 3, figsize = (18,18), sharex = True, sharey = True)

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

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
        
        lane_1 = contrasts [0:1] 
        lane_2 = contrasts[1:2]
        lane_3 = contrasts [2:3]
        lane_4 = contrasts [3:4]
        lane_5 = contrasts [4:5]
        lane_6 = contrasts [5:6]
        control_cutoffs = trial_cutoffs[0:3]
        experimental_lanes = contrasts [3:6]
        experimental_cutoffs = trial_cutoffs[3:6]
        max_cutoff = max(trial_cutoffs)
        #print max_cutoff

        #print "key: %s, index: %d" % (key, index)

    if key == 'Trial 1':
        plot (fig1, ax1[0, 0], lane_1, trial_cutoffs[0], max_cutoff, classifier, 'Trial 1', 'Lane 1', '#b2b2b2', '#bebebe', '#696969', '#2f4f4f')
        plot (fig1, ax1[1, 0], lane_2, trial_cutoffs[1], max_cutoff, classifier,'', 'Lane 2', '#b2b2b2','#bebebe', '#696969', '#2f4f4f')
        plot (fig1, ax1[2, 0], lane_3, trial_cutoffs[2], max_cutoff, classifier,'', 'Lane 3', '#b2b2b2','#bebebe', '#696969', '#2f4f4f')
        plot (fig1, ax1[3, 0], lane_4, trial_cutoffs[3], max_cutoff, classifier,'', 'Lane 4', '#56A3DC', '#b0c4de', '#4682b4', '#000080')
        plot (fig1, ax1[4,0], lane_5, trial_cutoffs[4], max_cutoff, classifier,'', 'Lane 5', '#56A3DC','#b0c4de', '#4682b4', '#000080')
        plot (fig1, ax1[5,0], lane_6, trial_cutoffs[5], max_cutoff, classifier,'', 'Lane 6', '#56A3DC', '#b0c4de', '#4682b4', '#000080')
    if key == 'Trial 2':
        plot (fig1, ax1[0, 1], lane_1, trial_cutoffs[0], max_cutoff, classifier,  'Trial 2', '', '#b2b2b2', '#bebebe', '#696969', '#2f4f4f')
        plot (fig1, ax1[1, 1], lane_2, trial_cutoffs[1], max_cutoff, classifier,'', '', '#b2b2b2', '#bebebe', '#696969', '#2f4f4f')
        plot (fig1, ax1[2, 1], lane_3, trial_cutoffs[2], max_cutoff, classifier,'', '', '#b2b2b2', '#bebebe', '#696969', '#2f4f4f')
        plot (fig1, ax1[3, 1], lane_4, trial_cutoffs[3], max_cutoff, classifier,'', '', '#56A3DC', '#b0c4de', '#4682b4', '#000080')
        plot (fig1, ax1[4,1], lane_5, trial_cutoffs[4], max_cutoff, classifier,'', '','#56A3DC', '#b0c4de', '#4682b4', '#000080')
        plot (fig1, ax1[5,1], lane_6, trial_cutoffs[5], max_cutoff, classifier,'', '','#56A3DC', '#b0c4de', '#4682b4', '#000080')
    if key == 'Trial 3':
        plot (fig1, ax1[0, 2], lane_1, trial_cutoffs[0], max_cutoff, classifier, 'Trial 3', '', '#b2b2b2', '#bebebe', '#696969', '#2f4f4f')
        plot (fig1, ax1[1, 2], lane_2, trial_cutoffs[1], max_cutoff, classifier,'', '','#b2b2b2', '#bebebe', '#696969', '#2f4f4f')
        plot (fig1, ax1[2, 2], lane_3, trial_cutoffs[2], max_cutoff, classifier,'','', '#b2b2b2', '#bebebe', '#696969', '#2f4f4f')
        plot (fig1, ax1[3, 2], lane_4, trial_cutoffs[3], max_cutoff, classifier,'','', '#56A3DC', '#b0c4de', '#4682b4', '#000080')
        plot (fig1, ax1[4, 2], lane_5, trial_cutoffs[4], max_cutoff, classifier,'','', '#56A3DC', '#b0c4de', '#4682b4', '#000080')
        plot (fig1, ax1[5, 2], lane_6, trial_cutoffs[5], max_cutoff, classifier,'','', '#56A3DC', '#b0c4de', '#4682b4', '#000080')
#==============================================================================
#  Plots trials on separate figures
# #    if key == 'Trial 1':
# #        plot (fig1, ax1[0], lane_1, trial_cutoffs[0], max_cutoff, classifier, 'Lane 1', '#b2b2b2', '#696969', 'black')
# #        plot (fig1, ax1[1], lane_2, trial_cutoffs[1], max_cutoff, classifier,'Lane 2', '#b2b2b2', '#696969', 'black')
# #        plot (fig1, ax1[2], lane_3, trial_cutoffs[2], max_cutoff, classifier,'Lane 3', '#b2b2b2', '#696969', 'black')
# #        plot (fig1, ax1[3], lane_4, trial_cutoffs[3], max_cutoff, classifier,'Lane 4', '#56A3DC', '#6a5acd', '#191970')
# #        plot (fig1, ax1[4], lane_5, trial_cutoffs[4], max_cutoff, classifier,'Lane 5', '#56A3DC', '#6a5acd', '#191970')
# #        plot (fig1, ax1[5], lane_6, trial_cutoffs[5], max_cutoff, classifier,'Lane 6', '#56A3DC', '#6a5acd', '#191970')
#             
# #    if key == 'Trial 2':
# #        plot (fig2, ax2[0], lane_1, trial_cutoffs[0], max_cutoff, classifier,'Lane 1', '#b2b2b2', '#696969', 'black')
# #        plot (fig2, ax2[1], lane_2, trial_cutoffs[1], max_cutoff, classifier,'Lane 2', '#b2b2b2', '#696969', 'black')
# #        plot (fig2, ax2[2], lane_3, trial_cutoffs[2], max_cutoff, classifier,'Lane 3','#b2b2b2', '#696969', 'black')
# #        plot (fig2, ax2[3], lane_4, trial_cutoffs[3], max_cutoff, classifier,'Lane 4','#56A3DC', '#6a5acd', '#191970')
# #        plot (fig2, ax2[4], lane_5, trial_cutoffs[4], max_cutoff, classifier,'Lane 5','#56A3DC', '#6a5acd', '#191970')
# #        plot (fig2, ax2[5], lane_6, trial_cutoffs[5], max_cutoff, classifier,'Lane 6','#56A3DC', '#6a5acd', '#191970')
# #    if key == 'Trial 3': 
# #        plot (fig3, ax3[0], lane_1, trial_cutoffs[0], max_cutoff, classifier,'Lane 1','#b2b2b2', '#696969', 'black')
# #        plot (fig3, ax3[1], lane_2, trial_cutoffs[1], max_cutoff, classifier,'Lane 2','#b2b2b2', '#696969', 'black')
# #        plot (fig3, ax3[2], lane_3, trial_cutoffs[2], max_cutoff, classifier,'Lane 3','#b2b2b2', '#696969', 'black')
# #        plot (fig3, ax3[3], lane_4, trial_cutoffs[3], max_cutoff, classifier,'Lane 4','#56A3DC', '#6a5acd', '#191970')
# #        plot (fig3, ax3[4], lane_5, trial_cutoffs[4], max_cutoff, classifier,'Lane 5','#56A3DC', '#6a5acd', '#191970')
# #        plot (fig3, ax3[5], lane_6, trial_cutoffs[5], max_cutoff, classifier,'Lane 6','#56A3DC', '#6a5acd', '#191970')
#==============================================================================

plt.show()

