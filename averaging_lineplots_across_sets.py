# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 09:31:07 2017

@author: Tapster
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:39:56 2017

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
    binary_copy_check = False
    blank_line_count = 0
    #print path_name
    while blank_line_count < 3:
        line = csv_reader.next()
            
        if line[0] is "":
            blank_line_count += 1
            if blank_line_count > 2:
                print ("Blank_line_count is: " + str(blank_line_count))
        else:
            blank_line_count = 0;
            
#        if line_count is 65:
#            print ("Line is a string of, " + line[0])
#            if line[0] is "":
#                print ("Line is a string that's empty")
                
        if "end" in line:
            print ("Hitted the end line")
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
#            if "2015-09-16_Day1_PM1" in path_name: 
#                y_key_add = 'Set 1'
#            elif "2015-09-16_Day1_PM2" in path_name:
#                y_key_add = 'Set 2'
#            else: 
#                y_key_add = ''
  
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
    output_df2 = output_df

    return output_df,  condition_label
    
#load our frame cutoffs from file
def load_frame_cutoffs(path):
    with open (path, 'rb') as f:
        csv_reader = csv.reader (f, delimiter = ',')
        dictionary = {row[0] : map(int,row[1:7]) for indx, row in enumerate(csv_reader) if indx != 0}
    return dictionary
    
path_to_cutoff_csv = 'C:\Users\Tapster\Desktop\Sophia\Cutoff Frames_Dictionary.csv'

cutoff_dictionary = load_frame_cutoffs(path_to_cutoff_csv)    

def get_sum_list (array, number, bin_size): 
    array_list = []
    for i in range(0, (bin_size)): 
        array_list.append(array[number+i])
    return array_list

def myround(num, divisor):
    return num - (num%divisor)

#non-hard coded paths
#root = tk.Tk()
#path_to_T1 = filedialog.askopenfilename(parent=root, title="Select CSV for T1") 
#path_to_T2 = filedialog.askopenfilename(parent=root, title="Select CSV for T2") 
#path_to_T3 = filedialog.askopenfilename(parent=root, title="Select CSV for T3")     

#Advancing Day 1
path_to_S1T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM1_Trial1_Lanes1-6.csv'
#path_to_S1T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM1_Trial2_Lanes1-6.csv'
#path_to_S1T3 ='E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM1_Trial3_Lanes1-6.csv'
#path_to_S2T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM2_Trial1_Lanes1-6.csv'
#path_to_S2T2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM2_Trial2_Lanes1-6.csv'
#path_to_S2T3 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Advancing_2015-09-16_Day1_PM2_Trial3_Lanes1-6.csv'
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

list_of_paths_to_load = [path_to_S1T1, path_to_S1T2, path_to_S1T3, path_to_S2T1, path_to_S2T2, path_to_S2T3]

results1 = {}
results2 = {}

for indx, path in enumerate(list_of_paths_to_load):
    with open(path, 'rb') as f:

        csv_reader = csv.reader(f, delimiter=',', skipinitialspace=True)
        classifier_df, condition_label = read_JAABA_csv_block(csv_reader)
        #classifier_df2 = classifier_df1
        #classifier_df2, condition_label = read_JAABA_csv_block(csv_reader, path)
        
        classifier_label = condition_label.lstrip('% ylabel=')
        classifier_label = classifier_label.rstrip(' (%)')
        results1['classifier'] = classifier_label
                
                        #figure out the key to enter into the cutoff_dictionary
        split_path = path.split("\\") [-1].split("_") [1:-1]
        joined_path = "_".join(split_path)
        cutoffs = cutoff_dictionary[joined_path]
        
        if indx < 3: 
            trial_label = 'Trial {}'.format(indx+1)
            cutoff_key = "{} cutoffs".format(trial_label)  
 
            results1[cutoff_key] = cutoffs

            results1[trial_label] = classifier_df
        elif indx > 2:
            #indx -= 3 
            trial_label = 'Trial {}'.format(indx-2)
            cutoff_key = "{} cutoffs".format(trial_label) 
    
    
            results2[cutoff_key] = cutoffs
         
            results2[trial_label] = classifier_df

def get_values(group, set_number): 
    for indx, contrast in enumerate(group):

        #print "indx: %d, contrast: %s" % (indx, contrast)
        index = contrast.rstrip(' binary')
        
        #print contrast
        #Load in the x data for our given contrast (contrasts can be lanes, IAA, EtOH, or any other condition)
        if set_number == 1: 
            xdata = trial_data1[index]
            ydata = trial_data1[contrast]
        if set_number ==2: 
            xdata = trial_data2[index]
            ydata = trial_data2[contrast]
        
        x = xdata[0:-1]
        y = ydata[0:-1]
        
        return x, y

#This for loop is iterating through different trials in the experiment.
index = 0
#print results
trial1_y_values = []
trial2_y_values = []
trial3_y_values = []
control_lanes = []
experimental_lanes = []
control_cutoffs = []
experimental_cutoffs = []
#==============================================================================
# SET 1
#==============================================================================
for key in sorted(results1.keys()):
    #print sorted(results.keys())
    if 'cutoffs' in key or 'classifier' in key:
        pass
    else:
        
        trial_data1 = results1[key]

        trial_cutoffs1 = results1[key + ' cutoffs']
 
        control_cutoffs.append(trial_cutoffs1[0:3])
        experimental_cutoffs.append(trial_cutoffs1[3:7])
     
        classifier = results1['classifier']
        column_names1 = trial_data1.columns
        contrasts = sorted([column_name for column_name in column_names1 if 'binary' in column_name])
        #print contrasts
        #print contrasts[0:1]
        
        lane_1 = contrasts [0:1] 
        lane_2 = contrasts[1:2]
        lane_3 = contrasts [2:3]
        lane_4 = contrasts [3:4]
        lane_5 = contrasts [4:5]
        lane_6 = contrasts [5:6]
        
        if key == 'Trial 1':
            x_values_1, y_values_1 = get_values (lane_1, 1)
            x_values_2, y_values_2= get_values (lane_2, 1)
            x_values_3, y_values_3= get_values (lane_3, 1)
            x_values_4, y_values_4 = get_values (lane_4, 1)
            x_values_5, y_values_5 = get_values (lane_5, 1)
            x_values_6, y_values_6= get_values (lane_6, 1)
            set1_trial1_y_values = [y_values_1, y_values_2, y_values_3, y_values_4, y_values_5, y_values_6]
            trial1_y_values.append(set1_trial1_y_values)
        if key == 'Trial 2':
            x_values_1, y_values_1 = get_values (lane_1, 1)
            x_values_2, y_values_2 = get_values (lane_2, 1)
            x_values_3, y_values_3 = get_values (lane_3, 1)
            x_values_4, y_values_4 = get_values (lane_4, 1)
            x_values_5, y_values_5 = get_values (lane_5, 1)
            x_values_6, y_values_6 = get_values (lane_6, 1)
            set1_trial2_y_values = [y_values_1, y_values_2, y_values_3, y_values_4, y_values_5, y_values_6]
            trial2_y_values.append(set1_trial2_y_values)
        if key == 'Trial 3': 
            x_values_1, y_values_1 = get_values (lane_1, 1)
            x_values_2, y_values_2 = get_values (lane_2, 1)
            x_values_3, y_values_3 = get_values (lane_3, 1)
            x_values_4, y_values_4 = get_values (lane_4, 1)
            x_values_5, y_values_5 = get_values (lane_5, 1)
            x_values_6, y_values_6 = get_values (lane_6, 1)
            set3_trial3_y_values = [y_values_1, y_values_2, y_values_3, y_values_4, y_values_5, y_values_6]
            trial3_y_values.append(set3_trial3_y_values)
#       
#==============================================================================
# SET 2
#==============================================================================
for key in sorted (results2.keys()):

     if 'cutoffs' in key or 'classifier' in key:
        pass
     else:
        
        trial_data2 = results2[key]
        trial_cutoffs2 = results2[key + ' cutoffs']
        #print trial_cutoffs2
        control_cutoffs.append(trial_cutoffs2[0:3])
        experimental_cutoffs.append(trial_cutoffs2[3:7])
     
        column_names2 = trial_data2.columns
        contrasts = sorted([column_name for column_name in column_names2 if 'binary' in column_name])
        lane_1 = contrasts [0:1] 
        lane_2 = contrasts[1:2]
        lane_3 = contrasts [2:3]
        lane_4 = contrasts [3:4]
        lane_5 = contrasts [4:5]
        lane_6 = contrasts [5:6]
        max_cutoff = max(trial_cutoffs2) #finish runway line, start alcohol line
        min_cutoff = min(trial_cutoffs2) #start baic line
        if key == 'Trial 1':
            x_values_1, y_values_1 = get_values (lane_1, 2)
            x_values_2, y_values_2  = get_values (lane_2, 2)
            x_values_3, y_values_3 = get_values (lane_3, 2)
            x_values_4, y_values_4 = get_values (lane_4, 2)
            x_values_5, y_values_5 = get_values (lane_5, 2)
            x_values_6, y_values_6 = get_values (lane_6, 2)
            set2_trial1_y_values = [y_values_1, y_values_2, y_values_3, y_values_4, y_values_5, y_values_6]
            trial1_y_values.append(set2_trial1_y_values)
        if key == 'Trial 2':
            x_values_1, y_values_1 = get_values (lane_1, 2)
            x_values_2, y_values_2= get_values (lane_2, 2)
            x_values_3, y_values_3= get_values (lane_3, 2)
            x_values_4, y_values_4 = get_values (lane_4, 2)
            x_values_5, y_values_5 = get_values (lane_5, 2)
            x_values_6, y_values_6 = get_values (lane_6, 2)
            set2_trial2_y_values = [y_values_1, y_values_2, y_values_3, y_values_4, y_values_5, y_values_6]
            trial2_y_values.append(set2_trial2_y_values)
        if key == 'Trial 3': 
            x_values_1, y_values_1 = get_values (lane_1, 2)
            x_values_2, y_values_2= get_values (lane_2, 2)
            x_values_3, y_values_3= get_values (lane_3, 2)
            x_values_4, y_values_4 = get_values (lane_4, 2)
            x_values_5, y_values_5 = get_values (lane_5, 2)
            x_values_6, y_values_6 = get_values (lane_6, 2)
            set2_trial3_y_values = [y_values_1, y_values_2, y_values_3, y_values_4, y_values_5, y_values_6]
            trial3_y_values.append(set2_trial3_y_values)
 

max_cutoff = max(cutoffs)
print max_cutoff
max_control_cutoff = max(max(control_cutoffs))
min_control_cutoff = min(min(control_cutoffs))
max_experimental_cutoff = max(max(experimental_cutoffs))
min_experimental_cutoff = min(min(experimental_cutoffs))

control_lanes_t1 = [trial1_y_values[0][0], trial1_y_values[0][1],trial1_y_values[0][2], trial1_y_values[1][0],trial1_y_values[1][1],trial1_y_values[1][2]]
control_lanes_t1_df = pd.DataFrame(control_lanes_t1)
experimental_lanes_t1 = [trial1_y_values[0][3], trial1_y_values[0][4], trial1_y_values[0][5], trial1_y_values[1][3], trial1_y_values[1][4], trial1_y_values[1][5]]
experimental_lanes_t1_array = np.array([experimental_lanes_t1])
#==============================================================================
# 
# TRIAL 1 
#==============================================================================
t1_x =  max(len(control_lanes_t1[0]), len(control_lanes_t1[3]))
print t1_x 
t1_x_rounded = myround(t1_x, 100)
print t1_x_rounded

t1_x_rounded_range = range(0, t1_x_rounded+1)

t1_control_y = np.nanmean((trial1_y_values[0][0], trial1_y_values[0][1], trial1_y_values[0][2], trial1_y_values[1][0], trial1_y_values[1][1], trial1_y_values[1][2]), axis = 0)

t1_experimental_y = np.nanmean((trial1_y_values[0][3], trial1_y_values[0][4], trial1_y_values[0][5], trial1_y_values[1][3], trial1_y_values[1][4], trial1_y_values[1][5]), axis = 0)

#Standard error
t1_control_sem = control_lanes_t1_df.sem()
control_lanes_t1_df_psem = t1_control_sem + t1_control_y

control_lanes_t1_df_msem = t1_control_y - t1_control_sem

#Plot binned 
y_binned = []
psem_binned = []
msem_binned = []
x_binned = []
for number in range (0, t1_x_rounded+1, 10): 
    #print number
    x_binned.append(float(number)/15/60)
    #value = (sum([t1_control_y[number+1], t1_control_y[number+2], t1_control_y[number+3], t1_control_y[number+4], t1_control_y[number+5], t1_control_y[number+6], t1_control_y[number+7], t1_control_y[number+8], t1_control_y[number+9]], t1_control_y[number]))/10
    #y_binned.append(value)
    summ = get_sum_list(t1_control_y, number, 10)
    average = sum(summ)/10 
    y_binned.append(average)
    
    psem_val = get_sum_list(control_lanes_t1_df_psem, number, 10)
    psem_average = sum(psem_val)/10
    psem_binned.append(psem_average)
    
    msem_val = get_sum_list(control_lanes_t1_df_msem, number, 10)
    msem_average = sum(msem_val)/10
    msem_binned.append(msem_average)

runway_cutoff = int(max_cutoff)/10
#print runway_cutoff
baic_cutoff = int(min_cutoff)/10
#print baic_cutoff                
alcohol_end = int(t1_x)/10

x_runway_binned = x_binned[0:runway_cutoff]
y_runway_binned = y_binned[0:runway_cutoff]
x_baic_binned = x_binned[baic_cutoff:runway_cutoff]
y_baic_binned = y_binned[baic_cutoff:runway_cutoff]
x_alcohol_binned = x_binned[baic_cutoff:alcohol_end]
y_alcohol_binned = y_binned[baic_cutoff:alcohol_end]

fig, ax = plt.subplots(3)
ax[0].plot(x_runway_binned, y_runway_binned, color = 'red', alpha = 0.7)        
ax[0].plot(x_alcohol_binned, y_alcohol_binned, color = 'blue')
ax[0].plot(x_baic_binned, y_baic_binned, color = 'green')
ax[0].fill_between(x_binned, psem_binned, msem_binned, color = 'gray', alpha = 0.25)



###TRIAL 2

###TRIAL 3

#fig, ax = plt.subplots(3)
#ax[0].plot (range(0, t1_x), t1_control_y)



