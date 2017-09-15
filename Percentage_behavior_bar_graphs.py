# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 14:20:28 2017

@author: Tapster

File is used to create bar charts of the percentage of each behavior occuring in each lane (one fly)
or for the whole trial (# of flies in trial) by reading the raw data in the .csv files exported from
JAABAPlot.
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
from collections import deque
import glob

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
    
    
# Input arguments to show which kinds of bar graphs on either trial for each lane,
# or trial for controls v experimentals. or for day 1 v day 2 with controls v exp.
# or for all the above. If GenDayData is false, the given glob path will find all
# csvs in that directory and all sub-directories, then graph the given data in
# trials in the form of control v experimentals and for each lane in the given trial.
# If GenDayData is True (NOTE: You should glob an entire set. Program won't continue 
# if you don't glob two paths with the same set), then the given paths to glob will 
# get all csvs in the directory and use that data to form a control v experimental 
# bar graph of the given set. 
#def create_bar_graphs(list_to_glob, DataCheck = False):
def create_bar_graphs(list_to_glob, GenDayData = False, GenTrialData = False):
    if GenDayData is True and len(list_to_glob) != 2:
        print 'NEED 2 path''s to graph a Set of Data!!'
        sys.exit('ERROR\nERROR\nNEED 2 paths in order to graph a set of data!! Exiting.')
    elif GenDayData is True:        
        path1 = glob.glob('{}\*.csv'.format(list_to_glob[0]))
        path2 = glob.glob('{}\*.csv'.format(list_to_glob[1]))
#        print list_to_glob[0]
#        print list_to_glob[1]
        
        for path in path1:
            print 'The given path: ' + str(os.path.normpath(path))
        print path1
        list_of_paths_to_load = path1 + path2
#        print '\n' + str(path2)
    else:
        path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Set 5_2015-09-23_Day2_PM1\Advancing_2015-09-23_Day2_PM1_Trial1_Lanes13456.csv'
        path_to_T1_diff = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Set 5_2015-09-23_Day2_PM1\Retreating_2015-09-23_Day2_PM1_Trial1_Lanes13456.csv'
        path_to_T1_other = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Set 5_2015-09-23_Day2_PM1\Pausing_2015-09-23_Day2_PM1_Trial1_Lanes13456.csv'
        path_to_T1_change = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Set 5_2015-09-23_Day2_PM1\Thrashing_2015-09-23_Day2_PM1_Trial1_Lanes13456.csv'
        path_to_T1_cha = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Set 5_2015-09-23_Day2_PM1\Pacing_2015-09-23_Day2_PM1_Trial1_Lanes13456.csv'
        print os.path.basename(path_to_T1)
        print os.path.abspath(path_to_T1)
        print os.path.dirname(path_to_T1)
        list_of_paths_to_load = [path_to_T1, path_to_T1_diff, path_to_T1_other, path_to_T1_change, path_to_T1_cha]
#    list_of_paths_to_load = [path_to_T1, path_to_T1_diff]
#    results = {}
#    num_class_labels = []
    num_percent_lists = []
    num_percents_before = []
    num_percents_after = []
    behavior_list = []
    before_dict = {}
    after_dict = {}
    
    if GenDayData is True:
        set_before_dict = {}
        set_after_dict = {}
        before_dict2 = {}
        before_dict3 = {}
        before_dict_list = [before_dict, before_dict2, before_dict3]
        after_dict2 = {}
        after_dict3 = {}
        after_dict_list = [after_dict, after_dict2, after_dict3]
        
    cutoffs = load_frame_cutoffs('C:\Users\Tapster\Desktop\Sophia\Cutoff Frames_Dictionary.csv')
    print cutoffs
    prev_name = None
    prev_day_name = None
    prev_trial_name = None
    trial_incr = 0
    
    for indx, path in enumerate(list_of_paths_to_load):
        with open(path, 'rb') as f:
            # Parsing the basename of the given path for labeling the bar graph and for fetching and organizing the behavioral data for the graphs
            abspath = os.path.normpath(path)
            abspath = abspath.split(os.sep)
#            print abspath
            #Get name of the Set #
            set_list = abspath[-2]
            set_list = set_list.split('_')
            setname = set_list[0]

            #Get name of the Day #
#            print set_list
            dayname = dayname = set_list[-2]
            if prev_day_name != dayname:
                print 'The name of the given DAY has changed to: ' + str(dayname)
                prev_day_name = dayname
                if GenDayData is True:
                    set_before_dict[dayname] = {}
                    set_after_dict[dayname] = {}
                    
                    set_before_dict[dayname]['Trial1'] = {}
                    set_before_dict[dayname]['Trial2'] = {}
                    set_before_dict[dayname]['Trial3'] = {}
                    
                    set_after_dict[dayname]['Trial1'] = {}
                    set_after_dict[dayname]['Trial2'] = {}
                    set_after_dict[dayname]['Trial3'] = {}
            
            #Get name of the Trial #
            basename = abspath[-1]
            basename = basename.split('_')
#            print basename
            trialname = basename[-2]
            if prev_trial_name != trialname:
                print 'The name of the given TRIAL has changed to: ' + str(trialname)
#                sys.exit('ERROR\nERROR\nNEED 2 paths in order to graph a set of data!! Exiting.')
                prev_trial_name = trialname
#                trial_incr = int(trialname[-1]) - 1
                before_trial_dict = {}
                after_trial_dict = {}
                
            #Get name tag of the trial #
            nametag = basename[1]+'_'+basename[2]+'_'+basename[3]+'_'+basename[4]
            print nametag
            
            
            csv_reader = csv.reader(f, delimiter=',', skipinitialspace=True)
            if GenDayData is True:
                percents_before, percents_after, condition_label, group_names, behavior_name = read_JAABA_csv_block(csv_reader, GenDayData, prev_name, behavior_list, before_trial_dict, after_trial_dict, cutoffs, nametag)
                set_before_dict[dayname][trialname].update(before_trial_dict)
                set_after_dict[dayname][trialname].update(after_trial_dict)
            else:
                percents_before, percents_after, condition_label, group_names, behavior_name = read_JAABA_csv_block(csv_reader, GenDayData, prev_name, behavior_list, before_dict, after_dict, cutoffs, nametag)
#            print 'These are the proportions'+ str(percentage_list)
#            num_percent_lists.append(percentage_list)
            num_percents_before.append(percents_before)
            num_percents_after.append(percents_after)
            print 'BEHAVIOR: ' + str(behavior_list)
#            print 'WITH PROPORTIONS: ' + str(percentage_list)

            print 'WITH PROPORTIONS: ' + str(percents_before)
            print 'WITH PROPORTIONS: ' + str(percents_after)
            print '\n\nSHOWING BEHAVIORS before: ' + str(before_dict) + ' \n\n'
#            classifier_label = condition_label.lstrip('% ylabel=')
#            classifier_label = classifier_label.rstrip(' (%)')
#            num_class_labels.append(classifier_label)
#            results['classifier'] = classifier_label
#            trial_label = 'Trial {}'.format(indx+1)
#            results[trial_label] = percentage_list
            #figure out the key to enter into the cutoff_dictionary
#            split_path = path.split("\\") [-1].split("_") [1:-1]
#            joined_path = "_".join(split_path)
#            cutoffs = cutoff_dictionary[joined_path]
    
#            cutoff_key = "{} cutoffs".format(trial_label)        
#            results[cutoff_key] = cutoffs    
    print 'These are the behaviors being graphed!!!!  ' + str(behavior_list)
#    print 'These are the proportions:    '+ str(num_percent_lists)
    print group_names
    
    if GenDayData is True:
        nump_before, nump_after = set_diction_to_list(set_before_dict, set_after_dict)
        nump_before_delta, nump_after_delta = get_delta_data(nump_before, nump_after)
        if GenTrialData is True:
            delta_nump_before, delta_nump_after = get_delta_data(nump_before, nump_after)
#            delta_mean_before, delta_mean_after = average_trial_proportions(delta_nump_before, delta_nump_after, group_names, 5)
#            delta_trial_before, delta_trial_after = average_trial_group(delta_nump_before, delta_nump_after, group_names, 5)
            
            mean_nump_before, mean_nump_after = average_trial_proportions(nump_before, nump_after, group_names, 5)
            trial_nump_before, trial_nump_after = average_trial_group(nump_before, nump_after, group_names, 5)
            create_trial_bar_plots(trial_nump_before[0], trial_nump_after[0], mean_nump_before[0], mean_nump_after[0], 'Set 6 Day 1: ')
            create_trial_bar_plots(trial_nump_before[1], trial_nump_after[1], mean_nump_before[1], mean_nump_after[1], 'Set 6 Day 2: ')
    else:
        nump_before, nump_after = diction_to_list(before_dict, after_dict)
    
        ##### For creating a bar plot of all lanes in the trial before the fly enters the alcohol chamber
        n_groups = len(group_names)
        fig, ax = plt.subplots()
        x_name_pos = np.arange(n_groups)
        print x_name_pos
        width = 0.15
        opa = 0.75
        colors = ['b', 'g', 'r', 'turquoise', 'brown']
        incr_bar_width = 0
        
        for ind, behaviors in enumerate(behavior_list):
            plt.bar(x_name_pos + incr_bar_width, nump_before[ind], width, alpha=opa, color=colors[ind], label = behaviors)
            incr_bar_width += 0.15
            
        plt.ylabel('Percent of behavior occuring')
        plt.title('Set 5, Day 2, Trial 1: Before Entering Alcohol Chamber')
        plt.xticks(x_name_pos + 0.3, group_names)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        
        ##### For creating a bar plot of all lanes in the trial AFTER the fly enters the alcohol chamber
        fig2, ax2 = plt.subplots()
        incr_bar_width = 0
        
        for ind, behaviors in enumerate(behavior_list):
            plt.bar(x_name_pos + incr_bar_width, nump_after[ind], width, alpha=opa, color=colors[ind], label = behaviors)
            incr_bar_width += 0.15
            
        plt.ylabel('Percent of behavior occuring')
        plt.title('Set 5, Day 2, Trial 1: After Entering Alcohol Chamber')
        plt.xticks(x_name_pos + 0.3, group_names)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        
    
        num_behaviors = len(behavior_list)
        mean_behavior_percents_before, mean_behavior_percents_after = average_group_proportions(nump_before, nump_after, group_names,  num_behaviors)
    #    for n in range(0, len(mean_behavior_proportions)):
    #        print mean_behavior_proportions[n]
    
        for n in range(0, len(mean_behavior_percents_before)):
            print mean_behavior_percents_before[n]
            print mean_behavior_percents_after[n]
            
            
        ##### For creating bar graphs showing the averaged behaviors for controls and experimentals
        ##### BEFORE the fly enters the alcohol chamber
        n_groups = 2
        fig3, ax3 = plt.subplots()
        x_name = [0,2]
        x_name_pos3 = np.array(x_name)
        width = 0.20
        incr_bar_width = 0
        
        for ind, behaviors in enumerate(behavior_list):
            plt.bar(x_name_pos3 + incr_bar_width, mean_behavior_percents_before[ind], width, alpha=opa, color=colors[ind], label = behaviors)
            incr_bar_width += 0.2
            
        plt.ylabel('Percent of behavior occuring')
        plt.title('Set 5, Day 2, Trial 1: Before Entering Alcohol Chamber')
        plt.xticks(x_name_pos + 0.3, ('Controls', '','Experimentals'))
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        
        ##### For creating bar graphs showing the averaged behaviors for controls and experimentals
        ##### AFTER the fly enters the alcohol chamber
        fig4, ax4 = plt.subplots()
        width = 0.20
        incr_bar_width = 0
        
        for ind, behaviors in enumerate(behavior_list):
            plt.bar(x_name_pos3 + incr_bar_width, mean_behavior_percents_after[ind], width, alpha=opa, color=colors[ind], label = behaviors)
            incr_bar_width += 0.2
            
        plt.ylabel('Percent of behavior occuring')
        plt.title('Set 5, Day 2, Trial 1: After Entering Alcohol Chamber')
        plt.xticks(x_name_pos + 0.3, ('Controls', '', 'Experimentals'))
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        

def read_JAABA_csv_block(csv_reader, GenDayData, prev_name, behavior_list, before_dict, after_dict, cutoffs, nametag):
    current_group = deque(maxlen=1)
    
    line_count = 0
    cutoff_ind = 0
    
    groups = []
    parsed_groups = []
    before_proportions = []
    after_proportions = []
    group_names = []

    grab_ydata = None
    header_read = False
    condition_label = None
    raw_data_check = False
    print '\n\n\n\n'
    while True:
        try:
            line = csv_reader.next()
        except StopIteration:
            print 'Reached end of the csv, breaking out of while LOOP!!'
            break
        
        if line:      
            #Change df_type for creation of the x and y axis of the bar graphs
            if raw_data_check and line[0].startswith('% group'):
                 split_line = line[0].split()
#                 print split_line[-1]
#                 print split_line[-2]
                 group_names.append(split_line[-2] + ' ' + split_line[-1])
                 print group_names[-1]
                
            if line[0].startswith('% raw data'):
                raw_data_check = True
            
            if line[0].startswith('% title'):
                df_title = line
            # % ylabel is what's really needed. 
            if line[0].startswith('% ylabel'):
                df_ylabel = line[0][9:]
                df_ylabel = df_ylabel.rstrip(' (%)')
                if any(char.isdigit() for char in df_ylabel):
                    print 'There is a number/s in the string'
                    df_ylabel = df_ylabel[0:-1]
                if df_ylabel[0].islower():
                    df_ylabel = df_ylabel[0].upper() + df_ylabel[1:]
                    
                print(' ')
                print('df_ylabel is: ' + str(df_ylabel))
                if prev_name != df_ylabel:
                    before_dict[df_ylabel] = {}
                    after_dict[df_ylabel] = {}
                    print 'There is a new behavior being graphed!  ' + str(df_ylabel)
                    behavior_list.append(df_ylabel)
                prev_name = df_ylabel
                condition_label = df_ylabel[0]
                header_read = True
            
            if header_read and '% group' in line[0]:
                line = filter(None, line)
                groups.append(line)
                parsed_groups.append(line[0].lstrip('% group '))
                #save the group name to use when generating dataframes
                current_group.append(line[0].lstrip('% group '))
                
            if grab_ydata: 
                line = filter (None, line)
                numt_line = map(float, line[:-1])
                
                cutoff_ind = int(split_line[-1]) - 1
                cur_cutoff_list = cutoffs[nametag]
                cur_cutoff_frame = cur_cutoff_list[cutoff_ind]
                print nametag
                print cur_cutoff_list
                before_cut = numt_line[0:cur_cutoff_frame]
                after_cut = numt_line[cur_cutoff_frame:-1]
#                print 'This is the current cutoff increment ' + str(cutoff_ind)
                print group_names[-1]
                print cur_cutoff_frame
#                print before_cut
#                print after_cut
                
                b_proportion = compute_proportion(before_cut)
                a_proportion = compute_proportion(after_cut)
                before_proportions.append(b_proportion)
                after_proportions.append(a_proportion)
#                proportion = compute_proportion(numt_line)
#                proportion_list.append(proportion)
                before_dict[df_ylabel][group_names[-1]] = b_proportion
                after_dict[df_ylabel][group_names[-1]] = a_proportion
                print 'The percentage of the given BEHAVIOR BEFORE ENTERING THE ALCOHOL CHAMBER is: ' + str(before_dict[df_ylabel][group_names[-1]])
                print 'The percentage of the given BEHAVIOR BEFORE ENTERING THE ALCOHOL CHAMBER is: ' + str(after_dict[df_ylabel][group_names[-1]])
#                print 'The percentage of the given BEHAVIOR BEFORE ENTERING THE ALCOHOL CHAMBER is: ' + str(b_proportion)
#                print 'The percentage of the given BEHAVIOR AFTER ENTERING THE ALCOHOL CHAMBER is: ' + str(a_proportion)
#                y_key = '{}'.format(current_group[0])
#                print 'y_key is: ' + y_key
#                ydata = {y_key:map(float, line[:-1])}
                grab_ydata = False

            if line[0].startswith('% experiment'):
                grab_ydata = True
            
        line_count += 1
        
    return before_proportions, after_proportions, condition_label, group_names, df_title

#load our frame cutoffs from file
def load_frame_cutoffs(path):
    with open (path, 'rb') as f:
        csv_reader = csv.reader (f, delimiter = ',')
        dictionary = {row[0] : map(int,row[1:7]) for indx, row in enumerate(csv_reader) if indx != 0}
    return dictionary

def compute_proportion(behavior_list):
    behave_occur = 0
    for n in range(0, len(behavior_list)):
        if behavior_list[n] == 1.0:
            behave_occur += 1
#    print 'Number of behavior occuring is: ' + str(behave_occur)
#    print 'Length of frames is: ' + str(len(behavior_list))
#    print behave_occur/float(len(behavior_list))
    return behave_occur/float(len(behavior_list)) * 100

def average_group_proportions(percent_lists_before, percent_lists_after, group_names, num_behaviors):
    print '\n\n'
    print 'Showing percent lists of before:    ' + str(percent_lists_before)
    print '\nNow showing percent lists of after:     ' + str(percent_lists_after)
    add_bcontrol_behaviors = []
    add_bexp_behaviors = []
    add_acontrol_behaviors = []
    add_aexp_behaviors = []
    for ind in range(0,num_behaviors):
        add_bcontrol_behaviors.append(0)
        add_bexp_behaviors.append(0)
        add_acontrol_behaviors.append(0)
        add_aexp_behaviors.append(0)
    print add_bcontrol_behaviors
    print add_bexp_behaviors
    print add_acontrol_behaviors
    print add_aexp_behaviors
    num_controls = 0
    num_experimentals = 0

    for index, name in enumerate(group_names):
        if name == 'Lane 1':
            print 'There is a lane ONE amount of behaviors'
            print len(add_bcontrol_behaviors) 
            num_controls += 1
            for n in range(0, len(add_bcontrol_behaviors)):
                add_bcontrol_behaviors[n] += percent_lists_before[n][index]
                add_acontrol_behaviors[n] += percent_lists_after[n][index]
            print n
        if name == 'Lane 2':
            print 'There is a lane TWO amount of behaviors'
            num_controls += 1
            for n in range(0, len(add_bcontrol_behaviors)):
                add_bcontrol_behaviors[n] += percent_lists_before[n][index]
                add_acontrol_behaviors[n] += percent_lists_after[n][index]
            print n
        if name == 'Lane 3':
            print 'There is a lane THREE amount of behaviors'
            num_controls += 1
            for n in range(0, len(add_bcontrol_behaviors)):
                add_bcontrol_behaviors[n] += percent_lists_before[n][index]
                add_acontrol_behaviors[n] += percent_lists_after[n][index]
            print n
        if name == 'Lane 4':
            print 'There is a lane FOUR amount of behaviors'
            num_experimentals += 1
            for n in range(0, len(add_bcontrol_behaviors)):
                add_bexp_behaviors[n] += percent_lists_before[n][index]
                add_aexp_behaviors[n] += percent_lists_after[n][index]
        if name == 'Lane 5':
            print 'There is a lane FIVE amount of behaviors'
            num_experimentals += 1
            for n in range(0, len(add_bcontrol_behaviors)):
                add_bexp_behaviors[n] += percent_lists_before[n][index]
                add_aexp_behaviors[n] += percent_lists_after[n][index]
        if name == 'Lane 6':
            print 'There is a lane SIX amount of behaviors'
            num_experimentals += 1
            for n in range(0, len(add_bcontrol_behaviors)):
                add_bexp_behaviors[n] += percent_lists_before[n][index]
                add_aexp_behaviors[n] += percent_lists_after[n][index]
        print 'The ' + str(index) + ' bout for adding behaviors to before control''s list:   ' + str(add_bcontrol_behaviors)
        print 'The ' + str(index) + ' bout for adding behaviors to before exp''s list:   ' + str(add_bexp_behaviors)
        print 'The ' + str(index) + ' bout for adding behaviors to after control''s list:   ' + str(add_acontrol_behaviors)
        print 'The ' + str(index) + ' bout for adding behaviors to after exp''s list:   ' + str(add_aexp_behaviors)
                
#    mean_control_behaviors = []
#    mean_exp_behaviors = []
    mean_behaviors_before = []
    mean_behaviors_after = []
    print 'Num controls ' + str(num_controls)
    print 'Num Exp ' + str(num_experimentals)
    for m in range(0, num_behaviors):
        print m
        new_behaviors_before = []
        new_behaviors_after = []
        new_behaviors_before.append(add_bcontrol_behaviors[m]/float(num_controls))
        new_behaviors_before.append(add_bexp_behaviors[m]/float(num_experimentals))
        mean_behaviors_before.append(new_behaviors_before)
        
        new_behaviors_after.append(add_acontrol_behaviors[m]/float(num_controls))
        new_behaviors_after.append(add_aexp_behaviors[m]/float(num_experimentals))
        mean_behaviors_after.append(new_behaviors_after)
#        mean_behaviors[m].append(add_control_behaviors[m]/float(num_controls))
#        mean_behaviors[m].append(add_exp_behaviors[m]/float(num_controls))
#        mean_control_behaviors.append(add_control_behaviors[m]/float(num_controls))
#        mean_exp_behaviors.append(add_exp_behaviors[m]/float(num_experimentals))
#    for n in range(0, len(mean_control_behaviors))
    
    return mean_behaviors_before, mean_behaviors_after

def average_trial_proportions(percent_lists_before, percent_lists_after, group_names, num_behaviors):
    print '\n\n'
    print 'Showing percent lists of before:    ' + str(percent_lists_before)
    print '\nNow showing percent lists of after:     ' + str(percent_lists_after)
    add_bcontrol_behaviors1 = [[], []]
    add_bexp_behaviors1 = [[], []]
    add_acontrol_behaviors1 = [[], []]
    add_aexp_behaviors1 = [[], []]
    
    add_bcontrol_behaviors2 = [[], []]
    add_bexp_behaviors2 = [[], []]
    add_acontrol_behaviors2 = [[], []]
    add_aexp_behaviors2 = [[], []]
    
    add_bcontrol_behaviors3 = [[], []]
    add_bexp_behaviors3 = [[], []]
    add_acontrol_behaviors3 = [[], []]
    add_aexp_behaviors3 = [[], []]
    for day in range(0, len(add_bcontrol_behaviors1)):
        for ind in range(0,num_behaviors):
            add_bcontrol_behaviors1[day].append(0)
            add_bexp_behaviors1[day].append(0)
            add_acontrol_behaviors1[day].append(0)
            add_aexp_behaviors1[day].append(0)
            
            add_bcontrol_behaviors2[day].append(0)
            add_bexp_behaviors2[day].append(0)
            add_acontrol_behaviors2[day].append(0)
            add_aexp_behaviors2[day].append(0)
            
            add_bcontrol_behaviors3[day].append(0)
            add_bexp_behaviors3[day].append(0)
            add_acontrol_behaviors3[day].append(0)
            add_aexp_behaviors3[day].append(0)

    print add_bcontrol_behaviors1
#    print add_bexp_behaviors
#    print add_acontrol_behaviors
#    print add_aexp_behaviors
    num_controls = 2
    num_experimentals = 3

    trial_names = ['Trial 1', 'Trial 2', 'Trial 3']
    day_names = ['Day 1', 'Day 2']
    for i, day_names in enumerate(day_names):
        for ind, trial in enumerate(trial_names):
            for index, name in enumerate(group_names):
                print 'The lane index is: ' + str(index)
                if name == 'Lane 1' or name == 'Lane 2' or name == 'Lane 3':
                    print 'There is a control lane amount of behaviors'
                    for n in range(0, len(add_bcontrol_behaviors1[i])):
                        if ind == 0:
                            add_bcontrol_behaviors1[i][n] += percent_lists_before[i][ind][n][index]
                            add_acontrol_behaviors1[i][n] += percent_lists_after[i][ind][n][index]
                        elif ind == 1:
                            add_bcontrol_behaviors2[i][n] += percent_lists_before[i][ind][n][index]
                            add_acontrol_behaviors2[i][n] += percent_lists_after[i][ind][n][index]
                        else:
                            add_bcontrol_behaviors3[i][n] += percent_lists_before[i][ind][n][index]
                            add_acontrol_behaviors3[i][n] += percent_lists_after[i][ind][n][index]
                            
                if name == 'Lane 4' or name == 'Lane 5' or name == 'Lane 6':
                    print 'There is a exp lane amount of behaviors'
                    for n in range(0, len(add_bcontrol_behaviors1[i])):
                        if ind == 0:
                            add_bexp_behaviors1[i][n] += percent_lists_before[i][ind][n][index]
                            add_aexp_behaviors1[i][n] += percent_lists_after[i][ind][n][index]
                        elif ind == 1:
                            add_bexp_behaviors2[i][n] += percent_lists_before[i][ind][n][index]
                            add_aexp_behaviors2[i][n] += percent_lists_after[i][ind][n][index]
                        else:
                            add_bexp_behaviors3[i][n] += percent_lists_before[i][ind][n][index]
                            add_aexp_behaviors3[i][n] += percent_lists_after[i][ind][n][index]
                print 'The ' + str(index) + ' bout for adding behaviors to before control''s list:   ' + str(add_bcontrol_behaviors1)
#                print 'The ' + str(index) + ' bout for adding behaviors to before exp''s list:   ' + str(add_bexp_behaviors)
#                print 'The ' + str(index) + ' bout for adding behaviors to after control''s list:   ' + str(add_acontrol_behaviors)
#                print 'The ' + str(index) + ' bout for adding behaviors to after exp''s list:   ' + str(add_aexp_behaviors)
                
#    mean_control_behaviors = []
#    mean_exp_behaviors = []
    mean_behaviors_before = [[], []]
    mean_behaviors_after = [[], []]
    print 'Num controls ' + str(num_controls)
    print 'Num Exp ' + str(num_experimentals)
    for j in range(0, len(mean_behaviors_before)):
        for m in range(0, num_behaviors):
            print m
            new_behaviors_before = []
            new_behaviors_after = []
            new_behaviors_before.append(add_bcontrol_behaviors1[j][m]/float(num_controls))
            new_behaviors_before.append(add_bexp_behaviors1[j][m]/float(num_experimentals))
            new_behaviors_before.append(add_bcontrol_behaviors2[j][m]/float(num_controls))
            new_behaviors_before.append(add_bexp_behaviors2[j][m]/float(num_experimentals))
            new_behaviors_before.append(add_bcontrol_behaviors3[j][m]/float(num_controls))
            new_behaviors_before.append(add_bexp_behaviors3[j][m]/float(num_experimentals))
            mean_behaviors_before[j].append(new_behaviors_before)
            
            new_behaviors_after.append(add_acontrol_behaviors1[j][m]/float(num_controls))
            new_behaviors_after.append(add_aexp_behaviors1[j][m]/float(num_experimentals))
            new_behaviors_after.append(add_acontrol_behaviors2[j][m]/float(num_controls))
            new_behaviors_after.append(add_aexp_behaviors2[j][m]/float(num_experimentals))
            new_behaviors_after.append(add_acontrol_behaviors3[j][m]/float(num_controls))
            new_behaviors_after.append(add_aexp_behaviors3[j][m]/float(num_experimentals))
            mean_behaviors_after[j].append(new_behaviors_after)
    #        mean_behaviors[m].append(add_control_behaviors[m]/float(num_controls))
    #        mean_behaviors[m].append(add_exp_behaviors[m]/float(num_controls))
    #        mean_control_behaviors.append(add_control_behaviors[m]/float(num_controls))
    #        mean_exp_behaviors.append(add_exp_behaviors[m]/float(num_experimentals))
#    for n in range(0, len(mean_control_behaviors))

#    b = percent_lists_before
#    a = percent_lists_after
#    print '\n'
#    print len(b)
#    print len(b[0])
#    print len(b[0][0])
#    print len(b[0][0][0])
#    proportion = (b[0][0][0][0] + b[0][0][0][1])/num_controls
#    proportion2 = (b[0][0][0][2] + b[0][0][0][3] + b[0][0][0][4])/num_experimentals
#                  
#    proportion3 = (a[0][0][0][0] + a[0][0][0][1])/num_controls
#    proportion4 = (a[0][0][0][2] + a[0][0][0][3] + b[0][0][0][4])/num_experimentals 
#                  
#    proportion5 = (b[1][0][0][0] + b[1][0][0][1])/num_controls
#    proportion6 = (b[1][0][0][2] + b[1][0][0][3] + b[1][0][0][4])/num_experimentals
#                  
#    proportion7 = (a[1][0][0][0] + a[1][0][0][1])/num_controls
#    proportion8 = (a[1][0][0][2] + a[1][0][0][3] + a[1][0][0][4])/num_experimentals
#                  
#    proportion9 = (b[1][2][0][0] + b[1][2][0][1])/num_controls
#    proportion10 = (b[1][2][0][2] + b[1][2][0][3] + b[1][2][0][4])/num_experimentals
#                  
#    proportion11 = (a[1][2][0][0] + a[1][2][0][1])/num_controls
#    proportion12 = (a[1][2][0][2] + a[1][2][0][3] + a[1][2][0][4])/num_experimentals
#    day1_t1_cbe = [proportion, proportion2]
#    day1_t1_aaf = [proportion3, proportion4]
#    day2_t1_cbe = [proportion5, proportion6]
#    day2_t1_aaf = [proportion7, proportion8]
#    day2_t3_cbe = [proportion9, proportion10]
#    day2_t3_aaf = [proportion11, proportion12]
#    
#    print mean_behaviors_before
#    print 'Comparing day1 t1 controls BEFORE: ' + str(mean_behaviors_before[0][0][0]) + ' against: ' + str(day1_t1_cbe[0])
#    print 'Comparing day1 t1 exp BEFORE: ' + str(mean_behaviors_before[0][0][1]) + ' against: ' + str(day1_t1_cbe[1])
#    print 'Comparing day1 t1 controls AFTER: ' + str(mean_behaviors_after[0][0][0]) + ' against: ' + str(day1_t1_aaf[0])
#    print 'Comparing day1 t1 exp AFTER: ' + str(mean_behaviors_after[0][0][1]) + ' against: ' + str(day1_t1_aaf[1])
#    
#    print 'Comparing day2 t1 controls BEFORE: ' + str(mean_behaviors_before[1][0][0]) + ' against: ' + str(day2_t1_cbe[0])
#    print 'Comparing day2 t1 exp BEFORE: ' + str(mean_behaviors_before[1][0][1]) + ' against: ' + str(day2_t1_cbe[1])
#    print 'Comparing day2 t1 controls AFTER: ' + str(mean_behaviors_after[1][0][0]) + ' against: ' + str(day2_t1_aaf[0])
#    print 'Comparing day2 t1 exp AFTER: ' + str(mean_behaviors_after[1][0][1]) + ' against: ' + str(day2_t1_aaf[1])
#    
#    print 'Comparing day2 t3 controls BEFORE: ' + str(mean_behaviors_before[1][0][4]) + ' against: ' + str(day2_t3_cbe[0])
#    print 'Comparing day2 t3 exp BEFORE: ' + str(mean_behaviors_before[1][0][5]) + ' against: ' + str(day2_t3_cbe[1])
#    print 'Comparing day2 t3 controls AFTER: ' + str(mean_behaviors_after[1][0][4]) + ' against: ' + str(day2_t3_aaf[0])
#    print 'Comparing day2 t3 exp AFTER: ' + str(mean_behaviors_after[1][0][5]) + ' against: ' + str(day2_t3_aaf[1])
    return mean_behaviors_before, mean_behaviors_after

def average_trial_group(before_list, after_list, group_names, num_behaviors):  
    add_before_set = [[],[]]
    add_after_set = [[],[]]
    trial_names = ['Trial 1', 'Trial 2', 'Trial 3']
    
    for day in range(0, len(add_before_set)):
        for ind in range(0,num_behaviors):
            add_before_set[day].append([])
            add_after_set[day].append([])

            for trial_behavior in range(0, len(trial_names)):
                add_before_set[day][ind].append(0)
                add_after_set[day][ind].append(0)

            
    num_flies_per_trial = len(group_names)
    day_names = ['Day 1', 'Day 2']
    
    for i, day_names in enumerate(day_names):
        for ind, trial in enumerate(trial_names):
            for n in range(0, num_behaviors):
                for index, name in enumerate(group_names):
                    add_before_set[i][n][ind] += before_list[i][ind][n][index]
                    add_after_set[i][n][ind] += after_list[i][ind][n][index]

                    print 'The ' + str(index) + ' bout for adding behaviors to before set list:   ' + str(add_before_set[i][n])
                    print 'The ' + str(index) + ' bout for adding behaviors to after set list:   ' + str(add_after_set[i][n])
            
    print 'FINISHED!'
    for i, day_names in enumerate(day_names):
        if i > 1:
            break
        for ind, trial in enumerate(trial_names):
            for n in range(0, num_behaviors):
                print 'Day' + str(i) + ' trial' + str(ind) + ' behavior: '+ str(n)
                add_before_set[i][n][ind] = (add_before_set[i][n][ind])/float(num_flies_per_trial)
                add_after_set[i][n][ind] = (add_after_set[i][n][ind])/float(num_flies_per_trial)
    print 'FINISHED AGAIN!'
#    b = before_list
#    a = after_list
#    print '\n'
#    print len(b)
#    print len(b[0])
#    print len(b[0][0])
#    print len(b[0][0][0])
#    day1_t1_cbe = (b[0][0][0][0] + b[0][0][0][1] + b[0][0][0][2] + b[0][0][0][3] + b[0][0][0][4])/num_flies_per_trial                  
#    day1_t1_aaf = (a[0][0][0][0] + a[0][0][0][1] + a[0][0][0][2] + a[0][0][0][3] + a[0][0][0][4])/num_flies_per_trial
#    day2_t1_cbe = (b[1][0][0][0] + b[1][0][0][1] + b[1][0][0][2] + b[1][0][0][3] + b[1][0][0][4])/num_flies_per_trial
#    day2_t1_aaf = (a[1][0][0][0] + a[1][0][0][1] + a[1][0][0][2] + a[1][0][0][3] + a[1][0][0][4])/num_flies_per_trial
#    day2_t3_cbe = (b[1][2][0][0] + b[1][2][0][1] + b[1][2][0][2] + b[1][2][0][3] + b[1][2][0][4])/num_flies_per_trial
#    day2_t3_aaf = (a[1][2][0][0] + a[1][2][0][1] + a[1][2][0][2] + a[1][2][0][3] + a[1][2][0][4])/num_flies_per_trial
#    
#    print add_before_set
#    print 'Comparing day1 t1 BEFORE: ' + str(add_before_set[0][0][0]) + ' against: ' + str(day1_t1_cbe)
#    print 'Comparing day1 t1 AFTER: ' + str(add_after_set[0][0][0]) + ' against: ' + str(day1_t1_aaf)
#    
#    print 'Comparing day2 t1 BEFORE: ' + str(add_before_set[1][0][0]) + ' against: ' + str(day2_t1_cbe)
#    print 'Comparing day2 t1 AFTER: ' + str(add_after_set[1][0][0]) + ' against: ' + str(day2_t1_aaf)
#    
#    print 'Comparing day2 t3 BEFORE: ' + str(add_before_set[1][0][2]) + ' against: ' + str(day2_t3_cbe)
#    print 'Comparing day2 t3 AFTER: ' + str(add_after_set[1][0][2]) + ' against: ' + str(day2_t3_aaf)
    
    return add_before_set, add_after_set

def diction_to_list(before_dict, after_dict):
    index_diction = {'Advancing': 0, 'Retreating': 1, 'Still': 2, 'Thrashing': 3, 'Pacing': 4}
    nump_before = []
    nump_after = []
    for behavior, lanes in before_dict.iteritems():
        nump_before.append([])
        nump_after.append([])
        
    print nump_before
    print nump_after
    
    #####Transfer everything from before_dict to nump_before and same for after_dict to nump_after
    for behavior, val in index_diction.iteritems():
        transfer_to_list(behavior, val, before_dict, nump_before)
        transfer_to_list(behavior, val, after_dict, nump_after)
    print '\n' + str(nump_before)
    print '\n' + str(nump_after)
    
    return nump_before, nump_after

def set_diction_to_list(before_dict, after_dict):
    index_diction = {'Advancing': 0, 'Retreating': 1, 'Still': 2, 'Thrashing': 3, 'Pacing': 4}
    day_list = ['Day1', 'Day2']
    trial_list= ['Trial1', 'Trial2', 'Trial3']
    nump_before = []
    nump_after = []
        
    print nump_before
    print nump_after
    test_list = []
    
    #####Transfer everything from before_dict to nump_before and same for after_dict to nump_after
    for day_ind, day in enumerate(day_list):
        nump_before.append([])
        nump_after.append([])
        test_list.append([])
        print 'Day: ' + str(day)
        for trial_incr, trial in enumerate(trial_list):
            nump_before[day_ind].append([])
            nump_after[day_ind].append([])
            test_list[day_ind].append([])
            print 'Trial: ' + str(trial)
            for behavior, val in index_diction.iteritems():
                nump_before[day_ind][trial_incr].append([])
                nump_after[day_ind][trial_incr].append([])
                test_list[day_ind][trial_incr].append([])
                    
            for behavior, val in index_diction.iteritems():
                print 'Behavior: ' + str(behavior)
                transfer_to_list(behavior, val, before_dict[day][trial], nump_before[day_ind][trial_incr])
                transfer_to_list(behavior, val, after_dict[day][trial], nump_after[day_ind][trial_incr])
                print nump_before[day_ind][trial_incr]
                print '\n' + str(nump_after[day_ind][trial_incr])
            print ':)'
    
    print '\nTest\nTest\nTest\n' + str(test_list)
    return nump_before, nump_after

def transfer_to_list(behavior, guide_diction_val, diction, given_list):
    correct_lane_check = 0
    prev_lane_check = 0
    
    for lane,val in diction[behavior].iteritems():
        correct_lane_check = lane[-1]
        
        if correct_lane_check <= prev_lane_check:
            sys.exit('ERROR\nERROR\nLOOPING THROUGH A DICTIONARY''S LANES WEREN''T IN SEQUENTIAL ORDER, EXITING!')
            
        prev_lane_check = correct_lane_check
        given_list[guide_diction_val].append(val)
        
def get_delta_data(nump_before, nump_after):
    # Get change of proportion in fly
    before_day1_list = nump_before[0]
    before_day2_list = nump_before[1]
    
    after_day1_list = nump_after[0]
    after_day2_list = nump_after[1]
#    print'\n'
#    print before_day1_list
#    for n in range(0,len(before_day1_list)):
#        print 'Length of BEFORE trial''s ' + str(len(before_day1_list[n]))
#        print 'Length of AFTER trial''s ' + str(len(after_day1_list[n]))
##        if len(before_day1_list)
#        for m in range(0, len(before_day1_list[n])):
#            print 'Length of BEFORE lane''s ' + str(len(before_day1_list[n][m]))
#            print 'Length of AFTER lane''s ' + str(len(after_day1_list[n][m]))
#    print 'Debug'
#    print 'Start'
    nump_before_delta = np.subtract(before_day2_list, before_day1_list)
    nump_after_delta = np.subtract(after_day2_list, after_day1_list)
    
#    print 'One fly data: ' + str(before_day2_list[0][0][0]) + ' and ' + str(before_day1_list[0][0][0]) + ' subbed to: ' + str(before_day2_list[0][0][0]-before_day1_list[0][0][0]) + ' compared to: ' + str(nump_before_delta[0][0][0])
#    print 'Second fly data:' + str(before_day2_list[0][1][0]) + ' and ' + str(before_day1_list[0][1][0]) + ' subbed to: ' + str(before_day2_list[0][1][0]-before_day1_list[0][1][0]) + ' compared to: ' + str(nump_before_delta[0][1][0])
#    print 'Third fly data: ' + str(before_day2_list[1][0][0]) + ' and ' + str(before_day1_list[1][0][0]) + ' subbed to: ' + str(before_day2_list[1][0][0]-before_day1_list[1][0][0]) + ' compared to: ' + str(nump_before_delta[1][0][0])
#    
#    print 'One fly data: ' + str(after_day2_list[0][0][0]) + ' and ' + str(after_day1_list[0][0][0]) + ' subbed to: ' + str(after_day2_list[0][0][0]-after_day1_list[0][0][0]) + ' compared to: ' + str(nump_after_delta[0][0][0])
#    print 'Second fly data:' + str(after_day2_list[0][1][0]) + ' and ' + str(after_day1_list[0][1][0]) + ' subbed to: ' + str(after_day2_list[0][1][0]-after_day1_list[0][1][0]) + ' compared to: ' + str(nump_after_delta[0][1][0])
#    print 'Third fly data: ' + str(after_day2_list[1][0][0]) + ' and ' + str(after_day1_list[1][0][0]) + ' subbed to: ' + str(after_day2_list[1][0][0]-after_day1_list[1][0][0]) + ' compared to: ' + str(nump_after_delta[1][0][0])
    return nump_before_delta, nump_after_delta

def gen_standard_error():
    print 'Hello'
    

def create_trial_bar_plots(nump_before, nump_after, mean_nump_before, mean_nump_after, title):
    #### Creates 8 plots. 2 for Day 1 before fly enters chamber (showing trials)
    #### 2 for Day 2. 2 for Day 1 before fly enters chamber, control v experimental
#    group_names = ['Trial 1', 'Trial 2', 'Trial 3']
#    
#    ##### Creating Day 1: Before Fly Enters Chamber
#    ax = create_plot(3, 0.15, 0.3, group_names, str(title) + 'Before', nump_before)
#    
#    plt.show()
#    
#    ##### Creating Day 1: After Fly Enters Chamber
#    ax = create_plot(3, 0.15, 0.3, group_names, str(title) + 'After', nump_after)
#    plt.show()
    
    group_names = ['Trial 1 Control', 'Trial 1 Exp', 'Trial 2 Control', ' Trial 2 Exp', 'Trial 3 Control', 'Trial 3 Exp']
    
    ##### Creating Day 1: Before Fly Enters Chamber, control v exp
    ax = create_plot(6, 0.15, 0.3, group_names, str(title) + 'Before', mean_nump_before)
    ax.text(0.5, 92.5, 'Trial 1', fontsize = 12)
    ax.text(2.5, 92.5, 'Trial 2', fontsize = 12)
    ax.text(4.5, 92.5, 'Trial 3', fontsize = 12)
    plt.show()
    
    ##### Creating Day 1: After Fly Enters Chamber, control v exp
#    ax = create_plot(6, 0.15, 0.3, group_names, str(title) + 'After', mean_nump_after)
#    
#    plt.show()

    
    
    
def create_day_bar_plots(nump_before, nump_after, mean_nump_before, mean_nump_after, title):
    ##### Creates 6 plots. 2 for creating Day1 v Day 2 before and after. 2 for creating Day1 Controls, Day1 exp v Day2 Controls, Day2 Exp before and after
    ##### 2 for increase/decrease of given behaviors between both days before and after
    group_names = ['Day 1', 'Day 2']
    
    ##### Creating Set: Before Fly Enters Chamber
    create_plot(2, 0.3, 0.6, group_names, title, nump_before)
    
    ##### Creating Set: After Fly Enters Chamber
    create_plot(2, 0.3, 0.6, group_names, title, nump_after)
    
    group_names = ['Day 1 Controls', 'Day 1 Exp', 'Day 2 Controls', 'Day 2 Exp']
    ##### Creating Set: Before Fly Enters Chamber controls v exp
    create_plot(4, 0.3, 0.6, group_names, title, nump_before)
    
    ##### Creating Set: After Fly Enters Chamber controls v exp
    create_plot(4, 0.3, 0.6, group_names, title, nump_after)
    
    ##### Creating Set: Before Fly Enters Chamber controls v exp delta
    
    
    ##### Creating Set: After Fly Enters Chamber controls v exp delta
    
    
def create_plot(n_groups, width, xtick_incr, group_names, title, data):
    fig, ax = plt.subplots()
    x_name_pos = np.arange(n_groups)
    opa = 0.75
    colors = ['b', 'g', 'r', 'turquoise', 'brown']
    incr_bar_width = 0
    behavior_list = ['Advancing', 'Retreating', 'Still', 'Thrashing', 'Pacing']
    
    for ind, behaviors in enumerate(behavior_list):
        plt.boxplot(x_name_pos + incr_bar_width, data[ind], width, color=colors[ind], label = behaviors)
        incr_bar_width += width
        
    plt.ylabel('Percent of Behavior Occuring')
    plt.title('{} Entering Alcohol Chamber'.format(title), y = 1.1, fontweight = 'bold')
    plt.xticks(x_name_pos + xtick_incr, group_names)
    plt.legend()
    
    plt.tight_layout()
    
    return ax
    
    
#def average_set_proportions(percent_lists_before, percent_lists_after, group_names, num_behaviors):
#    print '\n\n'
#    print 'Showing percent lists of before:    ' + str(percent_lists_before)
#    print '\nNow showing percent lists of after:     ' + str(percent_lists_after)
#    add_bcontrol_behaviors1 = []
#    add_bexp_behaviors1 = []
#    add_acontrol_behaviors1 = []
#    add_aexp_behaviors1 = []
#    
#    add_bcontrol_behaviors2 = []
#    add_bexp_behaviors2 = []
#    add_acontrol_behaviors2 = []
#    add_aexp_behaviors2 = []
#    
#    add_bcontrol_behaviors3 = []
#    add_bexp_behaviors3 = []
#    add_acontrol_behaviors3 = []
#    add_aexp_behaviors3 = []
#    for ind in range(0,num_behaviors):
#        add_bcontrol_behaviors1.append(0)
#        add_bexp_behaviors1.append(0)
#        add_acontrol_behaviors1.append(0)
#        add_aexp_behaviors1.append(0)
#        
#        add_bcontrol_behaviors2.append(0)
#        add_bexp_behaviors2.append(0)
#        add_acontrol_behaviors2.append(0)
#        add_aexp_behaviors2.append(0)
#        
#        add_bcontrol_behaviors3.append(0)
#        add_bexp_behaviors3.append(0)
#        add_acontrol_behaviors3.append(0)
#        add_aexp_behaviors3.append(0)
#        
#    day1 = []
#    day2 = []
##    print add_bcontrol_behaviors
##    print add_bexp_behaviors
##    print add_acontrol_behaviors
##    print add_aexp_behaviors
#    num_controls = 2
#    num_experimentals = 3
#
#    trial_names = ['Trial 1', 'Trial 2', 'Trial 3']
#    day_names = ['Day 1', 'Day 2']
#    for i, day_names in enumerate(day_names):
#        for ind, trial in enumerate(trial_names):
#            for index, name in enumerate(group_names):
#                print 'The lane index is: ' + str(index)
#                if name == 'Lane 1':
#                    print 'There is a lane ONE amount of behaviors'
##                    num_controls += 1
#                    for n in range(0, len(add_bcontrol_behaviors1)):
#                        if ind == 0:
#                            add_bcontrol_behaviors1[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors1[n] += percent_lists_after[i][ind][n][index]
#                        elif ind == 1:
#                            add_bcontrol_behaviors2[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors2[n] += percent_lists_after[i][ind][n][index]
#                        else:
#                            add_bcontrol_behaviors3[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors3[n] += percent_lists_after[i][ind][n][index]
##                        add_bcontrol_behaviors[n] += percent_lists_before[i][ind][n][index]
##                        add_acontrol_behaviors[n] += percent_lists_after[i][ind][n][index]
#                    print n
#                if name == 'Lane 2':
#                    print 'There is a lane TWO amount of behaviors'
##                    num_controls += 1
#                    for n in range(0, len(add_bcontrol_behaviors1)):
#                        if ind == 0:
#                            add_bcontrol_behaviors1[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors1[n] += percent_lists_after[i][ind][n][index]
#                        elif ind == 1:
#                            add_bcontrol_behaviors2[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors2[n] += percent_lists_after[i][ind][n][index]
#                        else:
#                            add_bcontrol_behaviors3[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors3[n] += percent_lists_after[i][ind][n][index]
##                        add_bcontrol_behaviors[n] += percent_lists_before[i][ind][n][index]
##                        add_acontrol_behaviors[n] += percent_lists_after[i][ind][n][index]
#                    print n
#                if name == 'Lane 3':
#                    print 'There is a lane THREE amount of behaviors'
##                    num_controls += 1
#                    for n in range(0, len(add_bcontrol_behaviors1)):
#                        if ind == 0:
#                            add_bcontrol_behaviors1[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors1[n] += percent_lists_after[i][ind][n][index]
#                        elif ind == 1:
#                            add_bcontrol_behaviors2[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors2[n] += percent_lists_after[i][ind][n][index]
#                        else:
#                            add_bcontrol_behaviors3[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors3[n] += percent_lists_after[i][ind][n][index]
##                        add_bcontrol_behaviors[n] += percent_lists_before[i][ind][n][index]
##                        add_acontrol_behaviors[n] += percent_lists_after[i][ind][n][index]
#                    print n
#                if name == 'Lane 4':
#                    print 'There is a lane FOUR amount of behaviors'
##                    num_experimentals += 1
#                    for n in range(0, len(add_bcontrol_behaviors1)):
#                        if ind == 0:
#                            add_bcontrol_behaviors1[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors1[n] += percent_lists_after[i][ind][n][index]
#                        elif ind == 1:
#                            add_bcontrol_behaviors2[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors2[n] += percent_lists_after[i][ind][n][index]
#                        else:
#                            add_bcontrol_behaviors3[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors3[n] += percent_lists_after[i][ind][n][index]
##                        add_bexp_behaviors[n] += percent_lists_before[i][ind][n][index]
##                        add_aexp_behaviors[n] += percent_lists_after[i][ind][n][index]
#                if name == 'Lane 5':
#                    print 'There is a lane FIVE amount of behaviors'
##                    num_experimentals += 1
#                    for n in range(0, len(add_bcontrol_behaviors1)):
#                        if ind == 0:
#                            add_bcontrol_behaviors1[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors1[n] += percent_lists_after[i][ind][n][index]
#                        elif ind == 1:
#                            add_bcontrol_behaviors2[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors2[n] += percent_lists_after[i][ind][n][index]
#                        else:
#                            add_bcontrol_behaviors3[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors3[n] += percent_lists_after[i][ind][n][index]
##                        add_bexp_behaviors[n] += percent_lists_before[i][ind][n][index]
##                        add_aexp_behaviors[n] += percent_lists_after[i][ind][n][index]
#                if name == 'Lane 6':
#                    print 'There is a lane SIX amount of behaviors'
##                    num_experimentals += 1
#                    for n in range(0, len(add_bcontrol_behaviors1)):
#                        if ind == 0:
#                            add_bcontrol_behaviors1[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors1[n] += percent_lists_after[i][ind][n][index]
#                        elif ind == 1:
#                            add_bcontrol_behaviors2[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors2[n] += percent_lists_after[i][ind][n][index]
#                        else:
#                            add_bcontrol_behaviors3[n] += percent_lists_before[i][ind][n][index]
#                            add_acontrol_behaviors3[n] += percent_lists_after[i][ind][n][index]
##                        add_bexp_behaviors[n] += percent_lists_before[i][ind][n][index]
##                        add_aexp_behaviors[n] += percent_lists_after[i][ind][n][index]
##                print 'The ' + str(index) + ' bout for adding behaviors to before control''s list:   ' + str(add_bcontrol_behaviors)
##                print 'The ' + str(index) + ' bout for adding behaviors to before exp''s list:   ' + str(add_bexp_behaviors)
##                print 'The ' + str(index) + ' bout for adding behaviors to after control''s list:   ' + str(add_acontrol_behaviors)
##                print 'The ' + str(index) + ' bout for adding behaviors to after exp''s list:   ' + str(add_aexp_behaviors)
#                
##    mean_control_behaviors = []
##    mean_exp_behaviors = []
#    mean_behaviors_before = []
#    mean_behaviors_after = []
#    print 'Num controls ' + str(num_controls)
#    print 'Num Exp ' + str(num_experimentals)
#    for m in range(0, num_behaviors):
#        print m
#        new_behaviors_before = []
#        new_behaviors_after = []
#        new_behaviors_before.append(add_bcontrol_behaviors1[m]/float(num_controls))
#        new_behaviors_before.append(add_bexp_behaviors1[m]/float(num_experimentals))
#        new_behaviors_before.append(add_bcontrol_behaviors2[m]/float(num_controls))
#        new_behaviors_before.append(add_bexp_behaviors2[m]/float(num_experimentals))
#        new_behaviors_before.append(add_bcontrol_behaviors3[m]/float(num_controls))
#        new_behaviors_before.append(add_bexp_behaviors3[m]/float(num_experimentals))
#        mean_behaviors_before.append(new_behaviors_before)
#        
#        new_behaviors_after.append(add_acontrol_behaviors1[m]/float(num_controls))
#        new_behaviors_after.append(add_aexp_behaviors1[m]/float(num_experimentals))
#        new_behaviors_after.append(add_acontrol_behaviors2[m]/float(num_controls))
#        new_behaviors_after.append(add_aexp_behaviors2[m]/float(num_experimentals))
#        new_behaviors_after.append(add_acontrol_behaviors3[m]/float(num_controls))
#        new_behaviors_after.append(add_aexp_behaviors3[m]/float(num_experimentals))
#        mean_behaviors_after.append(new_behaviors_after)
##        mean_behaviors[m].append(add_control_behaviors[m]/float(num_controls))
##        mean_behaviors[m].append(add_exp_behaviors[m]/float(num_controls))
##        mean_control_behaviors.append(add_control_behaviors[m]/float(num_controls))
##        mean_exp_behaviors.append(add_exp_behaviors[m]/float(num_experimentals))
##    for n in range(0, len(mean_control_behaviors))
#
#    hand_computed = []
#    b = percent_lists_before
#    a = percent_lists_after
#    t1c = []
#    
#    return mean_behaviors_before, mean_behaviors_after    

#def create_bar_graphs_cutoffs():
##    if DataCheck is True and len(list_to_glob) != 2:
##        print 'NEED 2 path''s to graph a Set of Data!!'
#    globbed_path = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Set 5_2015-09-23_Day2_PM1'
##    globbed_path = 
#    path_to_T1 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Set 5_2015-09-23_Day2_PM1\Advancing_2015-09-23_Day2_PM1_Trial1_Lanes13456.csv'
#    path_to_T1_diff = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Set 5_2015-09-23_Day2_PM1\Retreating_2015-09-23_Day2_PM1_Trial1_Lanes13456.csv'
#    path_to_T1_other = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Set 5_2015-09-23_Day2_PM1\Pausing_2015-09-23_Day2_PM1_Trial1_Lanes13456.csv'
#    path_to_T1_change = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Set 5_2015-09-23_Day2_PM1\Thrashing_2015-09-23_Day2_PM1_Trial1_Lanes13456.csv'
#    path_to_T1_cha = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Set 5_2015-09-23_Day2_PM1\Pacing_2015-09-23_Day2_PM1_Trial1_Lanes13456.csv'
#    print os.path.basename(path_to_T1)
#    print os.path.abspath(path_to_T1)
#    print os.path.dirname(path_to_T1)
#    list_of_paths_to_load = [path_to_T1, path_to_T1_diff, path_to_T1_other, path_to_T1_change, path_to_T1_cha]
#    results = {}
#    num_class_labels = []
#    num_percent_lists = []
#    behavior_list = []
#    before_dict = {}
#    after_dict = {}
#    cutoffs = load_frame_cutoffs('C:\Users\Tapster\Desktop\Sophia\Cutoff Frames_Dictionary.csv')
#    print cutoffs
#    prev_name = None
#    prev_day_name = None
#    prev_trial_name = None
#    
#    for indx, path in enumerate(list_of_paths_to_load):
#        with open(path, 'rb') as f:
#            # Parsing the basename of the given path for labeling the bar graph and for fetching and organizing the behavioral data for the graphs
#            abspath = os.path.normpath(path)
#            abspath = abspath.split(os.sep)
##            re.split('\\', abspath)
##            abspath.split("\\")
##            print abspath
#            #Get name of the Set #
#            set_list = abspath[-2]
#            set_list = set_list.split('_')
#            setname = set_list[0]
#
#            #Get name of the Day #
##            print set_list
#            dayname = dayname = set_list[-2]
#            if prev_day_name != dayname:
#                print 'The name of the given DAY has changed to: ' + str(dayname)
#                prev_day_name = dayname
#            
#            #Get name of the Trial #
#            basename = abspath[-1]
#            basename = basename.split('_')
##            print basename
#            trialname = basename[-2]
#            if prev_trial_name != trialname:
#                print 'The name of the given TRIAL has changed to: ' + str(trialname)
#                prev_trial_name = trialname
#                
#            #Get name tag of the trial #
#            nametag = basename[1:5]
#            print nametag
#            
#            
#            csv_reader = csv.reader(f, delimiter=',', skipinitialspace=True)
#            percentage_list, condition_label, group_names, behavior_name = read_JAABA_csv_block(csv_reader, prev_name, behavior_list, before_dict, after_dict, cutoffs, nametag)
##            print 'These are the proportions'+ str(percentage_list)
#            num_percent_lists.append(percentage_list)
#            print 'BEHAVIOR: ' + str(behavior_list)
#            print 'WITH PROPORTIONS: ' + str(percentage_list)
#            
#            classifier_label = condition_label.lstrip('% ylabel=')
#            classifier_label = classifier_label.rstrip(' (%)')
#            num_class_labels.append(classifier_label)
#            results['classifier'] = classifier_label
#            trial_label = 'Trial {}'.format(indx+1)
#            results[trial_label] = percentage_list
#            #figure out the key to enter into the cutoff_dictionary
#            split_path = path.split("\\") [-1].split("_") [1:-1]
#            joined_path = "_".join(split_path)
##            cutoffs = cutoff_dictionary[joined_path]
#    
#            cutoff_key = "{} cutoffs".format(trial_label)        
##            results[cutoff_key] = cutoffs    
#    print 'These are the behaviors being graphed!!!!  ' + str(behavior_list)
#    print 'These are the proportions:    '+ str(num_percent_lists)
##    print 'These are the proportions'+ str(num_percent_lists)
#    print group_names
#    
#    lane_diction = dict(zip(group_names, num_percent_lists))
#    
#    
#        
#    n_groups = len(group_names)
#    fig, ax = plt.subplots()
#    x_name_pos = np.arange(n_groups)
#    print x_name_pos
#    width = 0.15
#    opa = 1
#    colors = ['b', 'g', 'r', 'magenta', 'brown']
#    incr_bar_width = 0
#    
#    for ind, behaviors in enumerate(behavior_list):
#        plt.bar(x_name_pos + incr_bar_width, num_percent_lists[ind], width, alpha=opa, color=colors[ind], label = behaviors)
#        incr_bar_width += 0.15
#        
#    plt.ylabel('Percent of behavior occuring')
#    plt.title('Proportion of behaviors in ' + str('Set 5, Day 2, Trial 1'))
#    plt.xticks(x_name_pos + 0.3, group_names)
#    plt.legend()
#    
#    plt.tight_layout()
#    plt.show()
#    
#    
##    print lane_diction
#    num_behaviors = len(behavior_list)
##    mean_control_proportion, mean_exp_proportion = average_group_proportions(num_percent_lists, group_names, lane_diction, num_behaviors)
#    mean_behavior_proportions = average_group_proportions(num_percent_lists, group_names, lane_diction, num_behaviors)
#    for n in range(0, len(mean_behavior_proportions)):
#        print mean_behavior_proportions[n]
#        
##    for n in range(0, len(mean_control_proportion)):
##        print mean_control_proportion[n]
##        
##    for n in range(0, len(mean_exp_proportion)):
##        print mean_exp_proportion[n]
#        
#    n_groups = 2
#    fig2, ax2 = plt.subplots()
#    x_name_pos2 = np.arange(n_groups)
#    width = 0.20
#    incr_bar_width = 0
#    
#    for ind, behaviors in enumerate(behavior_list):
#        plt.bar(x_name_pos2 + incr_bar_width, mean_behavior_proportions[ind], width, alpha=opa, color=colors[ind], label = behaviors)
#        incr_bar_width += 0.2
#        
#    plt.ylabel('Percent of behavior occuring')
#    plt.title('Proportion of behaviors in ' + str('Set 5, Day 2, Trial 1'))
#    plt.xticks(x_name_pos + 0.3, ('Controls', 'Experimentals'))
#    plt.legend()
#    
#    plt.tight_layout()
#    plt.show()
#        
#
#
#def read_JAABA_csv_block_cutoffs(csv_reader, prev_name, behavior_list, before_dict, after_dict, cutoffs, nametag):
#    current_group = deque(maxlen=1)
#    
#    line_count = 0
#    
#    groups = []
#    parsed_groups = []
#    df_list = []
#    proportion_list = []
#    
#    group_names = []
#
#    grab_xdata = None
#
#    grab_ydata = None
#    
#    header_read = False
#    condition_label = None
#    raw_data_check = False
#    cutoff_incr = 0
#
#    while True:
#        try:
#            line = csv_reader.next()
#        except StopIteration:
#            print 'Reached end of the csv, breaking out of while LOOP!!'
#            break
#        
#        if line:      
#            #Change df_type for creation of the x and y axis of the bar graphs
#            if raw_data_check and line[0].startswith('% group'):
#                 split_line = line[0].split()
#                 print split_line[-1]
#                 print split_line[-2]
#                 group_names.append(split_line[-2] + ' ' + split_line[-1])
#                 print group_names[-1]
#                
#            if line[0].startswith('% raw data'):
#                raw_data_check = True
#            
#            
#            if line[0].startswith('% type'):
#                df_type = line
#            if line[0].startswith('% title'):
#                df_title = line
#            if line[0].startswith('% xlabel'):
#                df_xlabel = line
#            # % ylabel is what's really needed. 
#            if line[0].startswith('% ylabel'):
#                df_ylabel = line[0][9:]
#                df_ylabel = df_ylabel.rstrip(' (%)')
#                if any(char.isdigit() for char in df_ylabel):
#                    print 'There is a number/s in the string'
#                    df_ylabel = df_ylabel[0:-1]
#                print(' ')
#                print('df_ylabel is: ' + str(df_ylabel))
#                if prev_name != df_ylabel:
#                    before_dict[df_ylabel] = {}
#                    after_dict[df_ylabel] = {}
#                    print 'There is a new behavior being graphed!  ' + str(df_ylabel)
#                    behavior_list.append(df_ylabel)
#                prev_name = df_ylabel
#                condition_label = df_ylabel[0]
#                header_read = True
#            
#            if header_read and '% group' in line[0]:
#                line = filter(None, line)
#                groups.append(line)
#                parsed_groups.append(line[0].lstrip('% group '))
#                #save the group name to use when generating dataframes
#                current_group.append(line[0].lstrip('% group '))
#            
#            if grab_xdata:
#                line = filter(None, line)
#                x_key = current_group[0]
#                # convert read in "string version of line into actual numbers"
#                # The JAABA datafiles always have a lingering ',' after each data group
#                # this results in a ' ' character that float() cannot convert. 
#                # (So remember to strip the last element from the list)
#                xdata = {x_key:map(float, line[:-1])}
#                xdata_df = pd.DataFrame(xdata)
##                df_list.append(xdata_df)
#                grab_xdata = False
#                
#            if grab_ydata: 
#                line = filter (None, line)
#                numt_line = map(float, line[:-1])
#                
#                cur_cutoff_frame = cutoffs[nametag][cutoff_incr]
#                before_cut = numt_line[0:cur_cutoff_frame]
#                after_cut = numt_line[cur_cutoff_frame:-1]
#                print cutoff_incr
#                print group_names[-1]
#                print cur_cutoff_frame
#                print '\n\n\n\n'
#                print before_cut
#                print after_cut
#                cutoff_incr += 1
#                proportion = compute_proportion(numt_line)
#                proportion_list.append(proportion)
#                overall_dict[df_ylabel][group_names[-1]] = proportion
#                print 'The percentage of the given BEHAVIOR is: ' + str(proportion)
##                print 'Number type version of line: ' + str(numt_line)
#                y_key = '{}'.format(current_group[0])
#                print 'y_key is: ' + y_key
#                ydata = {y_key:map(float, line[:-1])}
##                print 'The given ydata is: ' + str(ydata)
#                ydata_df = pd.DataFrame(ydata)
#                df_list.append(ydata_df)
#                grab_ydata = False
#        
#            if '% xdata' in line:
#                grab_xdata = True
#            if line[0].startswith('% experiment'):
#                grab_ydata = True
#            
#        line_count += 1
#        
##    output_df = pd.concat(df_list, axis=1)    
#
#    return proportion_list, condition_label, group_names, df_title
    
globbed_path2 = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Set 6_2015-09-23_Day2_PM2'
globbed_path = 'E:\JAABA Analysis\JAABA Training and Output\Runway vs Chamber Behavior Analysis\Set 6_2015-09-22_Day1_PM2'
create_bar_graphs([globbed_path, globbed_path2], True, True)