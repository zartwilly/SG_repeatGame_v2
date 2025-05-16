#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 01:08:33 2025

@author: willy

execution + visualization

"""
import os
import json
import time

import RunApp as runapp
import visuDataBOKEH_v2 as vizBk_v2


if __name__ == '__main__':

    ti = time.time()
    
    logfiletxt = "traceApplication.txt"
    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate_test.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate_rho5_mu001.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomateMorePeriods.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate100Periods.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate50Periods.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate50PeriodsMultipleParams.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate100PeriodsMultipleParams.json"
    
    # scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate150Periods.json"
    
    
    bool_runAlgo = True # False, True
    
    scenario = None
    with open(scenarioFile) as file:
        scenario = json.load(file)
        pass
    
    start = time.time()
    if bool_runAlgo:
        runapp.run_algos_count_prodCartesien(scenario=scenario, logfiletxt=logfiletxt)
    print(f"Running time ALGOS = {time.time() - ti}")
    
    
    
    # ------------------------          VISU V2         -----------------------
    # scenario_dir = f"{scenario['scenarioName']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"
    # #scenario_dir = os.path.join(scenario["scenarioPath"], scenario_dir)
    # print(f"{scenario_dir}")
    scenario_dir = f"{scenario['scenarioName']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"
    folder_2_search = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataResult")
    folder_2_search_LRI = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "LRI_REPART")
    folder_2_save = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataViz")
    filename_csv = "dataframes.csv"
    print(f"{folder_2_search}")
    
    df = vizBk_v2.find_csvfile(folder_2_search=folder_2_search, folder_2_save=folder_2_save, filename_csv=filename_csv)
    
    scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataViz")
    
    vizBk_v2.plot_all_figures_withMeanLRI(df=df, 
                                 scenarioCorePathDataViz=scenarioCorePathDataViz, 
                                 folder_2_search_LRI=folder_2_search_LRI)

    
    print(f"runtime = {time.time() - ti}")