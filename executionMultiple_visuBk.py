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
import executionNinstances as exec_N_insts


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
    # scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate100PeriodsMultipleParams.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate50Periods_instance12.json"
    
    # scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate150Periods.json"
    
    # scenarioFile = "./data_scenario_JeuDominique/automateTest.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate50Periods_instance12.json"
    
    
    bool_runAlgo = True # False, True
    
    scenario = None
    with open(scenarioFile) as file:
        scenario = json.load(file)
        pass
    
    
    start = time.time()
    scenario_dir = None
    if bool_runAlgo:
        runapp.run_algos_count_prodCartesien(scenario=scenario, logfiletxt=logfiletxt)
        scenario_dir = f"{scenario['scenarioName']}"
    else: 
        scenario_dir = f"{scenario['scenarioName']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"
        
        
    # ------------------------          VISU V2         ----------------------- 
    folder_2_search = os.path.join(scenario["scenarioPath"], scenario["name"], scenario['simul']['n_instance'], scenario_dir, "datas", "dataResult")
    folder_2_search_LRI = os.path.join(scenario["scenarioPath"], scenario["name"], scenario['simul']['n_instance'], scenario_dir, "datas", "LRI_REPART")
    folder_2_save = os.path.join(scenario["scenarioPath"], scenario["name"], scenario['simul']['n_instance'], scenario_dir, "datas", "dataViz")
    filename_csv = "dataframes.csv"
    print(f"folder_2_search={folder_2_search},\n folder_2_save={folder_2_save},\n filename_csv={filename_csv}")
    print(f"scenarioPath = {scenario['scenarioPath']}")
    print(f"scenario_dir = {scenario_dir}")
    
    print("############################################# #########################")
    print(f"################################ instance_{ scenario['simul']['n_instance'] } #########################")
    print("############################################# #########################")
    
    df = vizBk_v2.find_csvfile(folder_2_search=folder_2_search, 
                                folder_2_save=folder_2_save, 
                                filename_csv=filename_csv)
    
    scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], 
                                            scenario["name"], 
                                            scenario['simul']['n_instance'], 
                                            scenario_dir, "datas", "dataViz")
    
    vizBk_v2.plot_all_figures_withMeanLRI(df=df, 
                                  period_min=scenario["simul"]["period_min"],
                                  scenarioCorePathDataViz=scenarioCorePathDataViz, 
                                  folder_2_search_LRI=folder_2_search_LRI)
    
    print(f"runtime = {time.time() - ti}")
    
    ###########################################################################
    # fusionner tous les dataframes dans le dossier resultExecNinstances
    folder = os.path.join(scenario["scenarioPath"], scenario["name"], "resultExecNinstances")
    folder_2_save = os.path.join(scenario["scenarioPath"], scenario["name"], "RESUME_Ninstances")
    df = exec_N_insts.merge_DF_4_ValSG_QTStock(folder=folder, folder_2_save=folder_2_save)
    # faire un groupby par le tuple Tup=(rho, mu, epsilon, lambda_poisson, learning_rate, M_exec_lri)
    # pour obtenir les valeurs (n_instance, algoName, ValSG, QTStock)
    
    # pour chaque tuple Tup, 
        # 1) I = [LRI, SSA, CSA, SyA]
        # 2) determiner valSG_min = min(valSG(LRI), valSG(SSA), valSG(CSA), valSG(SyA)) or min( valSG(i), i \in I  )
        # 3) calculer  valSG(i) = valSG(i) / valSG_min, i \in I
        # calculer QTStock(i) = QTStock(i) / QTStock_min avec 1) 2) 3)
        # calculer la moyenne de ValSG, QTStock avec ValSG = sum_{i=1}^{N} ValSG_i pour tout i \in I /  PROBLEME DE COMPREHENSION
        # garder, pour chaque instance, l'algo qui a ValSG_min  
    
    print("\n\n\n\n------------------------------------------------------------------")
    print(f"-----------runtime = {time.time() - ti}")
    print("-----------------------------------------------------------------")