#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 18:21:57 2025

@author: willy

multiprocess execution with application 
"""
import os
import time
import json
import RunApp as ra
import itertools as it

import multiprocessing as mp

import visuDataBOKEH_v2 as vizBk_v2

if __name__ == '__main__':

    logfiletxt = "traceApplication.txt"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate.json"
    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate_test.json"
    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate_rho5_mu001.json"
    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate_test_MP.json"
    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomateMorePeriods.json"
    
    scenarioFile = "./data_scenario_JeuDominique/automateTest.json"
    
    start = time.time()
    scenario = None
    with open(scenarioFile) as file:
        scenario = json.load(file)

    
    rhos = scenario["simul"]["rhos"]
    mus = scenario["simul"]["mus"]
    learning_rates = scenario["algo"]["LRI_REPART"]["learning_rates"]
    epsilons = scenario["simul"]["epsilons"]
    lambda_poissons = scenario["simul"]["lambda_poissons"]
    M_exec_lri_s = range(0,scenario["simul"]["M_execution_LRI"])
    algoNames = list(scenario["algo"].keys())
    
    
    algoNamesNoLRI = [algoName for algoName in algoNames if "LRI" not in algoName]
    algoNamesLRI = [algoName for algoName in algoNames if "LRI" in algoName]
    
    prod_cart_NoLRI = it.product(algoNamesNoLRI, rhos, mus, epsilons, lambda_poissons, learning_rates, [0])
    prod_cart_LRI = it.product(algoNamesLRI, rhos, mus, epsilons, lambda_poissons, learning_rates, M_exec_lri_s)
    
    prod_cart_NoLRI_LRI = it.chain(prod_cart_NoLRI, prod_cart_LRI)
    
    # create params
    Params = list()
    cpt = 0
    for (algoName, rho, mu, epsilon, lambda_poisson, learning_rate, M_exec_lri) in prod_cart_NoLRI_LRI:
        cpt += 1
        param_tmp = (scenario.copy(), logfiletxt, algoName, rho, mu, epsilon, 
                     lambda_poisson, learning_rate, M_exec_lri)
        Params.append(param_tmp)
    
    print(f"***** cpt={cpt} ******")
    # multi processing execution
    p = mp.Pool(mp.cpu_count()-1)
    p.starmap(
        ra.run_algos_one_instance,
        Params
    )
    
    
    ### visualization
    # scenario_dir = f"{scenario['scenarioName']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"
    # folder_2_search = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataResult")
    # folder_2_search_LRI = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "LRI_REPART")
    # folder_2_save = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataViz")
    # filename_csv = "dataframes.csv"
    # print(f"{folder_2_search}")
    
    # df = vizBK_v2.find_csvfile(folder_2_search=folder_2_search, folder_2_save=folder_2_save, filename_csv=filename_csv)
    
    # scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataViz")
    
    # vizBK_v2.plot_all_figures_withMeanLRI(df=df, 
    #                               scenarioCorePathDataViz=scenarioCorePathDataViz, 
    #                               folder_2_search_LRI=folder_2_search_LRI)
    
    
    scenario_dir = f"{scenario['scenarioName']}"
    scenario_dir = f"{scenario['scenarioName']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"
    folder_2_search = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataResult")
    folder_2_search_LRI = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "LRI_REPART")
    folder_2_save = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataViz")
    filename_csv = "dataframes.csv"
    print(f"folder_2_search={folder_2_search},\n folder_2_save={folder_2_save},\n filename_csv={filename_csv}")
    print(f"scenarioPath = {scenario['scenarioPath']}")
    print(f"scenario_dir = {scenario_dir}")
    
    df = vizBk_v2.find_csvfile(folder_2_search=folder_2_search, folder_2_save=folder_2_save, filename_csv=filename_csv)
    
    scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataViz")
    
    vizBk_v2.plot_all_figures_withMeanLRI(df=df, 
                                 scenarioCorePathDataViz=scenarioCorePathDataViz, 
                                 folder_2_search_LRI=folder_2_search_LRI)
    
    

    print(f"Running time = {time.time() - start} cpt={cpt}")
