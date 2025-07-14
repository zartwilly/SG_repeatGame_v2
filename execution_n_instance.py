#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 09:58:37 2025

@author: willy

generation d'une instance particulier M fois
instance = instance12
"""

import os
import json
import time
import glob
import pickle

import pandas as pd

import RunApp as runapp
import visuDataBOKEH_v2 as vizBk_v2
import executionNinstances as exec_N_insts

from pathlib import Path


###############################################################################
#                       debut:
###############################################################################
def merge_DF_from_all_algos_and_LRI_executions(folder:str):
    """
    

    Parameters
    ----------
    folder : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    dico_algos = dict()
    dico_algoLRI = dict()
    # Lister tous les fichiers pkl
    pkl_files = list(glob.glob(f"{folder}/*.pkl"))
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as file:
                data_app = pickle.load(file)
            print("APP.pkl Data loaded successfully:")
        except FileNotFoundError:
            print("Error: 'my_data.pkl' not found. Please ensure the file exists.")
        except Exception as e:
            print(f"An error occurred while loading the pickle file: {e}")
            
        algoName = pkl_file.split("_")[-4]
        Mexec = pkl_file.split("_")[-2]
        algoName = "LRI_REPART" if algoName == "REPART" else algoName
        if algoName != "LRI_REPART":
            dico_algos[algoName] = data_app.valSG_A
        else:
            dico_algoLRI[algoName+"_"+Mexec] = data_app.valSG_A
            
    dico_exec = dict()
    
    for algo_lri, valSG in  dico_algoLRI.items():
        dico_tmp = dico_algos.copy()
        m_exec = algo_lri.split("_")[-1]
        dico_tmp["LRI_REPART"] = valSG
        min_ValSG = min(dico_tmp.values())
        # Divide all values by the minimum value and create a new dictionary
        d_divided = {f"{key}_div": value / min_ValSG for key, value in dico_tmp.items()}
        
        d_divided.update(dico_tmp)
        
        dico_exec[f"exec{m_exec}"] = d_divided
        
    df_exec = pd.DataFrame(dico_exec).T
    
    rep = os.path.join(folder, "Results") 
    Path(rep).mkdir(parents=True, exist_ok=True)
    df_exec.to_csv(f"{rep}/dataframe_EXEC-LRI_ninstance_ValSG.csv")
    
    return dico_exec
        
    
###############################################################################
#                       FIN:
###############################################################################   

if __name__ == '__main__':
    ti = time.time()
    
    logfiletxt = "traceApplication.txt"
    
    # scenarioFile = "./data_scenario_JeuDominique/automateTest.json"
    n_instance = 12
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate50Periods_instance12.json"
    n_instance = 1
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate50Periods1einstance.json"
    n_instance = 1
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate50Periods1einstance_BESTIE.json"
    
    bool_runAlgo = True # False, True
    
    scenario = None
    with open(scenarioFile) as file:
        scenario = json.load(file)
        pass
    
    
    exec_N_insts.executionVisuBokeh_OneInstance(scenario=scenario, 
                                                n_instance=n_instance,
                                                bool_runAlgo=bool_runAlgo,
                                                logfiletxt=logfiletxt)
    
    # tableau recap des ValSG pour tous les versions de LRI_REPART
    folder = os.path.join(scenario["scenarioPath"], scenario["name"], "resultAPPInstances")
    dico_exec = merge_DF_from_all_algos_and_LRI_executions(folder=folder)

    print("\n\n\n\n------------------------------------------------------------------")
    print(f"-----------runtime = {time.time() - ti}")
    print("-----------------------------------------------------------------")
