#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 16:59:15 2025

@author: willy
"""

import os
import json
import time
import glob

import itertools as it
import pandas as pd

import RunApp as runapp
import visuDataBOKEH_v2 as vizBk_v2

from pathlib import Path


def merge_DF_4_ValSG_QTStock(folder, folder_2_save):
    """
    merge Dataframes from folder directory to one dataframe

    Parameters
    ----------
    folder : str
        DESCRIPTION.
        
    folder_2_save : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    Path(folder_2_save).mkdir(parents=True, exist_ok=True)
    
    # Lister tous les fichiers CSV
    csv_files = list(glob.glob(f"{folder}/*.csv"))
    
    # Lire tous les CSV en DataFrames dans une liste
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    
    # Concaténer tous les DataFrames en un seul (concat vertical)
    merged_df = pd.concat(dfs, ignore_index=True)
    
    cols = ['rho', 'mu', 'epsilon','lambda_poisson', 'learning_rate', 'n_instance', 'algoName']; 
    #seleted_cols = ['ValSG', 'QTStock']; 
    # df_tmp = merged_df.groupby(cols)[seleted_cols].sum()
    
    # ### 
    # #  for one tuple 
    # ### 
    # rho=5; mu=0.1; epsilon=0; lambda_poisson=0.5; learning_rate=0.001; 
    # df_test = merged_df[(merged_df.rho==rho) & (merged_df.mu==mu) &
    #                     (merged_df.epsilon==epsilon) &
    #                     (merged_df.lambda_poisson==lambda_poisson) & 
    #                     (merged_df.learning_rate==learning_rate) ]
    # N_insts = [] 
    # for n_instance in df_test.n_instance.unique():
    #     df_n_inst = df_test[df_test.n_instance ==  n_instance]
        
    #     # Trouver la valeur minimale de ValSG pour instance11
    #     min_valsg = df_n_inst['ValSG'].min()
    #     min_qtstcok = df_n_inst['QTStock'].min()
        
    #     print(f"Valeur minimale de ValSG pour instance11 : {min_valsg}")
        
    #     # Diviser toutes les valeurs de ValSG par cette valeur minimale
    #     df_n_inst['ValSG_div_min'] = df_n_inst['ValSG'] / min_valsg
    #     df_n_inst['QTStock_div_min'] = df_n_inst['QTStock'] / min_qtstcok
        
    #     N_insts.append(df_n_inst)
        
    # df_N_inst = pd.concat(N_insts, ignore_index=False)
    
    # df_MoyNinst_ValSG_QTStock = df_N_inst.groupby('algoName')[["ValSG_div_min", "QTStock_div_min"]].mean()
    # df_MoyNinst_ValSG_QTStock.rename(columns={"ValSG_div_min":"moyNinst_ValSG_div_min", 
    #                                           "QTStock_div_min":"moyNinst_QTStock_div_min"}, 
    #                                  inplace=True)
    
    # df_MoyNinst_ValSG_QTStock.to_csv(f"dataframe_Moyenne_Ninstance_ValSG_QTStock_rho{rho}_mu{mu}_epsilon{epsilon}_lr{learning_rate}.csv")
    
    
    # # compter les N fois de ValSG
    # cols = ['n_instance', 'algoName', 'ValSG', 'QTStock', 'ValSG_div_min', 'QTStock_div_min']
    # df_tmp_valsg = df_N_inst[cols].pivot(index='n_instance', columns='algoName', values=['ValSG_div_min'])
    # df_tmp_valsg_cp = df_tmp_valsg.copy()
    # df_tmp_valsg_cp[df_tmp_valsg_cp != 1] = 0
    # df_res_valsg = pd.merge(df_tmp_valsg, df_tmp_valsg_cp, left_index=True, right_index=True)
    
    # df_tmp_qtstock = df_N_inst[cols].pivot(index='n_instance', columns='algoName', values=['QTStock_div_min'])
    # df_tmp_qtstock_cp = df_tmp_qtstock.copy()
    # df_tmp_qtstock_cp[df_tmp_qtstock_cp != 1] = 0
    # df_res_qtstock = pd.merge(df_tmp_qtstock, df_tmp_qtstock_cp, left_index=True, right_index=True)
    
    # df_res_valsg_qtstock = df_N_inst[cols].pivot(index='n_instance', columns='algoName', values=['ValSG','QTStock'])
    
    # df_res = pd.concat([df_res_valsg, df_res_qtstock, df_res_valsg_qtstock], axis=1)
    
    # df_res.to_csv(f"dataframe_RES_Ninstances_ValSG_QTStock_rho{rho}_mu{mu}_epsilon{epsilon}_lr{learning_rate}.csv")
    
    
    ###########################################################################
    #  for many tuples
    ###########################################################################
    rhos = merged_df.rho.unique()
    mus = merged_df.mu.unique()
    epsilons = merged_df.epsilon.unique()
    lambda_poissons = merged_df.lambda_poisson.unique()
    learning_rates = merged_df.learning_rate.unique()
    
    for (rho, mu, epsilon, lambda_poisson, learning_rate) \
        in it.product(rhos, mus, epsilons, lambda_poissons, learning_rates):
        
            df_test = merged_df[(merged_df.rho==rho) & (merged_df.mu==mu) &
                                (merged_df.epsilon==epsilon) &
                                (merged_df.lambda_poisson==lambda_poisson) & 
                                (merged_df.learning_rate==learning_rate) ]
            N_insts = [] 
            for n_instance in df_test.n_instance.unique():
                df_n_inst = df_test[df_test.n_instance ==  n_instance]
                
                # Trouver la valeur minimale de ValSG pour instance11
                min_valsg = df_n_inst['ValSG'].min()
                min_qtstcok = df_n_inst[(df_n_inst.algoName == 'LRI_REPART') | 
                                        (df_n_inst.algoName == 'SSA')]['QTStock'].min()
                
                print(f"Valeur minimale de ValSG pour instance11 : {min_valsg}")
                
                # Diviser toutes les valeurs de ValSG par cette valeur minimale
                df_n_inst['ValSG_div_min'] = df_n_inst['ValSG'] / min_valsg
                df_n_inst['QTStock_div_min'] = df_n_inst['QTStock'] / min_qtstcok
                
                N_insts.append(df_n_inst)
                
            df_N_inst = pd.concat(N_insts, ignore_index=False)
            
            df_MoyNinst_ValSG_QTStock = df_N_inst.groupby('algoName')[["ValSG_div_min", "QTStock_div_min"]].mean()
            df_MoyNinst_ValSG_QTStock.rename(columns={"ValSG_div_min":"moyNinst_ValSG_div_min", 
                                                      "QTStock_div_min":"moyNinst_QTStock_div_min"}, 
                                             inplace=True)
            
            df_MoyNinst_ValSG_QTStock.to_csv(f"{folder_2_save}/dataframe_Moyenne_Ninstance_ValSG_QTStock_rho{rho}_mu{mu}_epsilon{epsilon}_lr{learning_rate}.csv")
            
            
            # compter les N fois de ValSG
            cols = ['n_instance', 'algoName', 'ValSG', 'QTStock', 'ValSG_div_min', 'QTStock_div_min']
            df_tmp_valsg = df_N_inst[cols].pivot(index='n_instance', columns='algoName', values=['ValSG_div_min'])
            df_tmp_valsg_cp = df_tmp_valsg.copy()
            df_tmp_valsg_cp[df_tmp_valsg_cp != 1] = 0
            df_res_valsg = pd.merge(df_tmp_valsg, df_tmp_valsg_cp, left_index=True, right_index=True)
            
            df_tmp_qtstock = df_N_inst[cols].pivot(index='n_instance', columns='algoName', values=['QTStock_div_min'])
            df_tmp_qtstock_cp = df_tmp_qtstock.copy()
            df_tmp_qtstock_cp[df_tmp_qtstock_cp != 1] = 0
            df_res_qtstock = pd.merge(df_tmp_qtstock, df_tmp_qtstock_cp, left_index=True, right_index=True)
            
            df_res_valsg_qtstock = df_N_inst[cols].pivot(index='n_instance', columns='algoName', values=['ValSG','QTStock'])
            
            df_res = pd.concat([df_res_valsg, df_res_qtstock, df_res_valsg_qtstock], axis=1)
            
            df_res.to_csv(f"{folder_2_save}/dataframe_RES_Ninstances_ValSG_QTStock_rho{rho}_mu{mu}_epsilon{epsilon}_lr{learning_rate}.csv")
    
    return merged_df
    
def executionVisuBokeh_OneInstance(scenario: dict, n_instance:str, 
                                   bool_runAlgo:bool, logfiletxt:str):
    """
    

    Parameters
    ----------
    scenario : dict
        DESCRIPTION.

    Returns
    -------
    None.

    """
    scenario['simul']['n_instance'] = f'instance{n_instance}'
    
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
    print(f"################################ instance_{n_instance} #########################")
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
    

if __name__ == '__main__':
    
    ti = time.time()
    
    logfiletxt = "traceApplication.txt"
    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate5Periods10instances_TEST.json"
    
    scenario = None
    with open(scenarioFile) as file:
        scenario = json.load(file)
        
    bool_runAlgo = True # False
        
    N_instance = scenario["simul"]["N_instance"]
    for n_instance in range(N_instance):
        
        executionVisuBokeh_OneInstance(scenario=scenario, n_instance=n_instance, 
                                       bool_runAlgo=bool_runAlgo)
        
    print(f"runtime = {time.time() - ti}")
    
    ###########################################################################
    # fusionner tous les dataframes dans le dossier resultExecNinstances
    folder = os.path.join(scenario["scenarioPath"], scenario["name"], "resultExecNinstances")
    folder_2_save = os.path.join(scenario["scenarioPath"], scenario["name"], "RESUME_Ninstances")
    df = merge_DF_4_ValSG_QTStock(folder=folder, folder_2_save=folder_2_save)
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
        
        