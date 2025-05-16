#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:34:17 2025

@author: willy
"""
import os
import json 
import glob

import pandas as pd

###############################################################################
#
#   parcours recursif de dossiers pour trouver tous les fichiers .csv: debut
#
###############################################################################
def trouver_csvfile(folder_2_search:str, folder_2_save:str, filename_csv:str):
    """
    parcours recursif de dossiers pour trouver tous les fichiers .csv afin 
    de former un dataframe resumant les executions de tous les p_uplets de 
    combinaisons de parametres

    Parameters
    ----------
    folder_2_search : str
        DESCRIPTION.
    folder_2_save : str
        DESCRIPTION.
        
    filename_csv: str

    Returns
    -------
    None.

    """
    
    #csv_files = glob.iglob(f'{folder_2_search}/**/*.csv', recursive=True)
    
    # merge all csv  files into dataframes
    csv_files = glob.iglob(f'{folder_2_search}/**/*.csv', recursive=True); 
    df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True);
    
    # fill na value ()
    df['scenarioName'].ffill(inplace=True)
    df['coef_phiepoplus'].ffill(inplace=True)
    df['coef_phiepominus'].ffill(inplace=True)
    
    # delete columns prosumers.1 and Unnamed: 0
    df = df.drop(['prosumers.1', 'Unnamed: 0'], axis=1)
    
    df.to_csv(os.path.join(folder_2_save, filename_csv), index=False)
    
    assert df.shape == (151200, 60), "**** VERY BAD df.shape != (151200, 60) ****"
    
    return df
    


###############################################################################
#
#   parcours recursif de dossiers pour trouver tous les fichiers .csv: FIN
#
###############################################################################


###############################################################################
#
#   parcours recursif de dossiers pour trouver tous les fichiers NE_brute.csv: DEBUT
#
###############################################################################
def trouver_NEbrute_file_by_algo(folder_2_search:str): #, algoName:str):
    """
    rechercher tous les fichiers NE_brute.csv 

    Parameters
    ----------
    folder_2_search : str
        DESCRIPTION.
    algoName : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    csv_files = glob.iglob(f'{folder_2_search}/**/*NE_brute.csv', recursive=True); 
    dfs = []
    for file in csv_files:
        # Extraire les valeurs
        valeur_mu = None
        valeur_rho = None
        valeur_epsilon = None
        valeur_lambda = None
        valeur_lr = None
        valeur_M_exec_lri = None
        
        for partie in file.split(os.sep):
            if partie.startswith('mu_'):
                valeur_mu = partie.split('_')[1]
            elif partie.startswith('rho_'):
                valeur_rho = partie.split('_')[1]
            elif partie.startswith('epsilon_'):
                valeur_epsilon = partie.split('_')[1]
            elif partie.startswith('lambda_'):
                valeur_lambda = partie.split('_')[1]
            elif partie.startswith('lr_'):
                valeur_lr = partie.split('_')[1]
            elif partie.startswith('M_exec_lri_'):
                valeur_M_exec_lri = partie.split('_')[-1]
                
        print(f"Mu : {valeur_mu}, Rho : {valeur_rho}, Epsilon : {valeur_epsilon},"+
              f" Lambda : {valeur_lambda}, LR : {valeur_lr}, M_exec_lri : {valeur_M_exec_lri}")
        
        df = pd.read_csv(file, index_col=0)
        
        df["mu"] = float(valeur_mu); df["rho"] = float(valeur_rho) ; 
        df["epsilon"] = float(valeur_epsilon) ; 
        df["lambda"] = float(valeur_lambda) ; 
        df["learning_rate"] = float(valeur_lr) ; 
        df["M_exec_lri"] = int(valeur_M_exec_lri)
        
        dfs.append(df)
        
    dfs = pd.concat(dfs)
    
    M_exec = len(dfs.M_exec_lri.unique())
    
    prosumer_cols = [col for col in dfs.columns if col.startswith('prosumer')];
    dfs["compte_1"] = dfs[prosumer_cols].apply(lambda row: sum(row == 1), axis=1)
    
    dfs[["period","compte_1"]].groupby("period").mean()
    
    return dfs
    
###############################################################################
#
#   parcours recursif de dossiers pour trouver tous les fichiers NE_brute.csv: FIN
#
###############################################################################


###############################################################################
#
#                   execution fonctions: Debut
#
###############################################################################
if __name__ == '__main__':

    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate_test.json"
    
    scenario = None
    with open(scenarioFile) as file:
        scenario = json.load(file)
        
    scenario_dir = f"{scenario['scenarioName']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"
    folder_2_search = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataResult")
    folder_2_save = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataViz")
    filename_csv = "dataframes.csv"
    print(f"{folder_2_search}")
    
    # trouver_csvfile(folder_2_search=folder_2_search, folder_2_save=folder_2_save, filename_csv=filename_csv)
    folder_2_search =  os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "LRI_REPART")
    dfs = trouver_NEbrute_file_by_algo(folder_2_search)
    