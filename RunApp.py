#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 11:25:56 2025

@author: willy
"""
import os
import io
import json
import glob
import pickle
import time

import InstanceGeneratorV2 as igV2
import application as apps
import smartgrid as sg
import agents as ag
import pandas as pd

import auxiliary_functions as aux

import itertools as it

from pathlib import Path


# scenarioName = f"{scenario['scenarioName']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"

#------------------------------------------------------------------------------
#                DEBUT : Save data in the log file
#------------------------------------------------------------------------------
def monitoring_after_algorithm(algoName, file, application):
    """
    
    monitoring some variables after running some algorithms
    

    Returns
    -------
    None.

    """ 
    file.write("\n___Storage___ \n")
    for i in range(application.SG.prosumers.size):
        file.write("__Prosumer " + str(i + 1) + "___\n")
        for t in range(application.SG.nbperiod):
            file.write("Period " + str(t + 1))
            file.write(" : Storage : " + str(application.SG.prosumers[i].storage[t])+ "\n")
            
    file.write("\n___InSG, OutSG___ \n")
    for t in range(application.SG.nbperiod):
        file.write(" *** Period " + str(t + 1))
        file.write(" InSG : " + str(application.SG.insg[t]))
        file.write(" OutSG: "+ str(application.SG.outsg[t]))
        file.write(" valNoSGCost: " + str(application.SG.ValNoSGCost[t]) +"*** \n")
        for i in range(application.SG.prosumers.size):
            file.write("__Prosumer " + str(i + 1) +":")
            file.write(" Cons = "+ str(application.SG.prosumers[i].consit[t]))
            file.write(", Prod = "+ str(application.SG.prosumers[i].prodit[t]))
            file.write(", mode = "+ str(application.SG.prosumers[i].mode[t]))
            file.write(", state = "+ str(application.SG.prosumers[i].state[t]))
            file.write("\n")
            
    file.write("\n___Metrics___"+ "\n")
    file.write("ValSG : "+ str(application.valSG_A)+ "\n")
    file.write("valNoSG_A    : "+ str(application.valNoSG_A)+ "\n")
    file.write("valNoSGCost_A    : "+ str(application.valNoSGCost_A)+ "\n")
    file.write("ValObjAi    : "+"\n")
    for i in range(application.SG.prosumers.size):
        file.write("__Prosumer " + str(i + 1) + "___ :" +str(round(application.Obj_ai[i], 2)) + "\n")
        
    file.write(f"________RUN END {algoName} " + str(1) +"_________" + "\n\n")
    

def monitoring_before_algorithm(file, application):
    """
    monitoring some variables BEFORE running some algorithms

    Returns
    -------
    None.

    """
    print("________RUN ",1,"_________")
    file.write("________RUN " + str(1) +"_________" + "\n")
    
    file.write("\n___Configuration___ \n")
    for prosumer_i in range(application.SG.prosumers.size):
        file.write("__Prosumer " + str(prosumer_i + 1) + "___\n")
        for period in range(application.SG.nbperiod):
            file.write("Period " + str(period + 1))
            file.write(" : Production : " + str(application.SG.prosumers[prosumer_i].production[period]))
            file.write(" Consumption : " + str(application.SG.prosumers[prosumer_i].consumption[period]))
            file.write(" Storage : " + str(application.SG.prosumers[prosumer_i].storage[period])+ "\n")
            
#------------------------------------------------------------------------------
#                FIN : Save data in the log file
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                DEBUT : Generer des donnees selon scenarios
#------------------------------------------------------------------------------
def create_repo_for_save_jobs(scenario:dict):
    
    
    scenarioCorePath = os.path.join(scenario["scenarioPath"], scenario["name"], scenario['simul']['n_instance'], scenario["scenarioName"])
    scenarioCorePathData = os.path.join(scenario["scenarioPath"], scenario["name"], scenario['simul']['n_instance'], scenario["scenarioName"], "datas")
    # scenarioCorePathDataAlgoName = os.path.join(scenario["scenarioPath"], scenario["name"], scenario['simul']['n_instance'], scenarioName, "datas", scenario["execution_parameters"]["algoName"])
    scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], scenario["name"], scenario['simul']['n_instance'], scenario["scenarioName"], "datas", "dataViz")
    scenarioCorePathDataResult = os.path.join(scenario["scenarioPath"], scenario["name"], scenario['simul']['n_instance'], scenario["scenarioName"], "datas", "dataResult")
    scenario["scenarioCorePath"] = scenarioCorePath
    scenario["scenarioCorePathData"] = scenarioCorePathData
    # scenario["scenarioCorePathDataAlgoName"] = scenarioCorePathDataAlgoName
    scenario["scenarioCorePathDataViz"] = scenarioCorePathDataViz
    scenario['scenarioCorePathDataResult'] = scenarioCorePathDataResult
    
    
    # create a scenarioPath if not exists
    Path(scenarioCorePathData).mkdir(parents=True, exist_ok=True)
    # Path(scenarioCorePathDataAlgoName).mkdir(parents=True, exist_ok=True)
    Path(scenarioCorePathDataViz).mkdir(parents=True, exist_ok=True)
    Path(scenarioCorePathDataResult).mkdir(parents=True, exist_ok=True)
    
    
    return scenario

def Initialization_game(scenario: dict):
    """
    initialization of variables of an object application 

    Parameters
    ----------
    scenario : dict
        DESCRIPTION.

    Returns
    -------
    Apps.

    """
    
    # Load all scenario parameters
    transitionprobabilities = scenario["simul"]["transitionprobabilities"]
    repartition = scenario["simul"]["repartition"]
    values = scenario["simul"]["values"]
    probabilities = scenario["simul"]["probabilities"]
    
    
    
    
    # Configuration of the instance generator
    path_name = os.path.join(scenario["scenarioCorePath"], scenario['scenarioName']+".pkl")
    
    g = None
    # Is there the data file on repository?
    checkfile = os.path.isfile(path_name)
    if checkfile:
        print("**** load pickle data : START ****")
        with open(os.path.join(scenario["scenarioCorePath"], scenario["scenarioName"]+".pkl"), "rb") as f:  # open a text file
            print(f" file = { os.path.join(scenario['scenarioCorePath'], scenario['scenarioName']+'.pkl') } ")
            g = pickle.load(f)
        f.close()
        print("**** load pickle data : END ****")
    else:
        g = igV2.Instancegenaratorversion2(N=scenario["instance"]["N_actors"],
                                           T=scenario["simul"]["nbPeriod"] 
                                           )
        print("**** Create pickle data : START ****")
        
        Path(scenario["scenarioCorePath"]).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(scenario["scenarioCorePath"], scenario['scenarioName']+'.pkl'), 'wb') as f:  # open a text file
            pickle.dump(g, f)
        f.close()
        with open(os.path.join(scenario["scenarioCorePath"], scenario['scenarioName']+'.json') , 'w') as fp:
            json.dump(scenario, fp, sort_keys=True, indent=4)
        print("**** Create pickle data : FIN ****")
        
    
    g.generate_dataset_from_automateOrDebug(transitionprobabilities, repartition,
                                            values, probabilities, scenario)
    
        
    return g

        
def load_dataset_to_application(g, scenario):
    """
    

    Parameters
    ----------
    g : TYPE
        DESCRIPTION.
    scenario : dict
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    algoName = scenario["execution_parameters"]['algoName']
    
    
    # Load all scenario parameters
    N_actors = scenario["instance"]["N_actors"]
    nbPeriod = scenario["simul"]["nbPeriod"]
    
    rho = scenario["execution_parameters"]["rho"]
    mu = scenario["execution_parameters"]["mu"]
    learning_rate = scenario["execution_parameters"]["learning_rate"] 
    
    coef_phiepoplus = scenario["simul"]["coef_phiepoplus"]
    coef_phiepominus = scenario["simul"]["coef_phiepominus"]
    
    
    initialprob = scenario["algo"]["LRI_REPART"]["initialprob"] if algoName in "LRI_REPART" else 0
    maxstep = scenario["algo"]["LRI_REPART"]["maxstep"] if algoName in "LRI_REPART" else 0
    maxstep_init = scenario["algo"]["LRI_REPART"]["maxstep_init"] if algoName in "LRI_REPART" else 0
    threshold = scenario["algo"]["LRI_REPART"]["threshold"] if algoName in "LRI_REPART" else 0
    
    # Initialisation of the apps
    application = apps.App(N_actors=N_actors, maxstep=maxstep, mu=mu, 
                           b=learning_rate, 
                           maxstep_init=maxstep_init, threshold=threshold)
    
    application.SG = sg.Smartgrid(N=N_actors, nbperiod=nbPeriod, 
                                  initialprob=initialprob, rho=rho,
                                  coef_phiepoplus=coef_phiepoplus, 
                                  coef_phiepominus=coef_phiepominus)
    
    # if algoName.upper() in 'LRI_REPART' or algoName.upper() in 'SSA':
    #     application.SG = sg.Smartgrid(N=N_actors, nbperiod=nbPeriod, 
    #                                   initialprob=initialprob, rho=rho,
    #                                   coef_phiepoplus=coef_phiepoplus, 
    #                                   coef_phiepominus=coef_phiepominus)
    # else:
    #     # algoName = {CSA, SyA}
    #     application.SG = sg.Smartgrid(N=N_actors, nbperiod=nbPeriod, 
    #                                   initialprob=initialprob, rho=rho,
    #                                   coef_phiepoplus=coef_phiepoplus, 
    #                                   coef_phiepominus=coef_phiepominus)
    
    # Initialisation of production, consumption and storage using the instance generator
    for i in range(N_actors):
        for t in range(nbPeriod+rho):
            application.SG.prosumers[i].production[t] = g.production[i][t]
            application.SG.prosumers[i].consumption[t] = g.consumption[i][t]
            application.SG.prosumers[i].storage[t] = g.storage[i][t]
            application.SG.prosumers[i].smax = g.storage_max[i][t]
            
            if algoName == "Bestie" \
                and eval(scenario.get("simul").get("dataset").get("debug").get("activate")) == True \
                and scenario.get("simul").get("dataset").get("debug").get("strategies_bestie") is not None :
                #print(f"** 4 **")
                state = None
                if scenario.get("simul").get("dataset").get("debug").get("strategies_bestie")\
                    .get("t_"+str(t))\
                    .get("a_"+str(i)).get("state") == "Deficit":
                        state = ag.State.DEFICIT
                elif scenario.get("simul").get("dataset").get("debug").get("strategies_bestie")\
                    .get("t_"+str(t))\
                    .get("a_"+str(i)).get("state") == "Surplus":
                        state = ag.State.SURPLUS
                else:
                    state = ag.State.SELF
                    
                mode = None
                if scenario.get("simul").get("dataset").get("debug").get("strategies_bestie")\
                    .get("t_"+str(t))\
                    .get("a_"+str(i)).get("mode") == "CONS+":
                        mode = ag.Mode.CONSPLUS
                elif scenario.get("simul").get("dataset").get("debug").get("strategies_bestie")\
                    .get("t_"+str(t))\
                    .get("a_"+str(i)).get("mode") == "CONS-":
                        mode = ag.Mode.CONSMINUS
                elif scenario.get("simul").get("dataset").get("debug").get("strategies_bestie")\
                    .get("t_"+str(t))\
                    .get("a_"+str(i)).get("mode") == "DIS":
                        mode = ag.Mode.DIS
                else:
                    mode = ag.Mode.PROD
                
                application.SG.prosumers[i].state[t] = state
                application.SG.prosumers[i].mode[t] = mode   
                
 
    return application

#------------------------------------------------------------------------------
#                FIN : Generer des donnees selon scenarios
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                DEBUT : manipulation des dataframes resultats
#------------------------------------------------------------------------------
def save_df(algoName, rho, mu, learning_rate, M_exec_lri, epsilon, lambda_poisson,
            scenarioCorePathDataAlgoNameParam, scenarioCorePathDataResult):
    """
    insert new columns into dataframe and save it to folder data_result

    Returns
    -------
    None.

    """
    # into new columns into dataframe 
    df_res = None
    if "LRI" not in algoName: 
        df_res = pd.read_csv( os.path.join( scenarioCorePathDataAlgoNameParam, f"run_{algoName}_MergeDF.csv" ) )
    else:
        df_res = pd.read_csv( os.path.join( scenarioCorePathDataAlgoNameParam, f"run_{algoName}_DF_T_Kmax.csv" ) )
    df_res.loc[:,"algoName"] = algoName
    df_res.loc[:,"rho"] = rho
    df_res.loc[:,"mu"] = mu
    df_res.loc[:,"learning_rate"] = learning_rate
    df_res.loc[:,"M_exec_lri"] = M_exec_lri
    df_res.loc[:,"epsilon"] = epsilon
    df_res.loc[:,"lambda_poisson"] = lambda_poisson
    # df_res.loc[:,"Unnamed: 0"] = "prosumers"
    df_res.rename(columns={"Unnamed: 0": "prosumers"}, inplace=True)
    
    # save dataframe into data_result folder
    df_res.to_csv( 
        os.path.join(scenarioCorePathDataResult, 
                 f"df_res_{algoName}_rho{rho}_mu{mu}_lr{learning_rate}_epsilon{epsilon}_MexecLri{M_exec_lri}.csv") )
    
    
def merge_dataframes_in_folder(folder_path, folder_path_2_save):
    # Liste des fichiers CSV dans le dossier
    csv_files = glob.glob(folder_path + "/*.csv")
    
    # Liste pour stocker les DataFrames
    dataframes = []
    
    # Charger chaque fichier CSV en DataFrame
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    # Merger les DataFrames
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df.to_csv( os.path.join(folder_path_2_save, "dataframes.csv") )
        return merged_df
    else:
        print("Aucun fichier CSV trouvé dans le dossier.")
        return None

#------------------------------------------------------------------------------
#                FIN : manipulation des dataframes resultats
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                DEBUT :  algo run with loading instance
#------------------------------------------------------------------------------
def run_algos(scenario: dict, logfiletxt: str):
    """
    run syA, SSA, CSA, Bestie and LRI algorithm

    Parameters
    ----------
    scenario: dict:
        contains list of parameters
    logfile : txt
        path Logs file 
        
    """
    scenarioName = f"{scenario['scenarioName']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"
    scenario["scenarioName"] = scenarioName
    
    
    scenario = create_repo_for_save_jobs(scenario)
    
    
    # # Initialisation of the apps
    g = Initialization_game(scenario)
    
    
    rhos = scenario["simul"]["rhos"]
    mus = scenario["simul"]["mus"]
    learning_rates = scenario["algo"]["LRI_REPART"]["learning_rates"]
    epsilons = scenario["simul"]["epsilons"]
    lambda_poissons = scenario["simul"]["lambda_poissons"]
    M_exec_lri_s = range(0,scenario["simul"]["M_execution_LRI"])
    algoNames = list(scenario["algo"].keys())
    for (algoName, rho, mu, epsilon, lambda_poisson, learning_rate, M_exec_lri) \
        in it.product(algoNames, rhos, mus, epsilons, lambda_poissons, learning_rates, M_exec_lri_s):
        
        start = time.time()
        
        execution_parameters = dict()
        execution_parameters["algoName"] = algoName
        execution_parameters["rho"] = rho
        execution_parameters["mu"] = mu
        execution_parameters["epsilon"] = epsilon
        execution_parameters["lambda_poisson"] = lambda_poisson
        execution_parameters["learning_rate"] = learning_rate
        execution_parameters["M_exec_lri"] = M_exec_lri
        scenario["execution_parameters"] = execution_parameters
        
        scenarioCorePathDataAlgoName = os.path.join(scenario["scenarioPath"], scenarioName, "datas", scenario["execution_parameters"]["algoName"])
        scenario["scenarioCorePathDataAlgoName"] = scenarioCorePathDataAlgoName
        
        
        logfileTempsTuple = os.path.join(scenario["scenarioCorePath"], "tempsExecutionParTuple.txt")
        fileTemps = io.open(logfileTempsTuple,"a")
        
        # # # Initialisation of the apps
        # g = Initialization_game(scenario)
        
        if algoName == "Bestie" and scenario["simul"]["dataset"].get("debug").get("activate") == True:
            g.generate_strategies_for_bestie(scenario)
        
        
        application = load_dataset_to_application(g, scenario)
        
        
        scenarioCorePathDataAlgoNameMuRhoLri = None
        
        if algoName == "CSA":
            scenarioCorePathDataAlgoNameMuRhoLri = os.path.join( scenario["scenarioCorePathDataAlgoName"], "mu_"+str(mu) )
            scenario["execution_parameters"]["scenarioCorePathDataAlgoNameMuRhoLri"] = scenarioCorePathDataAlgoNameMuRhoLri
            Path(scenarioCorePathDataAlgoNameMuRhoLri).mkdir(parents=True, exist_ok=True)
            
            # Display for the run beginning 
            logfile = os.path.join(scenarioCorePathDataAlgoNameMuRhoLri, algoName+"_"+logfiletxt)
            file = io.open(logfile,"w")                                                # Logs file
            monitoring_before_algorithm(file, application)
            file.write("\n_______{algoName}_______"+ "\n")
            
            # run algo
            application.runCSA(plot=False, file=file, scenario=scenario)
            
            # save dataframe
            save_df(algoName=algoName, rho=rho, mu=mu, learning_rate=learning_rate, M_exec_lri=M_exec_lri, 
                    epsilon=epsilon, lambda_poisson=lambda_poisson,
                    scenarioCorePathDataAlgoNameMuRhoLri=scenarioCorePathDataAlgoNameMuRhoLri, 
                    scenarioCorePathDataResult=scenario['scenarioCorePathDataResult'])
        elif algoName == "SSA":
            scenarioCorePathDataAlgoNameMuRhoLri = os.path.join( scenario["scenarioCorePathDataAlgoName"], "mu_"+str(mu), "rho_"+str(rho) )
            scenario["execution_parameters"]["scenarioCorePathDataAlgoNameMuRhoLri"] = scenarioCorePathDataAlgoNameMuRhoLri
            Path(scenarioCorePathDataAlgoNameMuRhoLri).mkdir(parents=True, exist_ok=True)
            
            # Display for the run beginning 
            logfile = os.path.join(scenarioCorePathDataAlgoNameMuRhoLri, algoName+"_"+logfiletxt)
            file = io.open(logfile,"w")                                                # Logs file
            monitoring_before_algorithm(file, application)
            file.write("\n_______{algoName}_______"+ "\n")
            
            # run algo
            application.runSSA(plot=False, file=file, scenario=scenario)
            
            # save dataframe
            save_df(algoName=algoName, rho=rho, mu=mu, learning_rate=learning_rate, M_exec_lri=M_exec_lri, 
                    epsilon=epsilon, lambda_poisson=lambda_poisson,
                    scenarioCorePathDataAlgoNameMuRhoLri=scenarioCorePathDataAlgoNameMuRhoLri, 
                    scenarioCorePathDataResult=scenario['scenarioCorePathDataResult'])
        elif algoName == "SyA":
            scenarioCorePathDataAlgoNameMuRhoLri = os.path.join( scenario["scenarioCorePathDataAlgoName"], "mu_"+str(mu) )
            scenario["execution_parameters"]["scenarioCorePathDataAlgoNameMuRhoLri"] = scenarioCorePathDataAlgoNameMuRhoLri
            Path(scenarioCorePathDataAlgoNameMuRhoLri).mkdir(parents=True, exist_ok=True)
            
            # Display for the run beginning 
            logfile = os.path.join(scenarioCorePathDataAlgoNameMuRhoLri, algoName+"_"+logfiletxt)
            file = io.open(logfile,"w")                                                # Logs file
            monitoring_before_algorithm(file, application)
            file.write("\n_______{algoName}_______"+ "\n")
            
            # run algo
            application.runSyA(plot=False, file=file, scenario=scenario)
            
            # save dataframe
            save_df(algoName=algoName, rho=rho, mu=mu, learning_rate=learning_rate, M_exec_lri=M_exec_lri, 
                    epsilon=epsilon, lambda_poisson=lambda_poisson,
                    scenarioCorePathDataAlgoNameMuRhoLri=scenarioCorePathDataAlgoNameMuRhoLri, 
                    scenarioCorePathDataResult=scenario['scenarioCorePathDataResult'])
        elif algoName == "Bestie":
            scenarioCorePathDataAlgoNameMuRhoLri = os.path.join( scenario["scenarioCorePathDataAlgoName"], "mu_"+str(mu), "rho_"+str(rho) )
            scenario["execution_parameters"]["scenarioCorePathDataAlgoNameMuRhoLri"] = scenarioCorePathDataAlgoNameMuRhoLri
            Path(scenarioCorePathDataAlgoNameMuRhoLri).mkdir(parents=True, exist_ok=True)
            
            # Display for the run beginning 
            logfile = os.path.join(scenarioCorePathDataAlgoNameMuRhoLri, algoName+"_"+logfiletxt)
            file = io.open(logfile,"w")                                                # Logs file
            monitoring_before_algorithm(file, application)
            file.write("\n_______{algoName}_______"+ "\n")
            
            # run algo
            application.runBestie(plot=False, file=file, scenario=scenario)
            
            # save dataframe
            save_df(algoName=algoName, rho=rho, mu=mu, learning_rate=learning_rate, M_exec_lri=M_exec_lri, 
                    epsilon=epsilon, lambda_poisson=lambda_poisson,
                    scenarioCorePathDataAlgoNameMuRhoLri=scenarioCorePathDataAlgoNameMuRhoLri, 
                    scenarioCorePathDataResult=scenario['scenarioCorePathDataResult'])
        else:
            # LRI_REPART
            scenarioCorePathDataAlgoNameMuRhoLri = os.path.join( scenario["scenarioCorePathDataAlgoName"], "mu_"+str(mu), "rho_"+str(rho), "lr_"+str(learning_rate), "M_exec_lri_"+str(M_exec_lri) )
            scenario["execution_parameters"]["scenarioCorePathDataAlgoNameMuRhoLri"] = scenarioCorePathDataAlgoNameMuRhoLri
            Path(scenarioCorePathDataAlgoNameMuRhoLri).mkdir(parents=True, exist_ok=True)
            
            # Display for the run beginning 
            logfile = os.path.join(scenarioCorePathDataAlgoNameMuRhoLri, algoName+"_"+logfiletxt)
            file = io.open(logfile,"w")                                                # Logs file
            monitoring_before_algorithm(file, application)
            file.write("\n_______{algoName}_______"+ "\n")
            
            # run algo
            application.runLRI_REPART(plot=False, file=file, scenario=scenario, algoName=algoName)
            
            # save dataframe
            save_df(algoName=algoName, rho=rho, mu=mu, learning_rate=learning_rate, M_exec_lri=M_exec_lri, 
                    epsilon=epsilon, lambda_poisson=lambda_poisson,
                    scenarioCorePathDataAlgoNameMuRhoLri=scenarioCorePathDataAlgoNameMuRhoLri, 
                    scenarioCorePathDataResult=scenario['scenarioCorePathDataResult'])
        
        monitoring_after_algorithm(algoName=algoName, file=file, application=application)
        
        # Save application to Pickle format
        with open(os.path.join(scenarioCorePathDataAlgoNameMuRhoLri, scenarioName+"_"+algoName+"_APP"+'.pkl'), 'wb') as f:  # open a text file
            pickle.dump(application, f)
        f.close()
        
        
        fileTemps.write(f"{algoName}, mu={mu}, rho={rho}, epsilon={epsilon}, lambda_poisson={lambda_poisson}, lr={learning_rate}, M_exec_lri={M_exec_lri} ==> *** runtime= { round(time.time() - start, 4)} *** \n")
        fileTemps.close()
    
    # merge all dataframe from dataResult folder into one dataframe
    df = merge_dataframes_in_folder(folder_path=scenario['scenarioCorePathDataResult'], 
                               folder_path_2_save=scenario["scenarioCorePathDataViz"])
    
    
def run_algos_count_prodCartesien(scenario: dict, logfiletxt: str):
    """
    run syA, SSA, CSA, Bestie and LRI algorithm for folder creation and count all execution

    Parameters
    ----------
    scenario : TYPE, optional
        DESCRIPTION. The default is scenario.
    logfiletxt : TYPE, optional
        DESCRIPTION. The default is logfiletxt.

    Returns
    -------
    None.

    """
    scenarioName = f"{scenario['name']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"
    scenario["scenarioName"] = scenarioName
    
    
    scenario = create_repo_for_save_jobs(scenario)
    
    
    # # Initialisation of the apps
    g = Initialization_game(scenario)
    
    
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
    
    compteur = 0
    cols=["n_instance", "algoName", "rho", "mu", "epsilon", "lambda_poisson", "learning_rate", "M_exec_lri", "ValSG", "QTStock"]
    df_ValSG_QTstock = pd.DataFrame(columns=cols)
    for (algoName, rho, mu, epsilon, lambda_poisson, learning_rate, M_exec_lri) in prod_cart_NoLRI_LRI:
        
        start = time.time()
        
        execution_parameters = dict()
        execution_parameters["algoName"] = algoName
        execution_parameters["n_instance"] = scenario["simul"]["n_instance"]
        execution_parameters["rho"] = rho
        execution_parameters["mu"] = mu
        execution_parameters["epsilon"] = epsilon
        execution_parameters["lambda_poisson"] = lambda_poisson
        execution_parameters["learning_rate"] = learning_rate
        execution_parameters["M_exec_lri"] = M_exec_lri
        scenario["execution_parameters"] = execution_parameters
        
        scenarioCorePathDataAlgoName = os.path.join(scenario["scenarioCorePathData"], scenario["execution_parameters"]["algoName"])
        scenario["scenarioCorePathDataAlgoName"] = scenarioCorePathDataAlgoName
        
        
        logfileTempsTuple = os.path.join(scenario["scenarioCorePath"], "tempsExecutionParTuple.txt")
        fileTemps = io.open(logfileTempsTuple,"a")
        
        # # # Initialisation of the apps
        # g = Initialization_game(scenario)
        
        if algoName == "Bestie" and eval(scenario["simul"]["dataset"].get("debug").get("activate")) == True:
            g.generate_strategies_for_bestie(scenario)
        
        
        application = load_dataset_to_application(g, scenario)
        
        scenarioCorePathDataAlgoNameParam = None
        if algoName.find("LRI") != -1:
            scenarioCorePathDataAlgoNameParam \
                = os.path.join( scenario["scenarioCorePathDataAlgoName"], "mu_"+str(mu), "rho_"+str(rho), "epsilon_"+str(epsilon), 
                                "lambda_"+str(lambda_poisson), "lr_"+str(learning_rate), "M_exec_lri_"+str(M_exec_lri) )
        elif algoName.find("SSA") != -1 or algoName.find("Bestie") != -1:
            scenarioCorePathDataAlgoNameParam \
                = os.path.join( scenario["scenarioCorePathDataAlgoName"], "mu_"+str(mu), "rho_"+str(rho), "epsilon_"+str(epsilon), 
                                "lambda_"+str(lambda_poisson), "M_exec_lri_"+str(M_exec_lri) )
        elif algoName.find("CSA") != -1 or algoName.find("SyA") != -1:
            scenarioCorePathDataAlgoNameParam \
                = os.path.join( scenario["scenarioCorePathDataAlgoName"], "mu_"+str(mu), "epsilon_"+str(epsilon), 
                                "lambda_"+str(lambda_poisson), "M_exec_lri_"+str(M_exec_lri) )
                
        scenario["execution_parameters"]["scenarioCorePathDataAlgoNameParam"] = scenarioCorePathDataAlgoNameParam
        Path(scenarioCorePathDataAlgoNameParam).mkdir(parents=True, exist_ok=True)
                
        # Display for the run beginning 
        logfile = os.path.join(scenarioCorePathDataAlgoNameParam, algoName+"_"+logfiletxt)
        file = io.open(logfile,"w")                                                # Logs file
        monitoring_before_algorithm(file, application)
        file.write("\n_______{algoName}_______"+ "\n")
        
        # run algo
        if algoName.find("LRI") != -1:
            application.runLRI_REPART(plot=False, file=file, scenario=scenario, algoName=algoName)
            pass
        elif algoName.find("SSA") != -1:
            application.runSSA(plot=False, file=file, scenario=scenario)
            pass
        elif algoName.find("Bestie") != -1:
            application.runBestie(plot=False, file=file, scenario=scenario)
            pass
        elif algoName.find("CSA") != -1:
            application.runCSA(plot=False, file=file, scenario=scenario)
            pass
        elif algoName.find("SyA") != -1:
            application.runSyA(plot=False, file=file, scenario=scenario)
            pass
        
        # compute ValSG and QTStock
        execution_parameters["ValSG"] = application.valSG_A
        execution_parameters["QTStock"] = application.QTStock_A
        df_ValSG_QTstock.loc[len(df_ValSG_QTstock)] = execution_parameters
        
        
        # save dataframe
        save_df(algoName=algoName, rho=rho, mu=mu, learning_rate=learning_rate, M_exec_lri=M_exec_lri, 
                epsilon=epsilon, lambda_poisson=lambda_poisson,
                scenarioCorePathDataAlgoNameParam=scenarioCorePathDataAlgoNameParam, 
                scenarioCorePathDataResult=scenario['scenarioCorePathDataResult'])
    
        monitoring_after_algorithm(algoName=algoName, file=file, application=application)
        
        # Save application to Pickle format
        with open(os.path.join(scenarioCorePathDataAlgoNameParam, scenarioName+"_"+algoName+"_APP"+'.pkl'), 'wb') as f:  # open a text file
            pickle.dump(application, f)
        f.close()
        
        
        fileTemps.write(f"{algoName}, mu={mu}, rho={rho}, epsilon={epsilon}, lambda_poisson={lambda_poisson}, lr={learning_rate}, M_exec_lri={M_exec_lri} ==> *** runtime= { round(time.time() - start, 4)} *** \n")
        fileTemps.close()
        
        compteur += 1
        
    
    print(f"compteur = {compteur}")
    resExecNinstances = os.path.join(scenario["scenarioPath"], scenario["name"], "resultExecNinstances")
    Path(resExecNinstances).mkdir(parents=True, exist_ok=True)
    df_ValSG_QTstock.to_csv(os.path.join(resExecNinstances,
                                         f"dataframe_ValSG_QTStock_{scenario['simul']['n_instance']}.csv") )
    
#------------------------------------------------------------------------------
#                FIN :  algo run with loading instance
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                DEBUT :  algo run with ONE loading instance
#------------------------------------------------------------------------------
def run_algos_one_instance(scenario:dict, logfiletxt:str, algoName:str, rho:int, 
                           mu:float, epsilon:float, lambda_poisson: float, 
                           learning_rate:float, M_exec_lri:int):
    
    # start = time.time()
    
    scenarioName = f"{scenario['scenarioName']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"
    scenario["scenarioName"] = scenarioName
    
    
    scenario = create_repo_for_save_jobs(scenario)
    
    
    # # Initialisation of the apps
    g = Initialization_game(scenario)
    
    
    # start = time.time()
    
    execution_parameters = dict()
    execution_parameters["algoName"] = algoName
    execution_parameters["rho"] = rho
    execution_parameters["mu"] = mu
    execution_parameters["epsilon"] = epsilon
    execution_parameters["lambda_poisson"] = lambda_poisson
    execution_parameters["learning_rate"] = learning_rate
    execution_parameters["M_exec_lri"] = M_exec_lri
    scenario["execution_parameters"] = execution_parameters
    
    scenarioCorePathDataAlgoName = os.path.join(scenario["scenarioPath"], scenarioName, "datas", scenario["execution_parameters"]["algoName"])
    scenario["scenarioCorePathDataAlgoName"] = scenarioCorePathDataAlgoName
    
    
    logfileTempsTuple = os.path.join(scenario["scenarioCorePath"], "tempsExecutionParTuple.txt")
    fileTemps = io.open(logfileTempsTuple,"a")
    
    # # # Initialisation of the apps
    # g = Initialization_game(scenario)
    
    if algoName == "Bestie" and scenario["simul"]["dataset"].get("debug").get("activate") == True:
        g.generate_strategies_for_bestie(scenario)
    
    
    application = load_dataset_to_application(g, scenario)
    
    scenarioCorePathDataAlgoNameParam = None
    if algoName.find("LRI") != -1:
        scenarioCorePathDataAlgoNameParam \
            = os.path.join( scenario["scenarioCorePathDataAlgoName"], "mu_"+str(mu), "rho_"+str(rho), "epsilon_"+str(epsilon), 
                            "lambda_"+str(lambda_poisson), "lr_"+str(learning_rate), "M_exec_lri_"+str(M_exec_lri) )
    elif algoName.find("SSA") != -1 or algoName.find("Bestie") != -1:
        scenarioCorePathDataAlgoNameParam \
            = os.path.join( scenario["scenarioCorePathDataAlgoName"], "mu_"+str(mu), "rho_"+str(rho), "epsilon_"+str(epsilon), 
                            "lambda_"+str(lambda_poisson), "M_exec_lri_"+str(M_exec_lri) )
    elif algoName.find("CSA") != -1 or algoName.find("SyA") != -1:
        scenarioCorePathDataAlgoNameParam \
            = os.path.join( scenario["scenarioCorePathDataAlgoName"], "mu_"+str(mu), "epsilon_"+str(epsilon), 
                            "lambda_"+str(lambda_poisson), "M_exec_lri_"+str(M_exec_lri) )
            
    scenario["execution_parameters"]["scenarioCorePathDataAlgoNameParam"] = scenarioCorePathDataAlgoNameParam
    Path(scenarioCorePathDataAlgoNameParam).mkdir(parents=True, exist_ok=True)
            
    # Display for the run beginning 
    logfile = os.path.join(scenarioCorePathDataAlgoNameParam, algoName+"_"+logfiletxt)
    file = io.open(logfile,"w")                                                # Logs file
    monitoring_before_algorithm(file, application)
    file.write("\n_______{algoName}_______"+ "\n")
    
    # run algo
    if algoName.find("LRI") != -1:
        application.runLRI_REPART(plot=False, file=file, scenario=scenario, algoName=algoName)
        pass
    elif algoName.find("SSA") != -1:
        application.runSSA(plot=False, file=file, scenario=scenario)
        pass
    elif algoName.find("Bestie") != -1:
        application.runBestie(plot=False, file=file, scenario=scenario)
        pass
    elif algoName.find("CSA") != -1:
        application.runCSA(plot=False, file=file, scenario=scenario)
        pass
    elif algoName.find("SyA") != -1:
        application.runSyA(plot=False, file=file, scenario=scenario)
        pass
    
    # save dataframe
    save_df(algoName=algoName, rho=rho, mu=mu, learning_rate=learning_rate, M_exec_lri=M_exec_lri, 
            epsilon=epsilon, lambda_poisson=lambda_poisson,
            scenarioCorePathDataAlgoNameParam=scenarioCorePathDataAlgoNameParam, 
            scenarioCorePathDataResult=scenario['scenarioCorePathDataResult'])

    monitoring_after_algorithm(algoName=algoName, file=file, application=application)
    
    # Save application to Pickle format
    with open(os.path.join(scenarioCorePathDataAlgoNameParam, scenarioName+"_"+algoName+"_APP"+'.pkl'), 'wb') as f:  # open a text file
        pickle.dump(application, f)
    f.close()
    
    
    # fileTemps.write(f"""{algoName}, mu={mu}, rho={rho}, epsilon={epsilon}, 
    #                 lambda_poisson={lambda_poisson}, lr={learning_rate}, 
    #                 M_exec_lri={M_exec_lri} ==> *** runtime= { round(time.time() - start, 4)} *** \n""")
    fileTemps.write(f"""{algoName}, mu={mu}, rho={rho}, epsilon={epsilon}, lambda_poisson={lambda_poisson}, lr={learning_rate}, M_exec_lri={M_exec_lri} ==> *** runtime = *** \n""")
    fileTemps.close()
    
    
#------------------------------------------------------------------------------
#                FIN :  algo run with ONE loading instance
#------------------------------------------------------------------------------


if __name__ == '__main__':

    logfiletxt = "traceApplication.txt"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate.json"
    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate_test.json"
    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate_rho5_mu001.json"
    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate_test_MP.json"
    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomateMorePeriods.json"
    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate100Periods.json"
    
    # scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate50Periods.json"
    
    # scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate150Periods.json"
    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate50PeriodsMultipleParams.json"
    
    import time
    start = time.time()
    with open(scenarioFile) as file:
        scenario = json.load(file)
        
        # run_algos(scenario=scenario, logfiletxt=logfiletxt)
        
        run_algos_count_prodCartesien(scenario=scenario, logfiletxt=logfiletxt)
        
        # if "SyA" in scenario["algo"]:
        #     run_syA(scenario, logfiletxt)
        # if "SSA" in scenario["algo"]:
        #     # run_SSA(scenario, logfiletxt)
        #     pass
        # if "CSA" in scenario["algo"]:
        #     # run_CSA(scenario, logfiletxt)
        #     pass
        # if "LRI_REPART" in scenario["algo"]:
        #     # run_LRI_REPART(scenario, logfiletxt )
        #     pass
        pass

    print(f"Running time = {time.time() - start}")
    
#------------------------------------------------------------------------------
#                FIN :  algo run with loading instance
#------------------------------------------------------------------------------