#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 09:00:33 2025

Modified from 09/06/2024 and 15/03/2025

@author: Quentin
"""
import os
import json
import pickle
import numpy as np
import random as rdm

from pathlib import Path


COEF_PERIOD_GEN = 3 # number of max periods to generate  max_period = COEF_PERIOD_GEN * T

class Instancegenaratorv2:
    '''
    This is the third version of the instance generator used to generate 
    production and consumption values for prosumers
    '''
    
    production = None
    consumption = None
    storage = None
    storage_max = None
    situation = None
    laststate = None
    
    def __init__(self, N:int, T:int):
        """
        
        We generate N actors and max_period periods with 
         max_period = COEF_PERIOD_GEN * T

        Parameters
        ----------
        N : int
            number of prosumers
        T : int
            number of periods .
        # rho : int
        #     the next periods to add at T periods. In finish, we generate (T + rho) periods.
        #     This parameter enables the prediction of values from T+1 to T + rho periods
        #     rho << T ie rho=3 < T=5

        Returns
        -------
        None.

        """
        max_period = COEF_PERIOD_GEN*T
        
        self.production = np.zeros((N, max_period))
        self.consumption = np.zeros((N, max_period))
        self.storage = np.zeros((N, max_period))
        self.storage_max = np.zeros((N, max_period))
        self.situation = np.zeros((N, max_period))
        self.laststate = np.ones((N, 3))
        
    def generate_dataset_from_automateOrDebug(self, transitionprobabilities, repartition, values, probabilities, scenario):
        """
        Generate dataset from automate with values in scenario file
        
        
        transitionprobabilities = probabilities of transition from A to B1, B1 to A, B2 to C, C to B2
        repartition = repartition between the two groups of situation {A,B1} and {B2,C}
        values : matrix containing the ranges of values used in each situation dedicated generator
        values[0] = [m1a,M1a]
        values[1] = [m1b,M1b,m2b,M2b,cb]
        values[2] = [m1c,M1c,m2c,M2c,m3c,M3c,m4c,M4c]
        probabilities : matrix containing probabilities for changing from one state to another inside the two state Markov chains B,C1,C2
        probabilities[0] = [P1b,P2b]
        probabilities[1] = [P1c,P2c,P3c,P4c]
        
        
        """
        
        # Initial random repartition between situation A(1), B1(2), B2(3) and C(4)
        for prosumer_i in range(repartition[0]):
            self.situation[prosumer_i][0] = rdm.randint(1,2)
        for prosumer_i in range(repartition[1]):
            self.situation[repartition[0] + prosumer_i][0] = rdm.randint(3,4)
            
        for prosumer_i in range(self.production.shape[0]):
            
            for period in range(self.production.shape[1]):
                
                self.storage_max[prosumer_i][period] = scenario.get('instance').get('smax')
                
                if eval(scenario["simul"]["dataset"]["automate"]["activate"]) == True:
                    self.insert_random_PCvalues_4_prosumers(
                        prosumer_i=prosumer_i, 
                        period=period, 
                        transitionprobabilities=transitionprobabilities, 
                        repartition=repartition, 
                        values=values, 
                        probabilities=probabilities)
                elif eval(scenario["simul"]["dataset"]["debug"]["activate"]) == True:
                    version = scenario["simul"]["dataset"]["debug"]["version"] 
                    
                    if version == "20092024":
                        self.dataset_debug_version20092024(prosumer_i, period)
                    elif version == "01072024":
                        self.dataset_debug_version01072024(prosumer_i, period)
                    else:
                        self.dataset_debug_versionRandom(prosumer_i, period)
                    
        
    def insert_random_PCvalues_4_prosumers(self, prosumer_i, period, transitionprobabilities, repartition, values, probabilities):
        """
        allocate random values of production and consumption for prosumers according to 
        transitionprobabilities, repartition, probabilities
        """
        if self.situation[prosumer_i][period] == 1 :
            
            # Set production and consumption for the period j
            self.production[prosumer_i][period] = 0
            self.consumption[prosumer_i][period] = rdm.randint(values[0][0],values[0][1])
            
            # Define situation for next period
            if period < self.production.shape[1]-1 :
                roll = rdm.uniform(0,1)
                if roll < 1 - transitionprobabilities[0] :
                    self.situation[prosumer_i][period+1] = 1
                
                else :
                    self.situation[prosumer_i][period+1] = 2
                    self.laststate[prosumer_i][0] = 1
            
        elif self.situation[prosumer_i][period] == 2 :
            
            # Set consumption for period j
            self.consumption[prosumer_i][period] = values[1][4]
            
            # Set production for period j
            if self.laststate[prosumer_i][0] == 1:
                if rdm.uniform(0,1) <= probabilities[0][0]:
                    self.laststate[prosumer_i][0] = 2    
                
                self.production[prosumer_i][period] = rdm.randint(values[1][0],values[1][1])
            
            else:
                if rdm.uniform(0,1) <= probabilities[0][1]:
                    self.laststate[prosumer_i][0] = 1  
                
                self.production[prosumer_i][period] = rdm.randint(values[1][2],values[1][3])
            
            # Define situation for next period
            if period < self.production.shape[1]-1 :
                roll = rdm.uniform(0,1)
                if roll < 1 - transitionprobabilities[1] :
                    self.situation[prosumer_i][period+1] = 1
                
                else :
                    self.situation[prosumer_i][period+1] = 2
                
        elif self.situation[prosumer_i][period] == 3 :
            
            # Set consumption for period j
            self.consumption[prosumer_i][period] = values[1][4]
            
            # Set production for period j
            if self.laststate[prosumer_i][0] == 1:
                if rdm.uniform(0,1) <= probabilities[0][0]:
                    self.laststate[prosumer_i][0] = 2  
                    
                self.production[prosumer_i][period] = rdm.randint(values[1][0],values[1][1])
            
            else:
                if rdm.uniform(0,1) <= probabilities[0][1]:
                    self.laststate[prosumer_i][0] = 1  
                self.production[prosumer_i][period] = rdm.randint(values[1][2],values[1][3])
           
            # Define situation for next period
            if period < self.production.shape[1]-1 :
                roll = rdm.uniform(0,1)
                
                if roll < 1 - transitionprobabilities[2]:
                    self.situation[prosumer_i][period+1] = 3
                
                else :
                    self.situation[prosumer_i][period+1] = 4
                    self.laststate[prosumer_i][1] = 1
                    self.laststate[prosumer_i][2] = 1
        
        else :
            self.production[prosumer_i][period] = rdm.randint(values[2][2],values[2][3])
            self.consumption[prosumer_i][period] = rdm.randint(values[2][4],values[2][5])
                                                 
            # Define situation for next period
            if period < self.production.shape[1]-1 :
                roll = rdm.uniform(0,1)
                if roll < 1 - transitionprobabilities[3] :
                    self.situation[prosumer_i][period+1] = 3
                    self.laststate[prosumer_i][0] = 2
                else :
                    self.situation[prosumer_i][period+1] = 4
        # pass
        
        # sprint(f"23 t={period}, prosumer_{prosumer_i}: P={self.production[prosumer_i][period]}, C={self.consumption[prosumer_i][period]}")
        pass
    
    
    
    def dataset_debug_version20092024(self, prosumer_i, period):
        """
                
        generate data from overleaf version of 20/09/2024
        this version contains new a version of stock value prediction with 
        variables SP, cal_G, Help.
        
        N=8, T=20, rho=5
        each player has data of T+rho periods
        
        we have 2 groups of actors : GA and PA
        Ga have production = 10, consumption = 4 and storage_max = 6
        PA have production = 0, consumption = 6 and storage_max = 2

        Parameters
        ----------
        prosumer_i : int
            DESCRIPTION.
        period : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if prosumer_i<4 and period<10:
            self.consumption[prosumer_i][period] = 4
            self.production[prosumer_i][period] = 10
            self.storage_max[prosumer_i][period] = 6
        elif prosumer_i<4 and period>=10:
            self.consumption[prosumer_i][period] = 4
            self.production[prosumer_i][period] = 1
            self.storage_max[prosumer_i][period] = 6
        elif prosumer_i>=4 and period<5:
            self.consumption[prosumer_i][period] = 6
            self.production[prosumer_i][period] = 0
            self.storage_max[prosumer_i][period] = 2
        elif prosumer_i>=4 and period>=5:
            self.consumption[prosumer_i][period] = 4
            self.production[prosumer_i][period] = 0
            self.storage_max[prosumer_i][period] = 2
            
            
    def dataset_debug_version01072024(self, prosumer_i, period):
        """
         version from latex document 01072024

        Parameters
        ----------
        prosumer_i : TYPE
            DESCRIPTION.
        period : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
          
        if prosumer_i < 10 :
            self.consumption[prosumer_i][period] = 10
            if period % 15 < 9 :
                self.production[prosumer_i][period] = 11
            else:
                self.production[prosumer_i][period] = 9

        else:
            self.consumption[prosumer_i][period] = 3
            if period % 15 < 5 :
                self.production[prosumer_i][period] = 3
            else:
                self.production[prosumer_i][period] = 2
    
    def dataset_debug_versionRandom(self, prosumer_i, period):
        
        """
        

        Parameters
        ----------
        prosumer_i : TYPE
            DESCRIPTION.
        period : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.consumption[prosumer_i][period] = np.random.randint(low=0, high=20)
        self.production[prosumer_i][period] = np.random.randint(low=0, high=11)
        
    def generate_strategies_for_bestie(self, scenario):
        """
        generate strategies for bestie algorithm

        Returns
        -------
        None.

        """
        # generate consumption and production for maxPeriod periods
        if scenario.get("simul").get("dataset").get("debug").get("strategie_bestie") is None:
            pass
        else:
            for prosumer_i in range(self.production.shape[0]):
                for period in range(self.production.shape[1]):
                    self.consumption[prosumer_i][period] = scenario.get("simul")\
                                                            .get("dataset").get("debug")\
                                                            .get("strategies_bestie")\
                                                            .get("t_"+str(period))\
                                                            .get("a_"+str(prosumer_i))\
                                                            .get("C")
                    
                    self.production[prosumer_i][period] = scenario.get("simul")\
                                                            .get("dataset").get("debug")\
                                                            .get("strategies_bestie")\
                                                            .get("t_"+str(period))\
                                                            .get("a_"+str(prosumer_i))\
                                                            .get("P")
                                                
                    self.storage[prosumer_i][period] = scenario.get("simul")\
                                                        .get("dataset").get("debug")\
                                                        .get("strategies_bestie")\
                                                        .get("t_"+str(period))\
                                                        .get("a_"+str(prosumer_i))\
                                                        .get("S")
                    self.storage_max[prosumer_i][period] = scenario.get("simul")\
                                                            .get("dataset").get("debug")\
                                                            .get("strategies_bestie")\
                                                            .get("t_"+str(period))\
                                                            .get("a_"+str(prosumer_i))\
                                                            .get("Smax")
                    pass
        
    
if __name__ == "__main__":
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate.json"
    
    with open(scenarioFile) as file:
        scenario = json.load(file)
        
    g = Instancegenaratorv2(N=scenario["instance"]["N_actors"],
                            T=scenario["simul"]["nbPeriod"] 
                            )
    
    
    #repartition = [5,5]
    repartition = scenario["simul"]["repartition"]
    #values = [m1a,M1a,m1b,M1b,m2b,M2b,cb,m1c,M1c,m2c,M2c,m3c,M3c,m4c,M4c]
    values = scenario["simul"]["values"]
    #probabilities = [P1b,P2b,P1c,P2c,P3c,P4c]
    probabilities = scenario["simul"]["probabilities"]
    transitionprobabilities = scenario["simul"]["transitionprobabilities"]
    
    g.generate_dataset_from_automateOrDebug(transitionprobabilities, repartition, values, probabilities, scenario)
    
    # folder creation
    scenarioCorePath = os.path.join(scenario["scenarioPath"], scenario["scenarioName"])
    scenarioName = f"{scenario['scenarioName']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"
    Path(scenarioCorePath).mkdir(parents=True, exist_ok=True)
    
    # with open(os.path.join(scenarioCorePath, scenario["scenarioName"]+'.pkl'), 'wb') as f:  # open a text file
    with open(os.path.join(scenarioCorePath, scenarioName+'.pkl'), 'wb') as f:  # open a text file
        pickle.dump(g, f)
    f.close()
    
    g.generate_strategies_for_bestie(scenario)
    
    # with open(os.path.join(scenarioCorePath, scenario["scenarioName"]+'.json') , 'w') as fp:
    with open(os.path.join(scenarioCorePath, scenarioName+'.json') , 'w') as fp:
        json.dump(scenario, fp, sort_keys=True, indent=4)
