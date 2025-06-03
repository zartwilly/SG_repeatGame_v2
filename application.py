#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:48:57 2024

@author: willy

application is the environment of the repeted game
"""
import typing
import copy
import numpy as np
import pandas as pd
import agents as ag
import smartgrid as sg
import itertools as it
import json, io, os

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class App:
    
    """
    this class is used to call the various algorithms
    """
    
    SG = None # Smartgrid
    N_actors = None  # number of actors
    nbperiod = None # max number of periods for a game.
    maxstep = None # Number of learning steps
    maxstep_init = None # Number of learning steps for initialisation Max, Min Prices. in this step, no strategies' policies are updated
    threshold = None
    mu = None # To define
    b = None # learning rate / Slowdown factor LRI
    ObjSG = None # Objective value for a SG over all periods for LRI
    Obj_ai = None # Objective value for each actor over all periods for LRI
    ObjValNoSG_ai = None
    valNoSG_A = None # sum of prices payed by all actors during all periods by running algo A without SG
    valSG_A = None # sum of prices payed by all actors during all periods by running algo A with SG
    valNoSGCost_A = None # 
    QTStock_A = None # sum of all prosumers' QTStock for all periods 
    dicoLRI_onePeriod_oneStep = None # a dictionnary to save a running for one period and one step
    Qttepo_plus = None     # compute quantity prosumers must give to EPO
    Qttepo_minus = None     # compute quantity prosumers must receive to EPO
    X_ai = None             # percent of reduction each actor must pay or receive money
    Y_ai = None             # Sum over periods of quantity of production prosumer a_i give to SG
    NE_ai = None            # Nash equilibrium for each prosumer
    # NE_brute_t = None       # Nash equilibrium Brute : all prosumers have NE_ai = 1 for all prosumers
    
    def __init__(self, N_actors, maxstep, mu, b, maxstep_init, threshold):
        self.maxstep = maxstep
        self.N_actors = N_actors
        self.mu = mu
        self.b = b
        self.maxstep_init = maxstep_init
        self.threshold = threshold
        self.ObjSG = 0
        self.Obj_ai = np.zeros(N_actors)
        self.ObjValNoSG_ai = np.zeros(N_actors)
        self.valNoSG_A = 0
        self.valSG_A = 0
        self.QTStock_A = 0
        self.dicoLRI_onePeriod_oneStep = dict()
        self.Qttepo_plus = 0
        self.Qttepo_minus = 0
        self.X_ai = np.zeros(N_actors)
        self.Y_ai = np.zeros(N_actors)
        self.NE_ai = np.zeros(N_actors)
        # self.NE_brute_t = 0
        
        
    def computeObj_ai(self):
        """
        Compute the fitness value of actors at the end of game
        
        Parameters
        ----------
        application : APP
            an instance of an application
        """
                
        for i in range(self.SG.prosumers.size):
            sumObjai = 0
            sumObjai = np.sum(self.SG.prosumers[i].price)
            self.Obj_ai[i] = sumObjai
            
    def computeObjValNoSG_ai(self):
        """
        compute ValNoSG for each prosumer ai over the periods 

        Returns
        -------
        None.

        """
        for i in range(self.SG.prosumers.size):
            sumObjValNoSG_ai = 0
            sumObjValNoSG_ai = np.sum(self.SG.prosumers[i].valNoSG)
            self.ObjValNoSG_ai[i] = sumObjValNoSG_ai
        
    def computeX_ai(self):
        """
        percent of reduction each actor must pay or receive money

        Returns
        -------
        None.

        """
        for i in range(self.SG.prosumers.size):
            sumX_ai = 0
            sumX_ai = round(1 - self.Obj_ai[i]/self.ObjValNoSG_ai[i], 2)
            self.X_ai[i] = sumX_ai
            
    def computeY_ai(self):
        """
        Sum over periods of production quantity that prosumer a_i gives to SG

        Returns
        -------
        None.

        """
        for i in range(self.SG.prosumers.size):
            sumY_ai = 0
            sumY_ai = np.sum(self.SG.prosumers[i].prodit)
            self.Y_ai[i] = sumY_ai
            
    def computeObjSG(self):
        """
        Compute the fitness value of the SG grid at the end of game
        
        Parameters
        ----------
        N_actors : int
            an instance of time t
        """
        
        sumValSG = np.sum(self.SG.ValSG)
        self.ObjSG = sumValSG
        
    def computeValSG(self):
        """
        

        Parameters
        ----------
        application : APP
            an instance of an application
        """
        self.valSG_A = np.sum(self.SG.ValSG)
        
    def computeValNoSG(self):
        """
        

        Parameters
        ----------
        application : APP
            an instance of an application
        """
        self.valNoSG_A = np.sum(self.SG.ValNoSG)
        
    def computeValNoSGCost_A(self):
        """
        

        Returns
        -------
        None.

        """
        self.valNoSGCost_A = np.sum(self.SG.ValNoSGCost)
        
    def computeQttepo(self):
        """
        compute Qttepoplus and Qttepominus 

        Returns
        -------
        None.

        """
        self.Qttepo_minus = np.sum(self.SG.QttEpo_minus)
        self.Qttepo_plus = np.sum(self.SG.QttEpo_plus)
        
    def computeSumQTStock(self):
        """
        compute the sum of QTStock for all prosumers for all periods

        Returns
        -------
        None.

        """
        self.QTStock_A = np.sum(self.SG.QTStock_t)
        
        
        
    ######### ----------------   LRI START ------------------------------------
    def save_LRI_2_json_onePeriod_oneStep(self, period, step, algoName, scenarioName):
        """
        save data from LRI execution for one period 

        Parameters
        ----------
        period : int, 
            DESCRIPTION. The default is t.
        step: int
            one epoch/step for learning

        Returns
        -------
        None.

        """
        N = self.N_actors
        insg = self.SG.insg[period]
        outsg = self.SG.outsg[period]
        Qttepo_minus_t = self.SG.QttEpo_minus[period]
        Qttepo_plus_t = self.SG.QttEpo_plus[period]
        ValEgoc = self.SG.ValEgoc[period]
        ValNoSG = self.SG.ValNoSG[period]
        ValSG = self.SG.ValSG[period]
        Reduct = self.SG.Reduct[period]
        LCostmax = self.SG.LCostmax[period]
        LCostmin = self.SG.LCostmin[period]
        Cost = self.SG.Cost[period]
        
        # GNeeds = dict()
        # for h, elt in enumerate(self.SG.GNeeds[period]):
        #     GNeeds["GNeeds_h="+str(h)] = elt
        # GPd = dict()
        # for h, elt in enumerate(self.SG.GPd[period]):
        #     GPd["GPd_h="+str(h)] = elt
        
        # tauS = dict()
        # for i, elt in enumerate(self.SG.TauS):
        #     tauS["Prosum="+str(i)] = elt
            
        GridNeeds = dict()
        for h, elt in enumerate(self.SG.GridNeeds[period]):
            GridNeeds["GridNeeds_h="+str(h)] = elt
            
        Free = dict()
        for h, elt in enumerate(self.SG.Free[period]):
            Free["Free_h="+str(h)] = elt
            
        coef_phiepoplus = self.SG.coef_phiepoplus
        coef_phiepominus = self.SG.coef_phiepominus
        
        #dicoLRI_onePeriod_oneStep = dict()
        for i in range(N):
            production = self.SG.prosumers[i].production[period]
            consumption = self.SG.prosumers[i].consumption[period]
            storage = self.SG.prosumers[i].storage[period]
            prodit = self.SG.prosumers[i].prodit[period]
            consit = self.SG.prosumers[i].consit[period]
            mode = self.SG.prosumers[i].mode[period]
            state = self.SG.prosumers[i].state[period]
            prmode0 = self.SG.prosumers[i].prmode[period][0]
            prmode1 = self.SG.prosumers[i].prmode[period][1]
            utility = self.SG.prosumers[i].utility[period]
            price = self.SG.prosumers[i].price[period]
            valOne_i = self.SG.prosumers[i].valOne[period]
            valNoSG_i = self.SG.prosumers[i].valNoSG[period]
            valStock_i = self.SG.prosumers[i].valStock[period]
            Repart_i = self.SG.prosumers[i].Repart[period]
            cost = self.SG.prosumers[i].cost[period]
            Lcost = self.SG.prosumers[i].Lcost[period]
            LCostmax = self.SG.prosumers[i].LCostmax["Lcost"]
            LCostmin = self.SG.prosumers[i].LCostmin["Lcost"]
                        
            
            #tau = self.SG.prosumers[i].tau
            
            storage_t_plus_1 = self.SG.prosumers[i].storage[period+1]
            
            
            Needs = dict()
            for h, elt in enumerate(self.SG.prosumers[i].Needs[period]):
                Needs["Needs_h="+str(h)] = elt
                
            # Provs = dict()
            # for h, elt in enumerate(self.SG.prosumers[i].Provs[period]):
            #     Provs["Provs_h="+str(h)] = elt
                
            # i_tense = dict()
            # for h, elt in enumerate(self.SG.prosumers[i].i_tense[period]):
            #     i_tense["itense_h="+str(h)] = elt
            
            Help = dict()
            for h, elt in enumerate(self.SG.prosumers[i].Help[period]):
                Help["Help_h="+str(h)] = elt
                
            tau_plus = dict()
            for h, elt in enumerate(self.SG.prosumers[i].tau_plus[period]):
                tau_plus["tau_plus_h="+str(h)] = elt
             
            tau_minus = dict()
            for h, elt in enumerate(self.SG.prosumers[i].tau_minus[period]):
                tau_minus["tau_minus_h="+str(h)] = elt
            
            SP = dict()
            for h, elt in enumerate(self.SG.prosumers[i].SP[period]):
                SP["SP_h="+str(h)] = elt
                
            Rq = dict()
            for h, elt in enumerate(self.SG.prosumers[i].Rq[period]):
                Rq["Rq_h="+str(h)] = elt
                
            Val = dict()
            for h, elt in enumerate(self.SG.prosumers[i].Val[period]):
                Val["Val_h="+str(h)] = elt
            
            
            
            self.dicoLRI_onePeriod_oneStep["prosumer"+str(i)] = {
                "prosumers": "prosumer"+str(i),
                "period": period,
                "step": step,
                "production":production,
                "consumption": consumption,
                "storage": storage,
                "storaget+1": storage_t_plus_1,
                "prodit": prodit,
                "consit": consit,
                "mode": str(mode),
                "state": str(state),
                "prmode0": prmode0,
                "prmode1": prmode1,
                "utility": utility,
                "price": price,
                "valOne_i": valOne_i,
                "valNoSG_i":valNoSG_i,
                "Repart_i": Repart_i,
                
                "cost": cost,
                "Lcost": Lcost,
                "insg": insg,
                "outsg": outsg,
                "QttepoPlus_t": Qttepo_plus_t,
                "QttepoMinus_t": Qttepo_minus_t,
                "ValEgoc": ValEgoc,
                "ValNoSG": ValNoSG,
                "ValSG": ValSG,
                "Reduct": Reduct,
                "LCostmax": LCostmax,
                "Lcostmax": self.SG.prosumers[i].Lcostmax[period],
                "LCostmin": LCostmin,
                "Lcostmin": self.SG.prosumers[i].Lcostmin[period], 
                "Cost": Cost,
                
                #"tau": str(tau[period]),
                #"tauS": str(tauS),
                "tau_plus": str(tau_plus),
                "tau_minus": str(tau_minus),
                
                "SP": str(SP),
                "gamma": self.SG.prosumers[i].gamma[period],
                
                "Needs": str(Needs),
                "GridNeeds": str(GridNeeds),
                "Free": str(Free),
                "Help": str(Help),
                "Rq": str(Rq),
                "Val": str(Val),
                
                "mu": self.SG.prosumers[i].mu[period],
                #"GNeeds": str(GNeeds),
                
                #"GPd": str(GPd),
                "Smax": self.SG.prosumers[i].smax,
                #"Provs": str(Provs),
                
                #"Min_K": str(self.SG.prosumers[i].Min_K[period]),
                #"i-tense": str(i_tense),
                "EstimStock": self.SG.prosumers[i].EstimStock[period],
                "poisson_random_value": self.SG.prosumers[i].poisson_random_value[period], 
                "QTStock": self.SG.prosumers[i].QTStock[period], 
                
                
                "Si": storage,
                "S_t+1": storage_t_plus_1,
                "valStock_i":valStock_i,
                
                "algoName": algoName, 
                "scenarioName": scenarioName,
                "coef_phiepoplus": coef_phiepoplus,
                "coef_phiepominus": coef_phiepominus
                }
        pass
    
    
    def run_LRI_4_onePeriodT_oneStepK(self, period:int, epsilon:float, 
                                      lambda_poisson: float, boolInitMinMax:bool,
                                      strategieStateDeficit:bool):
        """
        

        Parameters
        ----------
        period : int
            an instance of time t.
            
        epsilon: float
            a prediction error rate
            
        lambda_poisson: float
            a parameter for the generation on random value from Poisson Distribution
            
        boolInitMinMax : Bool
            require the game whether LRI probabilities strategies are updated or not.
            if True, LRI probabilities are not updated otherwise.
            This part is used for initializing the min and max values

        Returns
        -------
        None.

        """
        # Update prosumers' modes following LRI mode selection
        self.SG.updateModeLRI(period, self.threshold, strategieStateDeficit=strategieStateDeficit)
        
        # Update prodit, consit and period + 1 storage values
        self.SG.updateSmartgrid(period)
        
        # Calculate inSG and outSG
        self.SG.computeSumInput(period)
        self.SG.computeSumOutput(period)
        
        # Calculate Qttepo_t^{plus,minus}
        self.SG.computeQttepo_plus(period)
        self.SG.computeQttepo_minus(period)
    
        ## compute what each actor has to paid/gain at period t 
        ## Calculate ValNoSGCost, ValEgo, ValNoSG, ValSG, Reduct, Repart
        ## ------ start ------
        
        # calculate valNoSGCost_t
        self.SG.computeValNoSGCost(period)
        
        # calculate valEgoc_t
        self.SG.computeValEgoc(period)
        
        # calculate valNoSG_t
        self.SG.computeValNoSG(period)
        
        # calculate ValSG_t
        self.SG.computeValSG(period)
        
        # calculate Reduct_t
        self.SG.computeReduct(period)
        
        # calculate repart_t
        self.SG.computeRepart(period, mu=self.mu)
        
        # calculate price_t
        self.SG.computePrice(period)
        
        # # compute PC_CP_th for all prosumers at a period t
        # for i in range(self.N_actors):
        #     self.SG.prosumers[i].computePC_CP_th(period=period, nbperiod=self.SG.nbperiod, rho=self.rho)
        
        # # calculate tau for all prosumers at a period t
        # self.SG.computeTau_actors(period, rho=self.rho)
            
        # # calculate DispSG
        # for h in range(1, self.rho):
        #     self.SG.computeDispSG(period, h=h)
            
        # # calculate High_t, Low_t
        # self.SG.computeHighLow(period)
        
        # # calculate rs_{high,low}_{minus,plus}_t
        # self.SG.compute_RS_highPlus(period)
        # self.SG.compute_RS_highMinus(period)
        # self.SG.compute_RS_lowPlus(period)
        # self.SG.compute_RS_lowMinus(period)
        
        # # calculate Taus
        # self.SG.computeTaus(period) 
        
        # # calculate Needs
        # self.SG.computeNeeds(period)
        
        # # Calculate Provs for all h with 1 <= h <= rho and identifies h i-tense
        # self.SG.computeProvsforRho(period) 
        
        # # calculate QTStock
        # self.SG.ComputeQTStock(period, epsilon=scenario["execution_parameters"]["epsilon"])
        
        # # calculate ValStock
        # self.SG.computeValStock(period)
        
        # calculate tau_minus_plus
        self.SG.computeTauMinusPlus4Prosumers(period)
        
        # calculate SP
        self.SG.computeSP4Prosumers(period)
        
        # calculate Gamma
        self.SG.computeGamma4prosumers(period)
        
        # calculate Needs and GridNeeds
        self.SG.computeGridNeeds4Prosumers(period)
        
        # calculate Free
        self.SG.computeFree4Prosumers(period)
        
        # calculate Help
        self.SG.computeHelp4Prosumers(period)
        
        # calculate Mu
        # self.SG.computeMu4Prosummers(period)
        
        # compute Rq
        self.SG.computeRq(period)
        
        # compute Val
        self.SG.computeVal(period)
        
        # compute EstimStock
        self.SG.computeEstimStock(period)
        
        # generate Poisson ramdom value
        self.SG.computePoissonRandomValue(period, lambda_poisson=lambda_poisson)
        
        # calculate QTStock
        self.SG.computeQTStock(period, epsilon=epsilon)
        
        # calculate ValStock
        self.SG.computeValStock(period)
        
        # calculate learning indicator I(ai,s)
        self.SG.computeLearningIndicator(period)
        
        
        ## ------ end ------
        
        # Compute(Update) min/max Learning cost (LearningCost) for prosumers
        self.SG.computeLCost_LCostMinMax(period)
        
        # boolInitMinMax == False, we update probabilities (prmod) of prosumers strategies
        if not boolInitMinMax:
            # Calculate utility
            self.SG.computeUtility(period)
            
            # Update probabilities for choosing modes
            self.SG.updateProbaLRI(period, self.b)
        
        pass
    
        
    def runLRI_REPART(self, plot, file, scenario, algoName):
        """
        Run LRI algorithm with the repeated game
        
        Parameters
        ----------
        plot: bool
            yes if you want a figure of some variables
        file : TextIO
            file to save some informations of runtime
        scenario: dict
            DESCRIPTION
        algoName: txt
            name of the algorithm

        Returns
        -------
        None.

        """
        K = self.maxstep
        T = self.SG.nbperiod
        L = self.maxstep_init
        
        df_T_Kmax = []
        for t in range(T):
                        
            # Update the state of each prosumer
            self.SG.updateState(t)
            
            # Initialization game of min/max Learning cost (LearningCost) for prosumers
            for l in range(L):
                self.run_LRI_4_onePeriodT_oneStepK(period=t, 
                                                   epsilon=scenario["execution_parameters"]["epsilon"], 
                                                   lambda_poisson=scenario["execution_parameters"]["lambda_poisson"],
                                                   boolInitMinMax=True, 
                                                   strategieStateDeficit=scenario["simul"]["strategieStateDeficit"])
        
            # --- START Game with learning steps
            dicoLRI_onePeriod_KStep = dict()
            df_t_K = []
            datas = []
            for k in range(K):
                print(f"t = {t}, k={k}") if k%(K//5) == 0 else None
                self.dicoLRI_onePeriod_oneStep = dict()
                self.run_LRI_4_onePeriodT_oneStepK(period=t, 
                                                   epsilon=scenario["execution_parameters"]["epsilon"], 
                                                   lambda_poisson=scenario["execution_parameters"]["lambda_poisson"],
                                                   boolInitMinMax=False, 
                                                   strategieStateDeficit=scenario["simul"]["strategieStateDeficit"])
                
                self.save_LRI_2_json_onePeriod_oneStep(period=t, step=k, 
                                                        algoName=algoName, 
                                                        scenarioName=scenario["scenarioName"])
                
                
                dicoLRI_onePeriod_KStep["step_"+str(k)] = self.dicoLRI_onePeriod_oneStep
                
                datas.append(self.dicoLRI_onePeriod_oneStep)
                
                df_tk = pd.DataFrame.from_dict(self.dicoLRI_onePeriod_oneStep, orient="index")
                df_t_K.append(df_tk)
            
            
            ## ---> start : save execution to json file
            try:
                to_unicode = unicode
            except NameError:
                to_unicode = str
            # Write JSON file
            # with io.open(os.path.join(scenario["execution_parameters"]["scenarioCorePathDataAlgoNameParam"], f'runLRI_t={t}.json'), 'w', encoding='utf8') as outfile:
            #     str_ = json.dumps(dicoLRI_onePeriod_KStep,
            #                       indent=4, sort_keys=True,
            #                       #separators=(',', ': '), 
            #                       ensure_ascii=False
            #                       )
            #     outfile.write(to_unicode(str_))
            ## ---> end : save execution to json file
            
            ## ---> start : save execution to JSON file by data over period
            # with io.open(os.path.join(scenario["execution_parameters"]["scenarioCorePathDataAlgoNameParam"], f'runDataLRI_t={t}.json'), 'w', encoding='utf8') as outfile:
            #     str_ = json.dumps(datas,
            #                       indent=4, sort_keys=True,
            #                       #separators=(',', ': '), 
            #                       ensure_ascii=False, 
            #                       cls=NpEncoder
            #                       )
            #     outfile.write(to_unicode(str_))
            ## ---> end : save execution to JSON file by data over period
            
            ## ---> start : save execution to CSV file by data over period
            # merge list of dataframes of K steps to one dataframe 
            df_t_K = pd.concat(df_t_K, axis=0)
            runLRI_sumUp_K_txt = f"run_LRI_df_t_{t}.csv"
            df_t_K.to_csv(os.path.join(scenario["execution_parameters"]["scenarioCorePathDataAlgoNameParam"], runLRI_sumUp_K_txt)) \
                if scenario["simul"]["save_period_dataframe"] == True else None
            
            df_t_kmax = df_t_K[df_t_K['step']==df_t_K['step'].max()].copy(deep=True)
            df_T_Kmax.append(df_t_kmax)
                        
            ## ---> end : save execution to CSV file by data over period
            
            # --- END Game with learning steps
            print(f"t={t} termine")
            
            # ---> start : is nash equilibrium got in the period 
            self.isNashEquilibrium(period=t, epsilon=scenario["execution_parameters"]["epsilon"], file=file)
            # ---> end : is nash equilibrium got in the period 
         
        ## ---> start : save execution to CSV file Over periods with the k max steps
        df_T_Kmax = pd.concat(df_T_Kmax, axis=0)
        runLRI_sumUp_K_txt = f"run_{algoName}_DF_T_Kmax.csv"
        df_T_Kmax.to_csv(os.path.join(scenario["execution_parameters"]["scenarioCorePathDataAlgoNameParam"], runLRI_sumUp_K_txt))
        ## ---> end : save execution to CSV file Over periods with the k max steps
        
        # df_NE_brute = pd.DataFrame(self.SG.NE_brute_t).reset_index()
        # df_NE_brute.columns = ["period","NE_brute"]
        # df_NE_brute.to_csv(os.path.join(scenario["execution_parameters"]["scenarioCorePathDataAlgoNameParam"], "NE_brute.csv"))
        
        df_NE_brute = pd.DataFrame(self.SG.NE_brute_t).reset_index()
        prosumers = [f'prosumer{i}' for i in range(self.SG.prosumers.size)]
        prosumers.insert(0, "period")
        df_NE_brute.columns = prosumers
        df_NE_brute.to_csv(os.path.join(scenario["execution_parameters"]["scenarioCorePathDataAlgoNameParam"], "NE_brute.csv"))
        
         
        ## Compute metrics
        self.computeValSG()
        self.computeValNoSG()
        self.computeObj_ai()
        self.computeObjValNoSG_ai()
        self.computeX_ai()
        self.computeY_ai()
        self.computeObjSG()
        self.computeValNoSGCost_A()
        self.computeSumQTStock()
        
        #self.saveMetricsOfGame()
        # save all metrics to dataframe with columns obj_ai, objNoSG_ai, X_ai, valNoSG_ai
        
        file.write("___Threshold___ \n")

        ## show prmode for each prosumer at all period and determined when the threshold has been reached
        N = self.SG.prosumers.size
        for Ni in range(N):
            file.write(f"Prosumer = {Ni} \n")
            for t in range(T):
                if (self.SG.prosumers[Ni].prmode[t][0] < self.threshold and \
                    (self.SG.prosumers[Ni].prmode[t][1]) < self.threshold):
                    file.write("Period " + str(t) + " : "+ str(self.SG.prosumers[Ni].prmode[t][0])+" < threshold="+ str(self.threshold) + "\n")
                elif (self.SG.prosumers[Ni].prmode[t][0] >= self.threshold and \
                    (self.SG.prosumers[Ni].prmode[t][1]) < self.threshold):
                    file.write("Period " + str(t) + " : "+ str(self.SG.prosumers[Ni].prmode[t][0])+" >= threshold="+ str(self.threshold) + " ==>  prob[0] #### \n")
                elif (self.SG.prosumers[Ni].prmode[t][0] < self.threshold and \
                    (self.SG.prosumers[Ni].prmode[t][1]) >= self.threshold):
                    file.write("Period " + str(t) + " : "+ str(self.SG.prosumers[Ni].prmode[t][1])+" >= threshold="+ str(self.threshold) + " ==> prob[1] #### \n")
                else:
                    file.write("Period " + str(t) + " : "+ str(self.SG.prosumers[Ni].prmode[t][1])+" >= threshold="+ str(self.threshold) + " ==> prob[1] #### \n")
                
                    
        
        # Determines for each period if it attained a Nash equilibrium and if not if one exist
        file.write("___Nash___ : NOT DEFINE \n")
        
        
        
      
    ######### -------------------  LRI END ------------------------------------
            
                
    ######### -------------------  SyA START ----------------------------------
    def create_dico_for_onePeriod(self, algoName:str, period:int):
        """
        

        Parameters
        ----------
        period : int
            DESCRIPTION.

        Returns
        -------
        dico_onePeriod : dict
            DESCRIPTION.

        """
        
        GridNeeds = dict()
        for h, elt in enumerate(self.SG.GridNeeds[period]):
            GridNeeds["GridNeeds_h="+str(h)] = elt
            
        Free = dict()
        for h, elt in enumerate(self.SG.Free[period]):
            Free["Free_h="+str(h)] = elt
        
        
        dico_onePeriod = dict()
        for i in range(self.N_actors):
            
            storage_t_plus_1 = self.SG.prosumers[i].storage[period+1]
            
            Needs, Help = dict(), dict()
            tau_plus, tau_minus = dict(), dict()
            SP, Rq, Val = dict(), dict(), dict()
            
            if algoName == "SSA":
                for h, elt in enumerate(self.SG.prosumers[i].Needs[period]):
                    Needs["Needs_h="+str(h)] = elt
                    
                for h, elt in enumerate(self.SG.prosumers[i].Help[period]):
                    Help["Help_h="+str(h)] = elt
                    
                for h, elt in enumerate(self.SG.prosumers[i].tau_plus[period]):
                    tau_plus["tau_plus_h="+str(h)] = elt
                 
                for h, elt in enumerate(self.SG.prosumers[i].tau_minus[period]):
                    tau_minus["tau_minus_h="+str(h)] = elt
                
                for h, elt in enumerate(self.SG.prosumers[i].SP[period]):
                    SP["SP_h="+str(h)] = elt  
                
                for h, elt in enumerate(self.SG.prosumers[i].Rq[period]):
                    Rq["Rq_h="+str(h)] = elt
                    
                for h, elt in enumerate(self.SG.prosumers[i].Val[period]):
                    Val["Val_h="+str(h)] = elt
            
            dico_onePeriod["prosumer"+str(i)] = {
                "period": period,
                "production": self.SG.prosumers[i].production[period],
                "consumption": self.SG.prosumers[i].consumption[period],
                "storage": self.SG.prosumers[i].storage[period],
                "storaget+1": self.SG.prosumers[i].storage[period+1],
                "Smax": self.SG.prosumers[i].smax,
                "prodit": self.SG.prosumers[i].prodit[period],
                "consit": self.SG.prosumers[i].consit[period],
                "mode": str(self.SG.prosumers[i].mode[period]),
                "state": str(self.SG.prosumers[i].state[period]),
                "prmode0": self.SG.prosumers[i].prmode[period][0],
                "prmode1": self.SG.prosumers[i].prmode[period][1],
                "utility": self.SG.prosumers[i].utility[period],
                "price": self.SG.prosumers[i].price[period],
                "Cost": self.SG.Cost[period],
                "valOne_i": self.SG.prosumers[i].valOne[period],
                "valNoSG_i": self.SG.prosumers[i].valNoSG[period],
                "Repart_i": self.SG.prosumers[i].Repart[period],
                "cost": self.SG.prosumers[i].cost[period],
                "Lcost": self.SG.prosumers[i].Lcost[period],
                "insg": self.SG.insg[period],
                "outsg": self.SG.outsg[period],
                "QttepoPlus_t": self.SG.QttEpo_plus[period],
                "QttepoMinus_t": self.SG.QttEpo_minus[period],
                "ValEgoc": self.SG.ValEgoc[period],
                "ValNoSG": self.SG.ValNoSG[period],
                "ValSG": self.SG.ValSG[period],
                
                "tau_plus": str(tau_plus),
                "tau_minus": str(tau_minus),
                
                "SP": str(SP),
                "gamma": self.SG.prosumers[i].gamma[period],
                
                "Needs": str(Needs),
                "GridNeeds": str(GridNeeds),
                "Free": str(Free),
                "Help": str(Help),
                "Rq": str(Rq),
                "Val": str(Val),
                
                "EstimStock": self.SG.prosumers[i].EstimStock[period],
                "poisson_random_value": self.SG.prosumers[i].poisson_random_value[period], 
                "QTStock": self.SG.prosumers[i].QTStock[period], 
                
                
                "Si": self.SG.prosumers[i].storage[period],
                "S_t+1": storage_t_plus_1,
                "valStock_i": self.SG.prosumers[i].valStock[period]
                
                }
            
        return dico_onePeriod
    
    def runSyA(self, plot:bool, file:bool, scenario:dict): 
        """
        Run SyA algorithm on the app
        
        Parameters
        ----------
        plot : Boolean
            a boolean determining if the plots are edited or not
        
        file : Boolean
            file used to output logs
            
        scenario: dict
            dictionnary of all parameters for a game
            
        """
        T_periods = self.SG.nbperiod
        
        df_ts = []
        for t in range(T_periods):
            # Update the state of each prosumer
            self.SG.updateState(period=t)
            
            # Update prosumers' modes following SyA mode selection
            self.SG.updateModeSyA(period=t)
            
            # Update prodit,consit and period + 1 storage values
            self.SG.updateSmartgrid(period=t)
            
            ## compute what each actor has to paid/gain at period t 
            ## (ValEgo, ValNoSG, ValSG, reduct, repart, price, ) 
            ## ------ start -------
            # Calculate inSG and outSG
            self.SG.computeSumInput(period=t)
            self.SG.computeSumOutput(period=t)
            
            # Calculate Qttepo_t^{plus,minus}
            self.SG.computeQttepo_plus(period=t)
            self.SG.computeQttepo_minus(period=t)
            
            # calculate valNoSGCost_t
            self.SG.computeValNoSGCost(period=t)
            
            # calculate valEgoc_t
            self.SG.computeValEgoc(period=t)
            
            # calculate valNoSG_t
            self.SG.computeValNoSG(period=t)
            
            # calculate ValSG_t
            self.SG.computeValSG(period=t)
            
            # calculate Reduct_t
            self.SG.computeReduct(period=t)
            
            # calculate repart_t
            self.SG.computeRepart(period=t, mu=self.mu)
            
            # calculate price_t
            self.SG.computePrice(period=t)
            
            dico_onePeriod = dict()
            dico_onePeriod = self.create_dico_for_onePeriod(period=t, 
                                                            algoName=scenario["execution_parameters"]["algoName"])
                
            df_t = pd.DataFrame.from_dict(dico_onePeriod, orient="index")
            df_ts.append(df_t)
            
            ## ------ end -------
            
        # Compute metrics
        self.computeValSG()
        self.computeValNoSG()
        self.computeObj_ai()
        self.computeObjValNoSG_ai()
        self.computeX_ai()
        self.computeY_ai()
        self.computeObjSG()
        self.computeValNoSGCost_A()
        self.computeQttepo()
        
        
        # plot variables ValNoSG, ValSG
        
        # merge list of dataframes to one dataframe
        df = pd.concat(df_ts, axis=0)
        runAlgo_SumUp_txt = "run_SyA_MergeDF.csv"
        df.to_csv(os.path.join(scenario["execution_parameters"]["scenarioCorePathDataAlgoNameParam"], runAlgo_SumUp_txt))
                
                
    ######### -------------------  SyA END ------------------------------------
            
    
    ######### -------------------  SSA START ----------------------------------
    def runSSA(self, plot:bool, file:bool, scenario:dict): 
        """
        Run SSA (selfish Stock Algorithm) algorithm on the app
        
        Parameters
        ----------
        plot : Boolean
            a boolean determining if the plots are edited or not
        
        file : Boolean
            file used to output logs
            
        scenario: dict
            dictionnary of all parameters for a game
        
        """        
        T_periods = self.SG.nbperiod
        
        df_ts = []
        for t in range(T_periods):
            # Update the state of each prosumer
            self.SG.updateState(period=t)
            
            # calculate tau_minus_plus for prosumers
            #self.SG.computeTauMinusPlus4Prosumers(period=t)
            
            # calculate Xi for prosumers
            #self.SG.computeXi4Prosumers(period=t)
            
            ### calculate QTStock  ###
            # calculate tau_minus_plus
            self.SG.computeTauMinusPlus4Prosumers(period=t)
            
            # calculate SP
            self.SG.computeSP4Prosumers(period=t)
            
            # calculate Gamma
            self.SG.computeGamma4prosumers(period=t)
            
            # calculate Needs and GridNeeds
            self.SG.computeGridNeeds4Prosumers(period=t)
            
            # calculate Free
            self.SG.computeFree4Prosumers(period=t)
            
            # calculate Help
            self.SG.computeHelp4Prosumers(period=t)
            
            # compute Rq
            self.SG.computeRq(period=t)
            
            # compute Val
            self.SG.computeVal(period=t)
            
            # compute EstimStock
            self.SG.computeEstimStock(period=t)
            
            # generate Poisson ramdom value
            self.SG.computePoissonRandomValue(period=t, lambda_poisson=scenario["execution_parameters"]["lambda_poisson"])
            
            # calculate QTStock
            self.SG.computeQTStock(period=t, epsilon=scenario["execution_parameters"]["epsilon"])
            
            # calculate ValStock
            self.SG.computeValStock(period=t)
            ### calculate QTStock  ###
            
            # Update prosumers' modes following SyA mode selection
            self.SG.updateModeSSA(period=t)
            
            # Update prodit,consit and period + 1 storage values
            self.SG.updateSmartgrid(period=t)
            
            ## compute what each actor has to paid/gain at period t 
            ## (ValEgo, ValNoSG, ValSG, reduct, repart, price, ) 
            ## ------ start -------
            # Calculate inSG and outSG
            self.SG.computeSumInput(period=t)
            self.SG.computeSumOutput(period=t)
            
            # Calculate Qttepo_t^{plus,minus}
            self.SG.computeQttepo_plus(period=t)
            self.SG.computeQttepo_minus(period=t)
            
            # calculate valNoSGCost_t
            self.SG.computeValNoSGCost(period=t)
            
            # calculate valEgoc_t
            self.SG.computeValEgoc(period=t)
            
            # calculate valNoSG_t
            self.SG.computeValNoSG(period=t)
            
            # calculate ValSG_t
            self.SG.computeValSG(period=t)
            
            # calculate Reduct_t
            self.SG.computeReduct(period=t)
            
            # calculate repart_t
            self.SG.computeRepart(period=t, mu=self.mu)
            
            # calculate price_t
            self.SG.computePrice(period=t)
            
            ## ------ end -------
            
            dico_onePeriod = dict()
            dico_onePeriod = self.create_dico_for_onePeriod(period=t, 
                                                            algoName=scenario["execution_parameters"]["algoName"])
                
            df_t = pd.DataFrame.from_dict(dico_onePeriod, orient="index")
            df_ts.append(df_t)
            
        # Compute metrics
        self.computeValSG()
        self.computeValNoSG()
        self.computeObj_ai()
        self.computeObjValNoSG_ai()
        self.computeX_ai()
        self.computeY_ai()
        self.computeObjSG()
        self.computeValNoSGCost_A()
        self.computeQttepo()
        self.computeSumQTStock()
        
        # plot variables ValNoSG, ValSG
        
        # merge list of dataframes to one dataframe
        df = pd.concat(df_ts, axis=0)
        runAlgo_SumUp_txt = "run_SSA_MergeDF.csv"
        df.to_csv(os.path.join(scenario["execution_parameters"]["scenarioCorePathDataAlgoNameParam"], runAlgo_SumUp_txt))
        
    ######### -------------------  SSA END ------------------------------------

    ######### -------------------  CSA START ----------------------------------
    def runCSA(self, plot:bool, file:bool, scenario:dict): 
        """
        Run CSA (centralised Stock Algorithm) algorithm on the app
        
        Parameters
        ----------
        plot : Boolean
            a boolean determining if the plots are edited or not
        
        file : Boolean
            file used to output logs
            
        scenario: dict
            dictionnary of all parameters for a game
            
        """
        T_periods = self.SG.nbperiod
        
        df_ts = []
        for t in range(T_periods):
            # Update the state of each prosumer
            self.SG.updateState(period=t)
            
            # Update prosumers' modes following SyA mode selection
            self.SG.updateModeCSA(period=t)
            
            # Update prodit,consit and period + 1 storage values
            self.SG.updateSmartgrid(period=t)
            
            ## compute what each actor has to paid/gain at period t 
            ## (ValEgo, ValNoSG, ValSG, reduct, repart, price, ) 
            ## ------ start -------
            # Calculate inSG and outSG
            self.SG.computeSumInput(period=t)
            self.SG.computeSumOutput(period=t)
            
            # Calculate Qttepo_t^{plus,minus}
            self.SG.computeQttepo_plus(period=t)
            self.SG.computeQttepo_minus(period=t)
            
            # calculate valNoSGCost_t
            self.SG.computeValNoSGCost(period=t)
            
            # calculate valEgoc_t
            self.SG.computeValEgoc(period=t)
            
            # calculate valNoSG_t
            self.SG.computeValNoSG(period=t)
            
            # calculate ValSG_t
            self.SG.computeValSG(period=t)
            
            # calculate Reduct_t
            self.SG.computeReduct(period=t)
            
            # calculate repart_t
            self.SG.computeRepart(period=t, mu=self.mu)
            
            # calculate price_t
            self.SG.computePrice(period=t)
            
            ## ------ end -------
            
            dico_onePeriod = dict()
            dico_onePeriod = self.create_dico_for_onePeriod(period=t, 
                                                            algoName=scenario["execution_parameters"]["algoName"])
                
            df_t = pd.DataFrame.from_dict(dico_onePeriod, orient="index")
            df_ts.append(df_t)
            
        # Compute metrics
        self.computeValSG()
        self.computeValNoSG()
        self.computeObj_ai()
        self.computeObjValNoSG_ai()
        self.computeX_ai()
        self.computeY_ai()
        self.computeObjSG()
        self.computeValNoSGCost_A()
        self.computeQttepo()
        
        
        # plot variables ValNoSG, ValSG
    
        # merge list of dataframes to one dataframe
        df = pd.concat(df_ts, axis=0)
        runAlgo_SumUp_txt = "run_CSA_MergeDF.csv"
        df.to_csv(os.path.join(scenario["execution_parameters"]["scenarioCorePathDataAlgoNameParam"], runAlgo_SumUp_txt))
        
    ######### -------------------  CSA END ------------------------------------
    
    ######### -------------------  Bestie START ----------------------------------
    def runBestie(self, plot:bool, file:bool, scenario:dict): 
        """
        Run Bestie algorithm on the app: it tests the "best" algorithm in comparison to LRI, CSA and SSA
        
        Parameters
        ----------
        plot : Boolean
            a boolean determining if the plots are edited or not
        
        file : Boolean
            file used to output logs
            
        scenario: dict
            dictionnary of all parameters for a game
            
        """
        T_periods = self.SG.nbperiod
        
        df_ts = []
        for t in range(T_periods):
            # Update the state of each prosumer
            #self.SG.updateState(period=t)
            
            ### calculate QTStock: START  ###
            # calculate tau_minus_plus
            self.SG.computeTauMinusPlus4Prosumers(period=t)
            
            # calculate SP
            self.SG.computeSP4Prosumers(period=t)
            
            # calculate Gamma
            self.SG.computeGamma4prosumers(period=t)
            
            # calculate Needs and GridNeeds
            self.SG.computeGridNeeds4Prosumers(period=t)
            
            # calculate Free
            self.SG.computeFree4Prosumers(period=t)
            
            # calculate Help
            self.SG.computeHelp4Prosumers(period=t)
            
            # calculate Mu
            # self.SG.computeMu4Prosummers(period=t)
            
            # compute Rq
            self.SG.computeRq(period=t)
            
            # compute Val
            self.SG.computeVal(period=t)
            
            # compute EstimStock
            self.SG.computeEstimStock(period=t)
            
            # generate Poisson ramdom value
            self.SG.computePoissonRandomValue(period=t, lambda_poisson=scenario["execution_parameters"]["lambda_poisson"])
            
            # calculate QTStock
            self.SG.computeQTStock(period=t, epsilon=scenario["execution_parameters"]["epsilon"])
            
            # calculate ValStock
            self.SG.computeValStock(period=t)
            ### calculate QTStock : END ###
            
            # Update prosumers' modes following SyA mode selection
            #self.SG.updateModeCSA(period=t)
            
            # Update prodit,consit and period + 1 storage values
            self.SG.updateSmartgrid(period=t)
            
            ## compute what each actor has to paid/gain at period t 
            ## (ValEgo, ValNoSG, ValSG, reduct, repart, price, ) 
            ## ------ start -------
            # Calculate inSG and outSG
            self.SG.computeSumInput(period=t)
            self.SG.computeSumOutput(period=t)
            
            # Calculate Qttepo_t^{plus,minus}
            self.SG.computeQttepo_plus(period=t)
            self.SG.computeQttepo_minus(period=t)
            
            # calculate valNoSGCost_t
            self.SG.computeValNoSGCost(period=t)
            
            # calculate valEgoc_t
            self.SG.computeValEgoc(period=t)
            
            # calculate valNoSG_t
            self.SG.computeValNoSG(period=t)
            
            # calculate ValSG_t
            self.SG.computeValSG(period=t)
            
            # calculate Reduct_t
            self.SG.computeReduct(period=t)
            
            # calculate repart_t
            self.SG.computeRepart(period=t, mu=self.mu)
            
            # calculate price_t
            self.SG.computePrice(period=t)
            
            ## ------ end -------
            
            dico_onePeriod = dict()
            dico_onePeriod = self.create_dico_for_onePeriod(period=t, 
                                                            algoName=scenario["execution_parameters"]["algoName"])
                
            df_t = pd.DataFrame.from_dict(dico_onePeriod, orient="index")
            df_ts.append(df_t)
            
            # nash equilibrium at one period 
            
            
        # Compute metrics
        self.computeValSG()
        self.computeValNoSG()
        self.computeObj_ai()
        self.computeObjValNoSG_ai()
        self.computeX_ai()
        self.computeY_ai()
        self.computeObjSG()
        self.computeValNoSGCost_A()
        self.computeQttepo()
        
        # plot variables ValNoSG, ValSG
    
        # merge list of dataframes to one dataframe
        df = pd.concat(df_ts, axis=0)
        runAlgo_SumUp_txt = "run_Bestie_MergeDF.csv"
        df.to_csv(os.path.join(scenario["execution_parameters"]["scenarioCorePathDataAlgoNameParam"], runAlgo_SumUp_txt))
    ######### -------------------  Bestie END ------------------------------------
    
    
    
    ###########################################################################
    #                       identify nash equilibruim at one period:: start
    ###########################################################################
    def execute_SG_for_NashEquilibrium(self, period, epsilon, sg1):
        """
        compute all variables with new parameters modified in the isNashEquilibrium function
        
        """
        # Update prodit, consit and period + 1 storage values
        sg1.updateSmartgrid(period)
        
        # Calculate inSG and outSG
        sg1.computeSumInput(period)
        sg1.computeSumOutput(period)
    
        ## compute what each actor has to paid/gain at period t 
        ## Calculate ValNoSGCost, ValEgo, ValNoSG, ValSG, Reduct, Repart
        ##### ------ start ------
        
        # calculate valNoSGCost_t
        sg1.computeValNoSGCost(period)
        
        # calculate valEgoc_t
        sg1.computeValEgoc(period)
        
        # calculate valNoSG_t
        sg1.computeValNoSG(period)
        
        # calculate ValSG_t
        sg1.computeValSG(period)
        
        # calculate Reduct_t
        sg1.computeReduct(period)
        
        # calculate repart_t
        sg1.computeRepart(period, mu=self.mu)
        
        # calculate price_t
        sg1.computePrice(period)
    
        # # calculate Taus
        # sg1.computeTaus(period) 
        
        # # calculate Needs
        # sg1.computeNeeds(period)
        
        # # Calculate Provs for all h with 1 <= h <= rho and identifies h i-tense
        # sg1.computeProvsforRho(period) 
        ###################################################
        # calculate tau_minus_plus
        sg1.computeTauMinusPlus4Prosumers(period)
        
        # calculate SP
        sg1.computeSP4Prosumers(period)
        
        # calculate Gamma
        sg1.computeGamma4prosumers(period)
        
        # calculate Needs and GridNeeds
        sg1.computeGridNeeds4Prosumers(period)
        
        # calculate Free
        sg1.computeFree4Prosumers(period)
        
        # calculate Help
        sg1.computeHelp4Prosumers(period)
        
        # calculate Mu
        # sg1.computeMu4Prosummers(period)
        ###################################################
        
        
        
        # calculate QTStock
        sg1.computeQTStock(period, epsilon=epsilon)
        
        # calculate ValStock
        sg1.computeValStock(period)
        
        ##### ------ end ------
        
        # Compute(Update) min/max Learning cost (LearningCost) for prosumers
        sg1.computeLCost_LCostMinMax(period)
        
        # we update probabilities (prmod) of prosumers strategies
        # Calculate utility
        sg1.computeUtility(period)
        
        # Update probabilities for choosing modes
        sg1.updateProbaLRI(period, self.b)
        
        return sg1
    
    def admitNashequilibrium(self, period:int, file:typing.TextIO):
        """
        Output in the selected file  for each period if one of all possible strategy is a Nash equilibrium
        
        This method work the same way as isNashequilibrium except it will test all possible stategy and stop if one is a Nash equilibrium
        This can be quite long if none exist so consider not using it for huge instances

        Parameters
        ----------
        period : int
            DESCRIPTION.
        file : typing.TextIO
            DESCRIPTION.

        Returns
        -------
        None.

        """
        N = self.SG.prosumers.size
        
        for l in range(N * 2):
            if l%2 == 0:
                app1 = copy.deepcopy(self)
            
            if app1.SG.prosumers[l//2].state[period] == ag.State.DEFICIT :
                if app1.SG.prosumers[l//2].mode[period] == ag.Mode.CONSMINUS :
                    app1.SG.prosumers[l//2].mode[period] = ag.Mode.CONSPLUS
                else :
                    app1.SG.prosumers[l//2].mode[period] = ag.Mode.CONSMINUS
            elif app1.SG.prosumers[l//2].state[period] == ag.State.SELF :
                if app1.SG.prosumers[l//2].mode[period] == ag.Mode.CONSMINUS :
                    app1.SG.prosumers[l//2].mode[period] = ag.Mode.DIS
                else:
                    app1.SG.prosumers[l//2].mode[period] = ag.Mode.CONSMINUS
            else:
                if app1.SG.prosumers[l//2].mode[period] == ag.Mode.DIS :
                    app1.SG.prosumers[l//2].mode[period] = ag.Mode.PROD
                else:
                    app1.SG.prosumers[l//2].mode[period] = ag.Mode.DIS
                    
            sg1 = self.execute_SG_for_NashEquilibrium(period=period, sg1=app1.SG)
            
            ValSG_old = sg1.ValSG[period]
            
            fail = 0
            nash = 0
            
            for i in range(N):
                sg2 = copy.deepcopy(sg1)
                if sg2.prosumers[i].state[period] == ag.State.DEFICIT :
                    if sg2.prosumers[i].mode[period] == ag.Mode.CONSMINUS :
                        sg2.prosumers[i].mode[period] = ag.Mode.CONSPLUS
                    else :
                        sg2.prosumers[i].mode[period] = ag.Mode.CONSMINUS
                elif sg2.prosumers[i].state[period] == ag.State.SELF :
                    if sg2.prosumers[i].mode[period] == ag.Mode.CONSMINUS :
                        sg2.prosumers[i].mode[period] = ag.Mode.DIS
                    else:
                        sg2.prosumers[i].mode[period] = ag.Mode.CONSMINUS
                else:
                    if sg2.prosumers[i].mode[period] == ag.Mode.DIS :
                        sg2.prosumers[i].mode[period] = ag.Mode.PROD
                    else:
                        sg2.prosumers[i].mode[period] = ag.Mode.DIS
                
                ### ---- Execute the copy sg2 with new parameters  ----
                sg2 = self.execute_SG_for_NashEquilibrium(period=period, sg1=sg2)
                
                ValSG_tmp = sg2.ValSG[period]
                
                if ValSG_old < ValSG_tmp:
                    fail = 1
                    break
            
            if fail == 0:
                file.write("SG admit a nash equilibrium on period " + str(period) + "\n\n")
                for m in range(N):
                    file.write("Prosumer " + str(m) + "\n")
                    file.write("Found    : " + str(self.SG.prosumers[m].mode[period]) + "\n")
                    file.write("Expected : " + str(sg1.prosumers[m].mode[period]) + "\n")
                    file.write("--------------------------- \n")
                sg1.computeValSG(period=period)
                file.write("ValSG : " + str(sg1.ValSG[period]) + "\n")
                nash = 1
                break
            
        if nash == 0:
            file.write("SG does not admit a nash equilibrium on period " + str(period) + "\n")
                
        
    def isNashEquilibrium(self, period:int, epsilon:float, file:typing.TextIO): 
        """
        Output in selected file for one period if the strategy is a Nash equilibrium
        
        
        """
        print(f"NE: period={period}")
        N = self.SG.prosumers.size
        
        fail = 0
        
        ValSG_old = self.SG.ValSG[period]
        ValNoSG_old = self.SG.ValNoSG[period]
        
        for i in range(N):
            print(f"NE: prosumer_{i} - start")
            
            #Create a deepcopy of the smartgrid
            sg1 = None
            sg1 = copy.deepcopy(self.SG)
            
            #Change prosumer i mode for the other mode possible for his state
            if sg1.prosumers[i].state[period] == ag.State.DEFICIT :
                if sg1.prosumers[i].mode[period] == ag.Mode.CONSMINUS :
                    sg1.prosumers[i].mode[period] = ag.Mode.CONSPLUS
                else :
                    sg1.prosumers[i].mode[period] = ag.Mode.CONSMINUS
            elif sg1.prosumers[i].state[period] == ag.State.SELF :
                if sg1.prosumers[i].mode[period] == ag.Mode.CONSMINUS :
                    sg1.prosumers[i].mode[period] = ag.Mode.DIS
                else:
                    sg1.prosumers[i].mode[period] = ag.Mode.CONSMINUS
            else:
                if sg1.prosumers[i].mode[period] == ag.Mode.DIS :
                    sg1.prosumers[i].mode[period] = ag.Mode.PROD
                else:
                    sg1.prosumers[i].mode[period] = ag.Mode.DIS
        
            ### ---- Execute the copy sg1 with new parameters  ----
            sg1 = self.execute_SG_for_NashEquilibrium(period=period, epsilon=epsilon, sg1=sg1)
            
            
            # Test if the new benefit is higher than the one of the first smartgrid
            # if ValSG_old < ValSG_tmp then NO Nash Equilibrium (fail = 1)
            # if ValSG_old > ValSG_tmp then Nash Equilibrium (fail = 0)
            ValSG_tmp = sg1.ValSG[period]
            ValNoSG_tmp = sg1.ValNoSG[period]
            
            fail = 1 if ValSG_old < ValSG_tmp  else 0
            
            # Test if a Nash equilibrium exist for this period
            if fail == 1 : 
                file.write("SG is NOT in a Nash equilibrium for period " + str(period) + "\n")
                self.NE_ai[i] = 1
                print(f"NE: prosumer_{i} - NO NASH EQUILIBRIUM t={period} END")
            else :
                file.write("SG is  in Nash equilibrium on period " + str(period) + "\n")
                print(f"NE: prosumer_{i} - NASH EQUILIBRIUM t={period} END")
                #self.admitNashequilibrium(period=period, file=file)
                
                # create a numpy array to save information of Nash equilibrium 
                self.NE_ai[i] = 0
            pass
        
        # Nash Equilibrium Brute : all prosumers have a NE ie 1
        self.SG.NE_brute_t[period] = self.NE_ai
        # self.SG.NE_brute_t[period] = 1 if self.NE_ai.all()  else 0
    
    ###########################################################################
    #                       identify nash equilibruim at one period:: end
    ###########################################################################