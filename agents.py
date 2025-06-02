#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:08:11 2024

@author: willy

agents file contains functions and classes for items that are independant actions 
in the game
"""
import numpy as np
from enum import Enum
import auxiliary_functions as aux

#%% state
class State(Enum): # Define possible states for a prosumer
    DEFICIT = "DEFICIT" 
    SELF = "SELF"
    SURPLUS = "SURPLUS"
    
#%% mode
class Mode(Enum):  # Define possible modes for a prosumer
    CONSPLUS = "CONS+"
    CONSMINUS = "CONS-"
    DIS = "DIS"
    PROD = "PROD"
    
#%% prosumer
class Prosumer:
    
    # This class represent a prosumer or an actor of the smartgrid

    state = None  # State of the prosumer for each period , possible values = {DEFICIT,SELF,SURPLUS}
    production = None # Electricity production during each period
    consumption = None # Electricity consumption during each period
    prodit = None # Electricity inserted to the SG
    consit = None # Electricity consumed from the SG
    storage = None # Electricity storage at the beginning of each period
    smax = None # Electricity storage capacity
    gamma = None # Incentive to store or preserve electricity
    mode = None # Mode of the prosumer for each period, possible values = {CONSPLUS,CONSMINUS,DIS,PROD}
    prmode = None # Probability to choose between the two possible mode for each state at each period shaped like prmode[period][mode] (LRI only)
    utility = None # Result of utility function for each period
    #minbg = None # Minimum benefit obtained during all periods
    #maxbg = None # Maximum benefit obtained during all periods
    #benefit = None # Benefits for each period
    
    ##### new parameters variables for Repeated game ########
    # prediction stock
    #rho_cons = None # a prediction capacity of an actor for consumption
    #rho_prod = None # a prediction capacity of an actor for production
    #rho = None # the next periods to add at nbperiod periods. This parameter enables the prediction of values from nbperiod+1 to nbperiod+rho periods with rho << nbperiod
    #alphai = None       # the min value of tau_i^j such that tau_i < 0 and  1<=j<=rho 
    #tau = None # the stock demand of each actor for h next periods
    #Needs = None # the energy needs
    #Provs = None # 
    #Min_K = None #
    #i_tense = None # contains h value for which Nds_h > Prv_h
    #QTStock = None
    #CP_th = None # the difference between consumption and production at t+h with h in [1,rho]
    #PC_th = None # the difference between production and consumption at t+h with h in [1,rho]
    #Xi = None #
    #High = None # a MAX needed stock for each actor at step t
    #Low = None  # a MIN needed stock for each actor at step t
    #rs_high_plus = None  # a required stock value for max needed stock
    #rs_low_plus = None   # a required stock value for min needed stock
    #rs_high_minus = None # a missing stock value for max needed stock
    #rs_low_minus = None  # a missing stock value for min needed stock
    
    ###### what each actor will gain or lose
    ObjValue = None # Value of Objective function of each actor ObjAi
    price = None # price by by each prosumer during all periods
    valOne = None # value 
    valNoSG = None # 
    ImpVSG = None #
    VSG = None #
    valStock = None # 
    Repart = None # a repartition function based on shapley value
    cost = None # cost of each actor
    Lcost = None # a learning cost
    LearningIndicator = None  # a learning indicator 
    Lcostmin = None # a min value of Lcost over the various learning steps
    Lcostmax = None # a max value of Lcost over the various learning steps
    benefit = None # reward for each prosumer at each period
    
    ##### kept parameters variables for Repeated game #######
    rho = None # the next periods to add at nbperiod periods. This parameter enables the prediction of values from nbperiod+1 to nbperiod+rho periods with rho << nbperiod
    Needs = None # the energy needs
    QTStock = None
    Xi = None #
    
    
    ##### new parameters variables for Repeated game ########
    tau_minus = None # a required stock quantity of a player for the defined future rho
    tau_plus = None # an available stock quantity of a player for the defined future rho
    SP = None # a storage profile of a player for the defined future rho
    gamma = None # a minimum future steps respecting conditions in the overleaf paper
    Help = None
    Rq = None
    Val = None
    mu = None
    EstimStock = None
    poisson_random_value = None
    

    def __init__(self, nbperiod:int, initialprob:float, rho:int):
        """
        

        Parameters
        ----------
        nbperiod : int
            explicit max number of periods for a game.
            
        initialprob : float
            initial probability value of prmode[0]. 
            
        rho : int
            This parameter enables the prediction of values 
            from nbperiod+1 to nbperiod+rho periods or 
            from nbperiod to nbperiod+rho+1 periods
            rho << nbperiod ie rho=3 < nbperiod=5
            
        Returns
        -------
        None.

        """
        
        
        self.state = np.zeros(nbperiod+rho, dtype=State)
        self.production = np.zeros(nbperiod+rho) 
        self.consumption = np.zeros(nbperiod+rho)
        self.prodit = np.zeros(nbperiod)
        self.consit = np.zeros(nbperiod)
        self.storage = np.zeros(nbperiod+rho) 
        self.smax = 0
        self.mode = np.zeros(nbperiod+rho, dtype=Mode)
        self.prmode = np.zeros((nbperiod,2))
        for i in range(nbperiod):
            self.prmode[i][0] = initialprob
            self.prmode[i][1] = 1 - initialprob
        self.utility = np.zeros(nbperiod)
        
        ##### kept parameters variables for Repeated game #######
        self.rho = rho
        self.Needs = np.zeros(shape=(nbperiod, rho+1))
        self.QTStock = np.zeros(nbperiod)
        self.Xi = np.zeros(shape=(nbperiod, rho+1))
        
        ##### new parameters variables for Repeated game ########
        self.gamma = np.zeros(nbperiod) 
        self.SP = np.zeros(shape=(nbperiod, rho+1))
        self.tau_minus = np.zeros(shape=(nbperiod, rho+1))
        self.tau_plus = np.zeros(shape=(nbperiod, rho+1))
        self.Help = np.zeros(shape=(nbperiod, rho+1))
        self.Rq = np.zeros(shape=(nbperiod, rho+1))
        self.Val = np.zeros(shape=(nbperiod, rho+1))
        self.mu = np.zeros(nbperiod)
        self.EstimStock = np.zeros(nbperiod)
        self.poisson_random_value = np.zeros(nbperiod)
        
        
        ##### TODELETE : start new parameters variables for Repeated game ########
        # self.rho_cons = 0
        # self.rho_prod = 0
        # self.rho = rho
        # self.alphai = 0
        # self.tau = np.zeros(shape=(nbperiod, rho+1))
        # self.Needs = np.zeros(shape=(nbperiod, rho+1))
        # self.Provs = np.zeros(shape=(nbperiod, rho+1))
        # self.Min_K = np.zeros(shape=(nbperiod, rho+1))
        # self.i_tense = np.zeros(shape=(nbperiod, rho+1))
        # self.CP_th = np.zeros(shape=(nbperiod, rho+1))
        # self.PC_th = np.zeros(shape=(nbperiod, rho+1))
        ##### TODELETE : end
        
        # TODELETE: start
        # self.Xi = np.zeros(nbperiod+rho)
        # self.High = np.zeros(nbperiod+rho)
        # self.Low = np.zeros(nbperiod+rho)
        # self.rs_high_plus = np.zeros(nbperiod)
        # self.rs_low_plus = np.zeros(nbperiod)
        # self.rs_high_minus = np.zeros(nbperiod)
        # self.rs_low_minus = np.zeros(nbperiod)
        # TODELETE : END
        
        self.ObjValue = 0 #np.zeros(maxperiod)
        self.price = np.zeros(nbperiod)
        self.valOne = np.zeros(nbperiod)
        self.valNoSG = np.zeros(nbperiod)
        self.ImpVSG = np.zeros(nbperiod)
        self.VSG = np.zeros(nbperiod)
        self.valStock = np.zeros(nbperiod)
        self.Repart = np.zeros(nbperiod)
        self.cost = np.zeros(nbperiod)
        self.Lcost = np.zeros(nbperiod)
        self.LearningIndicator = np.zeros(nbperiod)
        self.Lcostmin = np.ndarray(shape=nbperiod, dtype='object')
        self.Lcostmax = np.zeros(shape=nbperiod, dtype='object')
        self.LCostmax = dict({"price":None, "valStock":None, "mode":None, "state":None, "Lcost":None})
        self.LCostmin = dict({"price":None, "valStock":None, "mode":None, "state":None, "Lcost":None})
        

    def computeValOne(self, period:int, nbperiod:int, rho:int) -> float:
        """
        compute the value of each actor at one period

        Parameters
        ----------
        period : int
            an instance of time t
            
        nbperiod : int
            explicit max number of periods for a game
            
        rho: int
            the next periods to add from the nbperiod periods
            rho << nbperiod ie rho=3 < nbperiod=5

        Returns
        -------
        float.

        """
        nextperiod = period if period == nbperiod+rho-1 else period+1
        
        
        phiminus = aux.apv(self.consumption[period] 
                               - self.production[period] 
                               - self.storage[period])
        phiplus = aux.apv(self.production[period] 
                              - self.consumption[period] 
                              - (self.storage[nextperiod] - self.storage[period]))
        
        self.valOne[period] = phiminus - phiplus
        
    def computeValNoSG(self, period:int, coef_phiepominus:int, coef_phiepoplus:int) -> float:
        """
        compute the value of one actor a_i at one period

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        float.

        """
        self.valNoSG[period] = aux.phiepominus(self.consit[period], coef_phiepominus) \
                                - aux.phiepoplus(self.prodit[period], coef_phiepoplus)
        
    def computeTauMinusPlus(self, period:int, rho:int) -> float:
        """
        compute a required (tau_minus) and the available (tau_plus) stock quantity  for the specified period h in the future rho

        Parameters
        ----------
        period : int
            an instance of time t
            
        rho : int
            the next periods to add from the nbperiod periods
            rho << nbperiod ie rho=3 < nbperiod=5

        Returns
        -------
        float
            DESCRIPTION.

        """
        for h in range(0, rho+1):
            self.tau_minus[period, h] = aux.apv(self.consumption[period+h] - self.production[period+h])
            self.tau_plus[period, h] = aux.apv(self.production[period+h] - self.consumption[period+h])
        
    def computeSP(self, period:int, rho:int) -> float:
        """
         compute the Storage profile/prediction 

        Parameters
        ----------
        period : int
            an instance of time t
            
        rho : int
            the next periods to add from the nbperiod periods
            rho << nbperiod ie rho=3 < nbperiod=5

        Returns
        -------
        float
            DESCRIPTION.

        """
        
        # ##########   compute for h in [0, rho] : start -> 20/01/2025   ######
        h = 0
        self.SP[period, h] = self.storage[period]
       
        for h in range(1, rho+1):
            sp_ith = -10
            if self.tau_plus[period, h-1] > 0:
                sp_ith = min(self.SP[period, h-1] + self.tau_plus[period, h-1], self.smax)
                pass
            elif self.tau_minus[period, h-1] > 0:
                sp_ith = max(self.SP[period, h-1] - self.tau_minus[period, h-1], 0)
                pass
            elif self.tau_plus[period, h-1] == 0 and self.tau_minus[period, h-1] == 0:
                sp_ith = self.SP[period, h-1]
                pass
            pass
           
            self.SP[period, h] = sp_ith
       
        # ##########   compute for h in [0, rho] : end -> 20/01/2025     ######
        
        # # TODO # may be a side effect with h starts to 1 or 2
        # self.SP[period, 0] = 0;
        # h=1
        # self.SP[period, h] = 0
        
        # # ##################   compute for h=1 : start  ##################
        # # h=1
        # # if self.tau_plus[period, h] > 0:
        # #     self.SP[period, h] \
        # #         = min(self.storage[period]+self.tau_plus[period, h], self.smax)
        # # elif self.tau_minus[period, h] > 0:
        # #     self.SP[period, h] \
        # #         = max(self.storage[period]-self.tau_minus[period, h], 0)
        # # else:
        # #     self.SP[period, h] = self.storage[period]
        # # ##################   compute for h=1 : end   ##################
        
        # for h in range(2, rho+1):
        #     if self.tau_plus[period, h-1] > 0:
        #         self.SP[period, h] \
        #             = min(self.SP[period, h-1]+self.tau_plus[period, h-1], self.smax)
        #     elif self.tau_minus[period, h-1] > 0:
        #         self.SP[period, h] \
        #             = max(self.SP[period, h-1]-self.tau_minus[period, h-1], 0)
        #     else:
        #         self.SP[period, h] = self.SP[period, h-1] 
            
                
    def computeGamma(self, period:int) -> float:
        """
        Compute gamma

        Parameters
        ----------
        period : int
            an instance of time t
            
        rho : int
            the next periods to add from the nbperiod periods
            rho << nbperiod ie rho=3 < nbperiod=5

        Returns
        -------
        float
            DESCRIPTION.

        """
        self.gamma[period] = None
        min_h = np.inf
        for h in range(1, self.rho+1):
            if self.SP[period, h] == self.smax or h == self.rho:
                if min_h > h :
                    min_h = h
        self.gamma[period] = min_h -1
        
        if self.gamma[period] == self.rho-1 and self.SP[period, self.rho-1] < self.smax:
            self.gamma[period] = self.rho
        # self.gamma[period] = self.rho
        
    def computeNeeds4OneProsumer(self, period:int) -> float:
        """
        Compute the need of each actor on rho periods

        Parameters
        ----------
        period : int
            an instance of time t
        
        Returns
        -------
        float
            DESCRIPTION.

        """
        for h in range(1, self.rho+1):
            self.Needs[period, h] = aux.apv(self.tau_minus[period, h] - self.SP[period, h])
            
        
    def simulate_poisson(self, lambda_poisson):
        k = 0
        p = 1
        while p > np.exp(-lambda_poisson):
            u = np.random.uniform(0, 1)
            p *= u
            k += 1
        return k - 1

    def generate_poisson_random_value(self, period:int, lambda_poisson:float) -> float:
        """
         generate a poisson random value from poisson distribution

        Parameters
        ----------
        period : int
            an instance of time t
        lambda_poisson : float
            parameter

        Returns
        -------
        float
            a poisson random value.

        """
        
        #-------- version 3
        n_simulations = 1000
        
        # Générer des variables aléatoires Poisson
        poisson_values = np.random.poisson(lambda_poisson, n_simulations)
        
        max_poisson_value = max(poisson_values)
        
        # affecter la valeur/100 pour la periode donnee
        self.poisson_random_value[period] = max_poisson_value/100
    
        
     
    # def computeMu4OneProsumer(self, period:int) -> float:
    #     """
    #     compute mu for each actor

    #     Parameters
    #     ----------
    #     period : int
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     float
    #         DESCRIPTION.

    #     """
    #     min_Smax_SP = np.inf
    #     for h in range(1, int(self.gamma[period])+1 ):
    #         if min_Smax_SP > self.smax - self.SP[period, h] :
    #             min_Smax_SP = self.smax - self.SP[period, h]
                
    #     self.mu[period] = min_Smax_SP
        
    
    # def computePC_CP_th(self, period:int, nbperiod:int, rho:int):
    #     """
    #     TODO : TODELETE because USELESS
    #     compute a difference between consumption and production at t+h with h \in [1, rho] 

    #     Parameters
    #     ----------
    #     period : int
    #         an instance of time t
    #     nbperiod: int
    #         max period
    #     rho: int
    #         number of periods to select for predition
    #         rho << nbperiod ie rho=3 < nbperiod=5

    #     Returns
    #     -------
    #     np.ndarray.

    #     """        
    #     rho_max = rho if period < nbperiod else nbperiod-rho                   # max prediction slots rho_max ; rho_max = rho if t < T else T-rho 
    #     for h in range(1, rho_max+1):
    #         if h == 1:
    #             self.CP_th[period,h] = self.consumption[period+h] - (self.production[period+h] + self.storage[period])
    #             self.PC_th[period,h] = (self.production[period+h] + self.storage[period]) - self.consumption[period+h]
    #         else:
    #             self.CP_th[period,h] = self.consumption[period+h] - self.production[period+h]
    #             self.PC_th[period,h] = self.production[period+h] - self.consumption[period+h]
            
    # def computeTau(self, period:int, nbperiod:int, rho:int) -> float:
    #     """
    #     compute a parameter $\tau_i^{t+h}$ indicates for each of the $\rho$ predicted steps 
    #     the cumulative need of each actor $a_i$ in terms of stock if he only 
    #     relied on his own productions

    #     Parameters
    #     ----------
    #     period : int
    #         an instance of time t
    #     maxperiod: int
    #         max period
    #     rho: int
    #         number of periods to select for predition 
    #         rho << nbperiod ie rho=3 < nbperiod=5

    #     Returns
    #     -------
    #     float.

    #     """
    #     #nextperiod = period if period == nbperiod+rho-1 else period+1
        
    #     rho_max = rho if period < nbperiod else nbperiod-rho                   # max prediction slots rho_max ; rho_max = rho if t < T else T-rho  
        
    #     # New version to compute tau
    #     for h in range(1, rho_max+1):
    #         if h == 1:
    #             self.tau[period,h] = self.consumption[period+h] - (self.production[period+h] + self.storage[period])
    #         else:
    #             self.tau[period,h] = self.consumption[period+h] - self.production[period+h]
                
        
        
    #     # TODO A tester : 1er version
    #     # for h in range(1, rho_max+1):
    #     #    self.tau[h] = np.sum(self.CP_th[:h+1])
        
    #     # print(f" tau = {self.tau[h]}")                                        =====> TODELETE
    #     # TODO A tester : 2e version
    #     # for h in range(1, rho_max+1):
    #     #     CP_thj = 0
    #     #     for j in range(1,h+1):
    #     #         CP_thj += self.consumption[period+j] - self.production[period+j]
                
    #     #     self.tau[h] = CP_thj - Si_tplus1
           
    # def computeNeeds4OneProsumer(self, period:int, nbperiod:int)-> float:
    #     """
    #     Compute the need of each actor on rho periods

    #     Parameters
    #     ----------
    #     period : int
    #         DESCRIPTION.
    #     rho : int
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     float
    #         DESCRIPTION.

    #     """
    #     rho_max = self.rho if period < nbperiod else nbperiod-self.rho
        
    #     for h in range(1, rho_max+1):
    #         #tmp = self.tau[period,:h+1]
    #         #tmp = tmp[tmp > 0]
    #         #self.Needs[period, h] = np.sum(tmp)
    #         self.Needs[period, h] = aux.apv(self.tau[period,h])
    
    # def computeProvsAtH0(self, period:int, nbperiod:int) -> float():
    #     """
    #     Calculate Prov at h=0

    #     Parameters
    #     ----------
    #     period : int
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     float
    #         DESCRIPTION.

    #     """
        
    #     # rho_max = self.rho if period < nbperiod else nbperiod-self.rho
        
    #     self.Provs[0] = self.storage[period]
        
            
                
        
    # def computeX(self, period:int)-> float:
    #     """
    #     the difference beteween tau and X is the absolute value

    #     Parameters
    #     ----------
    #     period : int
    #         an instance of time t
        
    #     maxperiod: int
    #         max period
            
    #     rho: int
    #         number of periods to select for predition from t+1 to t+rho 
    #         rho << nbperiod ie rho=3 < nbperiod=5

    #     Returns
    #     -------
    #     float.

    #     """
        
    #     # rho_max = rho if period < nbperiod else nbperiod-rho                   # max prediction slots rho_max ; rho_max = rho if t < T else T-rho 
        
    #     # sum_X = 0
    #     # for h in range(1, rho_max+1):
    #     #     sum_X += aux.apv(self.tau[period,h])
            
    #     # self.Xi[period] = sum_X
        
    #     sum_X = 0
    #     for h in range(1, self.rho+1):
    #         sum_X += aux.apv(self.tau_minus[period, h])
            
    #     self.Xi[period] = sum_X
        
        