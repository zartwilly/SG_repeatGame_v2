#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:07:22 2024

@author: willy

smartgrid_rg is the smartgrid in the repeat game that centralizes the parameters of the environment
"""
import math
import random as rdm
import numpy as np
import agents as ag
import auxiliary_functions as aux

class Smartgrid :
    
    # This class represent a smartgrid
    nbperiod = None # Number of periods
    rho = None      # the next periods to add at nbperiod periods. This parameter enables the prediction of values from nbperiod+1 to nbperiod+rho periods with rho << nbperiod
    prosumers = None # All prosumers inside the smartgrid
    LCostmax = None # Maximum benefit for each prosumer for each period
    LCostmin = None # Minimum benefit for each prosumer for each period
    insg = None # Sum of electricity input from all prosumers
    outsg = None # Sum of electricity outputs from all prosumers
    ValEgoc = None # sum of all valOne 
    ValNoSG = None
    ValSG = None
    ValNoSGCost = None
    Reduct = None
    strategy_profile = None 
    Cost = None
    
    QttEpo_plus = None      # quantity to give to EPO
    QttEpo_minus = None     # quantity to take to EPO
    
    GridNeeds = None
    Free = None
    
    NE_brute_t = None       # save brute Nash Equilibrium at each period
    
    # TODELETE
    # DispSG = None
    # TauS = None # contains the tau array for all players at period t 
    # GNeeds = None
    
    
    # TODO to delete
    # piepoplus = None # Unitary price of electricity purchased by EPO
    # piepominus = None # Unitary price of electricity sold by EPO
    # piplus = None # Unitary benefit of electricity sold to SG (independently from EPO)
    # piminus = None # Unitary cost of electricity bought from SG (independently from EPO)
    # unitaryben = None # Unitary benefit of electricity sold to SG (possibly partially to EPO)
    # unitarycost = None # Unitary cost of electricity bought from SG (possibly partially from EPO)
    # betaplus = None # Intermediate values for computing piplus and real benefit 
    # betaminus = None # Intermediate values for computing piminus and real cost
    # czerom = None # Upper bound a prosumer could have to pay
    # realprod = None # Real value of production for each prosumers (different from the predicted production)
    # realstate = None # Real state of each prosumers when using real production value (can be the same as the one determined with predicted production)
    
    
    def __init__(self, N:int, nbperiod:int, initialprob: float, rho:int, 
                 coef_phiepoplus:int, coef_phiepominus:int):
        """
        
        Parameters
        ----------
        N : int
            number of prosumers
            
        nbperiod : int
            explicit max number of periods for a game.
            
        initialprob : float 
                initial value of probabilities for LRI 
                
        rho : int
            steps to be taken into account for stock prediction
            the next periods to add at nbperiod periods. 
            This parameter enables the prediction of values from nbperiod+1 to nbperiod+rho periods 
            with rho << nbperiod
            
        
        """
        self.rho = rho
        self.coef_phiepominus = coef_phiepominus
        self.coef_phiepoplus = coef_phiepoplus
        self.prosumers = np.ndarray(shape=(N),dtype=ag.Prosumer)
        self.nbperiod = nbperiod
        for i in range(N):
            self.prosumers[i] = ag.Prosumer(nbperiod=nbperiod, initialprob=initialprob, rho=rho)   
        #self.bgmax = np.zeros((N,maxperiod))
        
        self.LCostmax = np.zeros(nbperiod)
        self.LCostmin = np.zeros(nbperiod)
        
        self.insg = np.zeros(nbperiod)       
        self.outsg = np.zeros(nbperiod)
        
        self.QttEpo_plus = np.zeros(nbperiod)
        self.QttEpo_minus = np.zeros(nbperiod)
        
        self.ValEgoc = np.zeros(nbperiod)
        self.ValNoSG = np.zeros(nbperiod)
        self.ValSG = np.zeros(nbperiod)
        self.ValNoSGCost = np.zeros(nbperiod)
        self.Reduct = np.zeros(nbperiod)
        dt = np.dtype([('agent', int), ('strategy', ag.Mode)])
        self.strategy_profile = np.ndarray(shape=(N, nbperiod), dtype=dt)
        self.Cost = np.zeros(nbperiod)
        self.GridNeeds = np.zeros(shape=(nbperiod,rho+1))
        self.Free = np.zeros(shape=(nbperiod,rho+1))
        
        self.NE_brute_t = np.full_like(np.zeros(shape=(nbperiod,N)), [-1 for i in range(N)]) #np.zeros(nbperiod)
        # self.TauS = np.ndarray(shape=(N, rho+1))
        # self.DispSG = np.zeros(rho+1)
        # self.GNeeds = np.zeros(shape=(nbperiod,rho+1))
        # self.GPd = np.zeros(shape=(nbperiod, rho+1))
        
    ###########################################################################
    #                   compute smartgrid variables :: start
    ###########################################################################
    def computeSumInput(self, period:int) -> float: 
        """
        Calculate the sum of the production of all prosumers during a period
        
        Parameters
        ----------
        period: int 
            an instance of time t
        """
        tmpsum = 0
        for i in range(self.prosumers.size):
            tmpsum = tmpsum + self.prosumers[i].prodit[period]
        self.insg[period] = tmpsum
    
    def computeSumOutput(self, period:int) -> float: 
        """
        Calculate sum of the consumption of all prosumers during a period
        
        Parameters
        ----------
        period: int 
            an instance of time t
        """
        tmpsum = 0
        for i in range(self.prosumers.size):
            tmpsum = tmpsum + self.prosumers[i].consit[period]
        self.outsg[period] = tmpsum
        
    def computeValEgoc(self, period:int) -> float:
        """
        compute ValEgoc ie the sum of all actors valOne

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        float.

        """
        sumValEgoc = 0
        for i in range(self.prosumers.size):
            self.prosumers[i].computeValOne(period=period, nbperiod=self.nbperiod, rho=self.rho)
            sumValEgoc += self.prosumers[i].valOne[period]
            
        self.ValEgoc[period] = sumValEgoc
        
    def computeValNoSG(self, period:int) -> float:
        """
        compute valNoSG for all actors

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        float.

        """
        sumValNoSG = 0
        for i in range(self.prosumers.size):
            self.prosumers[i].computeValNoSG(period=period, 
                                             coef_phiepominus=self.coef_phiepominus,
                                             coef_phiepoplus=self.coef_phiepoplus)
            sumValNoSG += self.prosumers[i].valNoSG[period]
            
        self.ValNoSG[period] = sumValNoSG
        
    def computeValSG(self, period:int) -> float:
        """
        compute the gain of the grid ie what a grid have to pay to EPO minus what a grid receive from EPO

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        float.

        """
        outinsg = aux.phiepominus( x=aux.apv(self.outsg[period] - self.insg[period]), coef=self.coef_phiepominus)
        inoutsg = aux.phiepoplus( x=aux.apv(self.insg[period] - self.outsg[period]), coef=self.coef_phiepoplus)
        self.ValSG[period] = outinsg - inoutsg
    
    def computeValNoSGCost(self, period:int) -> float:
        """
        compute the gain when we have no smart grid.

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        float.

        """
        phiPlusInsg = aux.phiepoplus(x=self.insg[period], coef=self.coef_phiepoplus)
        phiMinusOutsg = aux.phiepominus(x=self.outsg[period], coef=self.coef_phiepominus)
        self.ValNoSGCost[period] = phiPlusInsg - phiMinusOutsg
        
    def computeQttepo_plus(self, period:int) -> float:
        """
        compute quantity prosumers must give to EPO

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        float
            DESCRIPTION.

        """
        self.QttEpo_plus[period] = aux.apv( self.insg[period] - self.outsg[period] )
        
    def computeQttepo_minus(self, period:int) -> float:
        """
        compute quantity prosumers must receive to EPO

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        float
            DESCRIPTION.

        """
        self.QttEpo_minus[period] = aux.apv( self.outsg[period] - self.insg[period] )
        
    def computeReduct(self, period:int) -> float:
        """
        Compute Reduct ie ValNoSG_t - ValSG_t

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        float.

        """
        self.Reduct[period] = self.ValNoSG[period] - self.ValSG[period]
        
    def computePrice(self, period:int) -> float:
        """
        compute the price by which each actor have to pay or sell an electricity 

        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        np.array(N,1) : float

        """
        for i in range(self.prosumers.size):
            self.prosumers[i].price[period] \
                = self.prosumers[i].valNoSG[period] \
                    - self.prosumers[i].Repart[period]
                    
    def computeTauMinusPlus4Prosumers(self, period:int) -> float:
        """
        compute Tau_minus, tau_plus for all prosumers

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        float
            DESCRIPTION.

        """
        for i in range(self.prosumers.size):
            self.prosumers[i].computeTauMinusPlus(period=period, rho=self.rho)
                    
    def computeXi4Prosumers(self, period:int) -> float:
        """
        compute Xi for all prosumers

        Parameters
        ----------
        period : int
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        """
        for i in range(self.prosumers.size):
            self.prosumers[i].computeX(period=period)
            
    def computeSP4Prosumers(self, period:int) -> float:
        """
        compute SP for all prosumers

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        float
            DESCRIPTION.

        """
        for i in range(self.prosumers.size):
            self.prosumers[i].computeSP(period=period, rho=self.rho)
    
    def computeGamma4prosumers(self, period:int) -> float:
        """
        compute Gamma for all prosumers
        
        Parameters
        ----------
        period : int
            an instance of time t.
            
        Returns
        -------
        float
            DESCRIPTION.

        """
        for i in range(self.prosumers.size):
            self.prosumers[i].computeGamma(period=period)
    
    def computeGridNeeds4Prosumers(self, period:int) -> float:
        """
        compute GridNeeds for all prosumers

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        float
            DESCRIPTION.

        """
        for i in range(self.prosumers.size):
            self.prosumers[i].computeNeeds4OneProsumer(period=period)
            
        for h in range(1, self.rho+1):
            self.GridNeeds[period, h] = 0
            for i in range(self.prosumers.size):
                self.GridNeeds[period, h] += self.prosumers[i].Needs[period, h]
            # print(f"h={h}, GridNeeds={self.GridNeeds[period, h]}")
            
    def computeFree4Prosumers(self, period:int) -> float:
        """
        compute the amount of free energy injected in the grid

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        float
            DESCRIPTION.

        """
        for h in range(1, self.rho+1):
            Gh = 0
            for i in range(self.prosumers.size):
               Gh += aux.apv(self.prosumers[i].production[period+h] \
                - self.prosumers[i].consumption[period+h] \
                - (self.prosumers[i].smax - self.prosumers[i].SP[period,h]))
    
            self.Free[period, h] = Gh
            
    def computeHelp4Prosumers(self, period:int) -> float:
        """
        compute Help for all prosumers

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        float
            DESCRIPTION.

        """
        for i in range(self.prosumers.size):
            for h in range(1, self.rho+1):
                self.prosumers[i].Help[period, h] \
                    = min(
                        self.prosumers[i].Needs[period, h]*self.Free[period, h]/self.GridNeeds[period, h],
                        self.prosumers[i].Needs[period, h]
                        )
                    
    def computeRq(self, period:int) -> float:
        """
        compute Rq for all prosumers

        Parameters
        ----------
        period : int
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        """
        for i in range(self.prosumers.size):
            for h in range(1, self.rho+1):
                self.prosumers[i].Rq[period, h] \
                    = self.prosumers[i].Needs[period, h] - self.prosumers[i].Help[period, h]
               
    def computeVal(self, period:int) -> float:
        """
        compute Val for all prosumers

        Parameters
        ----------
        period : int
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        """
        for i in range(self.prosumers.size):
            for h in range(1, self.rho+1):
                min_1 = self.prosumers[i].Rq[period, h]
                min_2 = self.prosumers[i].smax - self.prosumers[i].SP[period, h]
                min_3 = np.inf
                
                for x in range(1, h):
                    smax = self.prosumers[i].smax
                    sp_x = self.prosumers[i].SP[period, x]
                    som_Val = 0
                    for y in range(x, h):
                        som_Val += self.prosumers[i].Val[period, y]
                    diff = aux.apv(smax - sp_x - som_Val)
                    
                    min_3 = diff if min_3 > diff else min_3
                
                self.prosumers[i].Val[period, h] \
                    = min(min_1, min_2, min_3)
    
    def computeEstimStock(self, period:int) -> float:
        """
        compute stock Estimation for all prosumers

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        float
            DESCRIPTION.

        """
        for i in range(self.prosumers.size):
            sum_val_over_h = 0
            for h in range(1, int(self.prosumers[i].gamma[period])+1):
                sum_val_over_h += self.prosumers[i].Val[period, h] / h
                
            self.prosumers[i].EstimStock[period] = sum_val_over_h
    
    def computePoissonRandomValue(self, period:int, lambda_poisson:float) -> float:
        """
        generate Poisson random value for all prosumers

        Parameters
        ----------
        period : int
            DESCRIPTION.
        lambda_poisson : float
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        """
        for i in range(self.prosumers.size):
            self.prosumers[i].generate_poisson_random_value(period=period, lambda_poisson=lambda_poisson)
    
    def computeQTStock(self, period:int, epsilon:float) -> float:
        """
        Calculate the quantity of stocks with 

        Parameters
        ----------
        period : int
            an instance of time t.
            
        epsilon : float
            a prediction error rate ( epsilon \in [0,1] )

        Returns
        -------
        float
            DESCRIPTION.

        """
            
        # for i in range(self.prosumers.size):
        #     somme = 0
        #     for h in range(1, int(self.prosumers[i].gamma[period])+1):
        #         #TODO
        #         somme += self.prosumers[i].Val[period, h] / h
                
        #     self.prosumers[i].QTStock[period] = somme
            
        for i in range(self.prosumers.size):
            
            value = ( 1 + 2 * ( self.prosumers[i].poisson_random_value[period] - 0.5) * epsilon) \
                    * self.prosumers[i].EstimStock[period]
            
            self.prosumers[i].QTStock[period] = value
        
        
    def computeValStock(self, period:int) -> float:
        """
        calculate the prosumer stock impact during a strategy profile SP^t=strat_{1,s}^t,...,strat_{N,s}^t

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        None.

        """
        nextperiod = period if period == self.nbperiod+self.rho-1 else period+1 
        
        for i in range(self.prosumers.size):
            Si = self.prosumers[i].storage[period]
            Si_tplus1 = self.prosumers[i].storage[nextperiod]
            QTstock_i = self.prosumers[i].QTStock[period]
            
            # -------------------  verion from 18/01/2025  -----------------
            if Si < QTstock_i:
                if Si_tplus1 < Si:
                    self.prosumers[i].valStock[period] \
                        = aux.phiepominus( Si - Si_tplus1, coef=self.coef_phiepominus)
                else:
                    self.prosumers[i].valStock[period] \
                        = aux.phiepominus(min( QTstock_i, Si_tplus1) - Si, coef=self.coef_phiepominus)
                pass
            else:
                # ------------------- version of 03/03/2025 ------------------
                if QTstock_i <= Si_tplus1 and Si_tplus1 <= Si:
                    self.prosumers[i].valStock[period] \
                        = aux.phiepoplus( Si - Si_tplus1, coef=self.coef_phiepoplus )
                elif QTstock_i >= Si_tplus1:
                    self.prosumers[i].valStock[period] \
                        = - aux.phiepominus( QTstock_i - Si_tplus1 ,coef=self.coef_phiepominus)
                elif Si_tplus1 > Si:
                    self.prosumers[i].valStock[period] \
                        = - aux.phiepoplus( Si_tplus1 - Si, coef=self.coef_phiepoplus )
                else: 
                    self.prosumers[i].valStock[period] \
                        = 5000
                # ------------------- version of 03/03/2025 ------------------
                # if Si_tplus1 < Si:
                #     self.prosumers[i].valStock[period] \
                #         = - aux.phiepominus( aux.apv(QTstock_i - Si_tplus1), coef=self.coef_phiepominus)
                # else:
                #     self.prosumers[i].valStock[period] = 0
                # pass
            
            # -------------------  verion from 18/01/2025  -----------------
            
            # ------------------- OLD verion from 18/09/2024  -----------------
            # self.prosumers[i].valStock[period] \
            #     = aux.phiepominus(min(aux.apv(Si_tplus1 - Si), 
            #                           aux.apv(QTstock_i - Si)), 
            #                       self.coef_phiepominus) \
            #         - aux.phiepominus(min( aux.apv(Si - Si_tplus1), 
            #                                aux.apv(QTstock_i - Si_tplus1)),
            #                           self.coef_phiepominus)
            # ------------------- OLD verion from 18/09/2024  -----------------
    
        
    
    # def computeTaus(self, period:int) -> float:
    #     """
        

    #     Parameters
    #     ----------
    #     period : int
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     float
    #         DESCRIPTION.

    #     """   
    #     for i in range(self.prosumers.size):
    #         self.prosumers[i].computeTau(period=period, nbperiod=self.nbperiod, rho=self.rho)
             
                    
    # def computeNeeds(self, period:int) -> float:
    #     """
    #     Compute needs for all prosumers 

    #     Parameters
    #     ----------
    #     period : int
    #         DESCRIPTION.
    #     h : int
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     float
    #         return float on array variable Needs

    #     """
    #     for i in range(self.prosumers.size):
    #         self.prosumers[i].computeNeeds4OneProsumer(period=period, nbperiod=self.nbperiod)
            
    # def computeGNeeds(self, period:int, h:int) -> float:
    #     """
    #     sum of the Needs for all players

    #     Parameters
    #     ----------
    #     period : int
    #         an instance of time t.
    #     h : int
    #         the next h periods to predict the stock at the period "period" .
    #         1 <= h <= rho

    #     Returns
    #     -------
    #     float
    #         DESCRIPTION.

    #     """
    #     tmp = 0
    #     for i in range(self.prosumers.size):
    #         tmp += self.prosumers[i].Needs[period, h]
    #     self.GNeeds[period, h] = tmp
            
    # def computeGPd(self, period:int, h:int) -> float:
    #     """
    #     sum of the absolute tau for the h value

    #     Parameters
    #     ----------
    #     period : int
    #         an instance of time t.
    #     h : int
    #         the next h periods to predict the stock at the period "period" .
    #         1 <= h <= rho

    #     Returns
    #     -------
    #     float
    #         DESCRIPTION.

    #     """
    #     tmp = 0
    #     for i in range(self.prosumers.size):
    #         tmp += aux.apv( - self.prosumers[i].tau[period,h] )
    #     self.GPd[period, h] = tmp
            
    # def computeProvsAtH(self, period:int, h:int) -> float:
    #     """
    #     Calculate Provs for one h with 1 <= h <= rho

    #     Parameters
    #     ----------
    #     period : int
    #         DESCRIPTION.
    #     h : int
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     float
    #         DESCRIPTION.

    #     """
    #     for i in range(self.prosumers.size):
    #         if self.prosumers[i].Needs[period,h] > 0:
    #             self.prosumers[i].Provs[period,h] \
    #                 = max(0, self.prosumers[i].Provs[period,h-1] - self.prosumers[i].Needs[period,h])
    #         else:
    #             # GNeeds_hminus1 = 0 if h <= 1 else self.GNeeds[h-1]
    #             # #print(f"period={period}, GNeeds_hminus1={GNeeds_hminus1}, h={h}, GPd[h-1]= {self.GPd[h-1]}")
    #             # tmp_h = -self.prosumers[i].tau[h] + self.prosumers[i].Provs[h-1] * (1 - min(1, GNeeds_hminus1/self.GPd[h-1]))
    #             # self.prosumers[i].Provs[h] = tmp_h
                
                
    #             # ######################   DBG   ######################
    #             # GNeeds_hminus1 = 0 if h <= 1 else self.GNeeds[h-1]
    #             # if self.GPd[h-1]:
    #             #     tmp_h = -self.prosumers[i].tau[h] + self.prosumers[i].Provs[h-1] * (1 - min(1, GNeeds_hminus1/self.GPd[h-1]))
    #             #     self.prosumers[i].Provs[h] = tmp_h
    #             # else:
    #             #     # print(f"period={period}, GNeeds_hminus1={GNeeds_hminus1}, h={h}, GPd[h-1]= {self.GPd[h-1]}")
    #             #     self.prosumers[i].Provs[h] = 0
                
    #             # ######################   DBG   ######################
                
    #             ######################   DBG: NEW   ######################
    #             # GNeeds_hminus1 = 0 if h <= 1 else self.GNeeds[h-1]
    #             # Contrib = 0
    #             # if self.GPd[h-1] == 0:
    #             #     Contrib = 0
    #             # else:
    #             #     Contrib = self.prosumers[i].Provs[h-1] * (1 - min(1, GNeeds_hminus1/self.GPd[h-1] ))
    #             # self.prosumers[i].Provs[h] = min(self.prosumers[i].smax, 
    #             #                                  -self.prosumers[i].tau[h] + Contrib)
    #             ######################   DBG: NEW   ######################
                
    #             ######################   DBG: NEW NEW  ######################
    #             frac = 0
    #             if self.GPd[period, h] != 0:
    #                 frac = self.GNeeds[period, h] / self.GPd[period, h]
    #             else:
    #                 frac = self.GNeeds[period, h]
    #             contrib = self.prosumers[i].Provs[period,h-1] + (-self.prosumers[i].tau[period,h]) * min(1, frac) 
                    
    #             self.prosumers[i].Provs[period,h] = min(self.prosumers[i].smax, contrib)
                
    #             ######################   DBG: NEW NEW  ######################
                
                
    #         #self.GPd[h] += self.prosumers[i].Provs[h]
                        
    #         self.prosumers[i].i_tense[period,h] = 1 if self.prosumers[i].Needs[period,h] > self.prosumers[i].Provs[period,h-1] else -1
            
    #         self.prosumers[i].Min_K[period,h] = h if self.prosumers[i].Provs[period,h] == self.prosumers[i].smax else np.inf
            

                
    # def computeProvsforRho(self, period:int) -> float:
    #     """
    #     Calculate Provs for all h with 1 <= h <= rho and identifies h i-tense

    #     Parameters
    #     ----------
    #     period : int
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     float
    #         DESCRIPTION.

    #     """
    #     for i in range(self.prosumers.size):
    #         # self.prosumers[i].computeProvsAtH0(period=period, nbperiod=self.nbperiod)
    #         self.prosumers[i].Provs[period,0] = self.prosumers[i].storage[period]
        
    #     for h in range(1, self.rho+1):
    #         self.computeGNeeds(period=period, h=h) 
    #         self.computeGPd(period=period, h=h)
    #         self.computeProvsAtH(period=period, h=h)
           
    
    # def ComputeQTStock(self, period:int) -> float:
    #     """
    #     Calculate the quantity of stocks with 

    #     Parameters
    #     ----------
    #     period : int
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     float
    #         DESCRIPTION.

    #     """
    #     # for i in range(self.prosumers.size):  
    #     #     itense_arr = (self.prosumers[i].i_tense == 1).nonzero()[0]
    #     #     min_k = min(self.prosumers[i].Min_K)
    #     #     qtstock_i = 0
    #     #     for h in itense_arr:
    #     #         qtstock_i += self.prosumers[i].Needs[h] * (1 - self.GPd[h] / self.GNeeds[h])
    #     #     self.prosumers[i].QTStock[period] = qtstock_i
            
    #     for i in range(self.prosumers.size):
    #         min_k = min(self.prosumers[i].Min_K[period,1:])
    #         min_k = 1 if min_k == np.inf else int(min_k)
    #         # print("min_k = ", min_k, "--> Min_K:", self.prosumers[i].Min_K[period,1:] )
    #         qtstock_i = 0
    #         Stpi = min(self.rho, min_k)
    #         for h in range(1, Stpi+1):
    #             # print('h = ', h, " i_tense= ", self.prosumers[i].i_tense[h])
    #             if self.prosumers[i].i_tense[period,h] == 1:
    #                 qtstock_i += aux.apv(min(self.prosumers[i].smax, self.prosumers[i].Needs[period,h]) - self.prosumers[i].Provs[period,h-1])
    #                 #print('h = ', h, ' qtstock_i = ', qtstock_i)
            
    #         self.prosumers[i].QTStock[period] = qtstock_i
    
    # def computeValStock(self, period:int) -> float:
    #     """
    #     calculate the prosumer stock impact during a strategy profile SP^t=strat_{1,s}^t,...,strat_{N,s}^t

    #     Parameters
    #     ----------
    #     period : int
    #         an instance of time t.

    #     Returns
    #     -------
    #     None.

    #     """
        
    #     nextperiod = period if period == self.nbperiod+self.rho-1 else period+1 
        
    #     for i in range(self.prosumers.size):
    #         Si = self.prosumers[i].storage[period]
    #         Si_tplus1 = self.prosumers[i].storage[nextperiod]
    #         QTstock_i = self.prosumers[i].QTStock[period]
            
    #         self.prosumers[i].valStock[period] \
    #             = aux.phiepominus(min( aux.apv(Si_tplus1 - Si), QTstock_i )) \
    #                 - aux.phiepominus(min( aux.apv(Si - Si_tplus1), QTstock_i ))
    
            
    def computeLCost_LCostMinMax(self, period:int):
        """
        Compute the learning Cost of all players, 
        learning (max, min) Cost for all players over the learning steps

        Parameters
        ----------
        period : int
            an instance of time t.

        Returns
        -------
        None.

        """
        for i in range(self.prosumers.size):
            self.prosumers[i].Lcost[period] \
                = aux.apv(self.prosumers[i].price[period] - self.prosumers[i].valStock[period])
                
            if self.prosumers[i].LCostmin["Lcost"] == None \
                or self.prosumers[i].LCostmin["Lcost"] > self.prosumers[i].Lcost[period] :
                self.prosumers[i].LCostmin["Lcost"] = self.prosumers[i].Lcost[period]
                self.prosumers[i].LCostmin["price"] = self.prosumers[i].price[period]
                self.prosumers[i].LCostmin["valStock"] = self.prosumers[i].valStock[period]
                self.prosumers[i].LCostmin["mode"] = self.prosumers[i].mode[period]
                self.prosumers[i].LCostmin["state"] = self.prosumers[i].state[period]
                
            if self.prosumers[i].LCostmax["Lcost"] == None \
                or self.prosumers[i].LCostmax["Lcost"] < self.prosumers[i].Lcost[period] :
                self.prosumers[i].LCostmax["Lcost"] = self.prosumers[i].Lcost[period]
                self.prosumers[i].LCostmax["price"] = self.prosumers[i].price[period]
                self.prosumers[i].LCostmax["valStock"] = self.prosumers[i].valStock[period]
                self.prosumers[i].LCostmax["mode"] = self.prosumers[i].mode[period]
                self.prosumers[i].LCostmax["state"] = self.prosumers[i].state[period]
                
            # TODO =====> TODELETE
            # if self.prosumers[i].Lcostmin[period] == 0 \
            #     or self.prosumers[i].Lcostmin[period] > self.prosumers[i].Lcost[period]:
            #         #self.prosumers[i].Lcostmin[period] = self.prosumers[i].Lcost[period]
            #         self.prosumers[i].LCostmin["Lcost"] = self.prosumers[i].Lcost[period]
            #         self.prosumers[i].LCostmin["price"] = self.prosumers[i].price[period]
            #         self.prosumers[i].LCostmin["valStock"] = self.prosumers[i].valStock[period]
            #         self.prosumers[i].LCostmin["mode"] = self.prosumers[i].mode[period]
            #         self.prosumers[i].LCostmin["state"] = self.prosumers[i].state[period]
                    
            # if self.prosumers[i].Lcostmax[period] == 0 \
            #     or self.prosumers[i].Lcostmax[period] < self.prosumers[i].Lcost[period] :
            #         #self.prosumers[i].Lcostmax[period] = self.prosumers[i].Lcost[period]
            #         self.prosumers[i].LCostmax["Lcost"] = self.prosumers[i].Lcost[period]
            #         self.prosumers[i].LCostmax["price"] = self.prosumers[i].price[period]
            #         self.prosumers[i].LCostmax["valStock"] = self.prosumers[i].valStock[period]
            #         self.prosumers[i].LCostmax["mode"] = self.prosumers[i].mode[period]
            #         self.prosumers[i].LCostmax["state"] = self.prosumers[i].state[period]
            
    def computeUtility(self, period:int): 
        """
        Calculate utility function using min, max and last prosumer's Learning cost (LCost)
        
        Parameters
        ----------
        period : int
            an instance of time t.
            
        Returns
        -------
        None.
        
        """
        N = self.prosumers.size
        
        for i in range(N):
            # TODO =====> TODELETE
            # print(f"i={i}, Lcost={round(self.prosumers[i].Lcost[period],2)}, LCostmax={round(self.prosumers[i].LCostmax['Lcost'],2)}, LCostmin={round(self.prosumers[i].LCostmin['Lcost'], 2)}")
            # if (self.prosumers[i].LCostmax !=0 or self.prosumers[i].LCostmin != 0):
            #     self.prosumers[i].utility[period] \
            #         = (self.prosumers[i].LCostmax["Lcost"] - self.prosumers[i].Lcost[period]) \
            #             / (self.prosumers[i].LCostmax["Lcost"] - self.prosumers[i].LCostmin["Lcost"])
            # else:
            #     self.prosumers[i].utility[period] = 0
                
            # print(f"i={i}, LCostmax={self.prosumers[i].LCostmax != 0 }")
            
            if self.prosumers[i].LCostmax["Lcost"] != self.prosumers[i].LCostmin["Lcost"]:
                # print(f"t={period}, prosu={i} LCostmax={self.prosumers[i].LCostmax['Lcost']}, LCostmin={self.prosumers[i].LCostmin['Lcost']}")
                self.prosumers[i].utility[period] \
                    = (self.prosumers[i].LCostmax["Lcost"] - self.prosumers[i].Lcost[period]) \
                        / (self.prosumers[i].LCostmax["Lcost"] - self.prosumers[i].LCostmin["Lcost"])
            else:
                self.prosumers[i].utility[period] = 0
        
    ###########################################################################
    #                   compute smartgrid variables :: end
    ###########################################################################    
        
    ###########################################################################
    #                   compute actors' repartition gains :: start
    #                       repart, shapley, UCB
    ########################################################################### 
    def computeRepart(self, period:int, mu:float):
        """
        compute the part of the gain of each actor
        
        Parameters
        ----------
        period : int
            an instance of time t
            
        mu: float (mu in [0,1])
            a input parameter of the game

        Returns
        -------
        np.array(N,1) : float
        
        """
        N = self.prosumers.size
        
        part1 = mu * (self.Reduct[period] / N )
        
        
        for i in range(N):
            frac = (self.Reduct[period] * self.prosumers[i].prodit[period]) / max(1, self.insg[period])
             
            self.prosumers[i].Repart[period] = part1 + (1-mu) * frac
        

    ###########################################################################
    #                   compute actors' repartition gains :: end
    ########################################################################### 
    
    
    ###########################################################################
    #                       update prosumers variables:: start
    ###########################################################################
    def updateState(self, period:int): 
        """
        Change prosumer's state based on its production, comsumption and available storage
        
        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        
        
        """
        N = self.prosumers.size
        
        for i in range(N):    
            if self.prosumers[i].production[period] >= self.prosumers[i].consumption[period] :
                self.prosumers[i].state[period] = ag.State.SURPLUS
            
            elif self.prosumers[i].production[period] + self.prosumers[i].storage[period] >= self.prosumers[i].consumption[period] :
                self.prosumers[i].state[period] = ag.State.SELF
            
            else :
                self.prosumers[i].state[period] = ag.State.DEFICIT
                
    def updateSmartgrid(self, period:int): 
        """
        Update storage , consit, prodit based on mode and state
        
        Parameters
        ----------
        period : int
            an instance of time t

        Returns
        -------
        
        """
        N = self.prosumers.size
        
        nextperiod = period if period == self.nbperiod+self.rho-1 else period+1
        
        for i in range(N):
            if self.prosumers[i].state[period] == ag.State.DEFICIT:
                self.prosumers[i].prodit[period] = 0
                if self.prosumers[i].mode[period] == ag.Mode.CONSPLUS:
                    self.prosumers[i].storage[nextperiod] = 0
                    self.prosumers[i].consit[period] = self.prosumers[i].consumption[period] - (self.prosumers[i].production[period] + self.prosumers[i].storage[period])
                
                else :
                    self.prosumers[i].storage[nextperiod] = self.prosumers[i].storage[period]
                    self.prosumers[i].consit[period] = self.prosumers[i].consumption[period] - self.prosumers[i].production[period]
            
            elif self.prosumers[i].state[period] == ag.State.SELF:
                self.prosumers[i].prodit[period] = 0
                
                if self.prosumers[i].mode[period] == ag.Mode.CONSMINUS:
                    self.prosumers[i].storage[nextperiod] = self.prosumers[i].storage[period]
                    self.prosumers[i].consit[period] = self.prosumers[i].consumption[period] - self.prosumers[i].production[period]
                
                else :
                    self.prosumers[i].storage[nextperiod] = self.prosumers[i].storage[period] - (self.prosumers[i].consumption[period] - self.prosumers[i].production[period])
                    self.prosumers[i].consit[period] = 0
            else :
                self.prosumers[i].consit[period] = 0
                
                if self.prosumers[i].mode[period] == ag.Mode.DIS:
                    self.prosumers[i].storage[nextperiod] = min(self.prosumers[i].smax,self.prosumers[i].storage[period] +\
                                                                   (self.prosumers[i].production[period] - self.prosumers[i].consumption[period]))
                    self.prosumers[i].prodit[period] = aux.apv(self.prosumers[i].production[period] - self.prosumers[i].consumption[period] -\
                                                                (self.prosumers[i].smax - self.prosumers[i].storage[period] ))
                else:
                    self.prosumers[i].storage[nextperiod] = self.prosumers[i].storage[period]
                    self.prosumers[i].prodit[period] = self.prosumers[i].production[period] - self.prosumers[i].consumption[period]
    
    def updateProbaLRI(self, period:int, slowdown:float): 
        """
        Update probability for LRI based mode choice
        
        Parameters
        ----------
        period : int
            an instance of time t.
            
        slowdown : float
            a learning parameter called slowdown factor 0 <= b <= 1
            
        Returns
        -------
        None.
        """
        N = self.prosumers.size
        
        for i in range(N):
            if self.prosumers[i].state[period] == ag.State.SURPLUS:
                if self.prosumers[i].mode[period] == ag.Mode.DIS :
                    self.prosumers[i].prmode[period][0] = min(1, self.prosumers[i].prmode[period][0] + slowdown * self.prosumers[i].utility[period] * (1 - self.prosumers[i].prmode[period][0]))
                    self.prosumers[i].prmode[period][1] = 1 - self.prosumers[i].prmode[period][0]
                
                else :
                    self.prosumers[i].prmode[period][1] = min(1, self.prosumers[i].prmode[period][1] + slowdown * self.prosumers[i].utility[period] * (1 - self.prosumers[i].prmode[period][1]))
                    self.prosumers[i].prmode[period][0] = 1 - self.prosumers[i].prmode[period][1]
                    
            elif self.prosumers[i].state[period] == ag.State.SELF:
                if self.prosumers[i].mode[period] == ag.Mode.DIS :
                    self.prosumers[i].prmode[period][0] = min(1,self.prosumers[i].prmode[period][0] + slowdown * self.prosumers[i].utility[period] * (1 - self.prosumers[i].prmode[period][0]))
                    self.prosumers[i].prmode[period][1] = 1 - self.prosumers[i].prmode[period][0]
                
                else :
                    self.prosumers[i].prmode[period][1] = min(1,self.prosumers[i].prmode[period][1] + slowdown * self.prosumers[i].utility[period] * (1 - self.prosumers[i].prmode[period][1]))
                    self.prosumers[i].prmode[period][0] = 1 - self.prosumers[i].prmode[period][1]
            else :
                if self.prosumers[i].mode[period] == ag.Mode.CONSPLUS :
                    self.prosumers[i].prmode[period][0] = min(1,self.prosumers[i].prmode[period][0] + slowdown * self.prosumers[i].utility[period] * (1 - self.prosumers[i].prmode[period][0]))
                    self.prosumers[i].prmode[period][1] = 1 - self.prosumers[i].prmode[period][0]
                
                else :
                    self.prosumers[i].prmode[period][1] = min(1,self.prosumers[i].prmode[period][1] + slowdown * self.prosumers[i].utility[period] * (1 - self.prosumers[i].prmode[period][1]))
                    self.prosumers[i].prmode[period][0] = 1 - self.prosumers[i].prmode[period][1]
    
    def updateModeLRI(self, period:int, threshold:float): 
        """
        Update mode using rules from LRI
        
        Parameters
        ----------
        period : int
            an instance of time t.
            
        threshold : float
            a parameter for which a we stop learning when prabability mode is greater than threshold
            threshold in [0,1]
            
        Returns
        -------
        None.
        """
        N = self.prosumers.size
        
        for i in range(N):
            rand = rdm.uniform(0,1)
            
            if self.prosumers[i].state[period] == ag.State.SURPLUS:
                if (rand <= self.prosumers[i].prmode[period][0] and self.prosumers[i].prmode[period][1] < threshold) or self.prosumers[i].prmode[period][0] > threshold :
                    self.prosumers[i].mode[period] = ag.Mode.DIS
                
                else :
                    self.prosumers[i].mode[period] = ag.Mode.PROD
            
            elif self.prosumers[i].state[period] == ag.State.SELF :
                if (rand <= self.prosumers[i].prmode[period][0] and self.prosumers[i].prmode[period][1] < threshold) or self.prosumers[i].prmode[period][0] > threshold :
                    self.prosumers[i].mode[period] = ag.Mode.DIS
                
                else :
                    self.prosumers[i].mode[period] = ag.Mode.CONSMINUS
            
            else :
                if (rand <= self.prosumers[i].prmode[period][0] and self.prosumers[i].prmode[period][1] < threshold) or self.prosumers[i].prmode[period][0] > threshold :
                    self.prosumers[i].mode[period] = ag.Mode.CONSPLUS
                else :
                    self.prosumers[i].mode[period] = ag.Mode.CONSMINUS
                    
    
    def updateModeSyA(self, period:int): 
        """
        Update mode using rules from SyA algortihm
        
        """
        N = self.prosumers.size
        
        for i in range(N):
            if self.prosumers[i].state[period] == ag.State.DEFICIT :
                self.prosumers[i].mode[period] = ag.Mode.CONSPLUS
                
            elif self.prosumers[i].state[period] == ag.State.SELF :
                self.prosumers[i].mode[period] = ag.Mode.DIS
                
            else :
                self.prosumers[i].mode[period] = ag.Mode.DIS
    
    def updateModeCSA(self, period:int): 
        """
        Update mode using rules from CSA algortihm
        
        """
        N = self.prosumers.size
        
        for i in range(N):
            if self.prosumers[i].state[period] == ag.State.DEFICIT :
                self.prosumers[i].mode[period] = ag.Mode.CONSMINUS
                
            elif self.prosumers[i].state[period] == ag.State.SELF :
                self.prosumers[i].mode[period] = ag.Mode.CONSPLUS
                
            else :
                self.prosumers[i].mode[period] = ag.Mode.PROD
                
    
    def updateModeSSA(self, period:int):
        """
        Update Mode using the self stock algorithm (SSA)
        
        before executing this function, running computeXi from agents.py
        
        """
        for i in range(self.prosumers.size):
            
            QTStock_i = self.prosumers[i].QTStock[period]
            if self.prosumers[i].state[period] == ag.State.DEFICIT :
                self.prosumers[i].mode[period] = ag.Mode.CONSPLUS
                
            elif self.prosumers[i].state[period] == ag.State.SELF :
                self.prosumers[i].mode[period] = ag.Mode.DIS
                
            elif self.prosumers[i].storage[period] >= QTStock_i:
                self.prosumers[i].mode[period] = ag.Mode.PROD
            else:
                self.prosumers[i].mode[period] = ag.Mode.DIS
    
    
    ###########################################################################
    #                       update prosumers variables:: end
    ###########################################################################
    