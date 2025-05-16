#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon june 3 11:03:54 2024

@author: willy

auxiliary_functions file contains all functions that are used to all project files.
"""

def apv(x): # Return absolute positive value 
    return max(0,x)  
    
def phiepoplus(x, coef=15): 
    """
    x :  benefit function of selling energy to EPO
    coef : parameter of instance
    """
    return x * coef

def phiepominus(x, coef=90): 
    """
    x :  cost function of buying energy from EPO 
    coef : parameter of instance
    """
    return x * coef

