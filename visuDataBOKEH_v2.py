#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:02:00 2025

@author: willy
"""
import os
import glob
import json
import time
import pandas as pd
import itertools as it
import numpy as np


from bokeh.layouts import layout
from bokeh.plotting import figure, show, output_file, save
from bokeh.transform import factor_cmap
from bokeh.transform import dodge
from bokeh.palettes import Spectral5, HighContrast3, Category10

from bokeh.models import ColumnDataSource
from bokeh.models import FactorRange
from bokeh.models import Legend
from bokeh.models import HoverTool


###############################################################################
#                   CONSTANTES: debut
###############################################################################
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
COLORS = {"SyA":"gray", "Bestie":"red", "CSA":"yellow", "SSA":"green", "LRI_REPART":"blue"}

TOOLS="""hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,
            reset,tap,save,box_select,poly_select,lasso_select,examine,help"""
TOOLS_MODES = "hover, tap, reset, save, pan, wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo, box_select"
            
TOOLTIPS_Val_SG_NoSG = [ ("value", "$y{(0,0)}"), ]
TOOLTIPS_LCOST = [("value", "$y{(0,0)}")]
TOOLTIPS_MODES = [("value", "$y{.5,3}")]
TOOLTIPS_STATE_MODES = "$name @period: @$name"
TOOLTIPS_XY_ai = [("value", "$y{(0.1,1)}")]
###############################################################################
#                   CONSTANTES: FIN
###############################################################################


###############################################################################
#                       load  files : debut
###############################################################################
def find_csvfile(folder_2_search:str, folder_2_save:str, filename_csv:str="dataframes.csv"):
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
    filename_csv : str, optional
        DESCRIPTION. The default is "dataframes.csv".

    Returns
    -------
    None.

    """
    
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
    
    # assert df.shape == (151200, 60), "**** VERY BAD df.shape != (151200, 60) ****"
    
    return df


def OLD_scenarioDataResult(df, scenarioCorePathDataViz, filename="dataframes.csv"):
    """
    extract data from csv dataset  

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    scenarioCorePathDataViz : TYPE
        DESCRIPTION.
    filename : TYPE, optional
        DESCRIPTION. The default is "dataframes.csv".

    Returns
    -------
    df : pd.DataFrame
        DESCRIPTION.
    df_valSG_T : pd.DataFrame
        DESCRIPTION.
    df_valNoSG_T : pd.DataFrame
        DESCRIPTION.
    df_valSG : pd.DataFrame
        DESCRIPTION.
    df_valNoSG : pd.DataFrame
        DESCRIPTION.

    """
    if df is None:
        df = pd.read_csv( os.path.join(scenarioCorePathDataViz, filename) )
    
    cols_to_sel_valSG = ["algoName", "mu", "rho", "epsilon", "lambda_poisson", "learning_rate", "period", "M_exec_lri", "ValSG"]
    cols_100_valSG = cols_to_sel_valSG[:-2]
    df_valSG_T = df[cols_to_sel_valSG].groupby(cols_100_valSG).mean().reset_index()
    
    cols_to_sel_valNoSG = ["algoName", "mu", "rho", "epsilon", "lambda_poisson", "learning_rate", "period", "M_exec_lri", "ValNoSG"]
    cols_100_valNoSG = cols_to_sel_valNoSG[:-2]
    df_valNoSG_T = df[cols_to_sel_valNoSG].groupby(cols_100_valNoSG).mean().reset_index()
    
    
    cols_valSG = df_valSG_T.columns.tolist()
    items_2_remove_ValSG = ["period", "ValSG"]
    cols_100_valSG = [col for col in cols_valSG if col not in items_2_remove_ValSG]
    df_valSG = df_valSG_T.groupby(cols_100_valSG).sum().reset_index()
    
    cols_valNoSG = df_valNoSG_T.columns.tolist()
    items_2_remove_VaNolSG = ["period", "ValNoSG"]
    cols_100_valNoSG = [col for col in cols_valNoSG if col not in items_2_remove_VaNolSG]
    df_valNoSG = df_valNoSG_T.groupby(cols_100_valNoSG).sum().reset_index()
    
                
    # df_valNoSG_T = df[['algoName','rho',"mu",'learning_rate','valNoSG_i','period','prosumers']]\
    #                 .drop_duplicates().drop(['prosumers'], axis=1)\
    #                 .groupby(['algoName','rho',"mu",'learning_rate','period']).sum()
                    
    
    # df_valSG = df[["algoName","period","ValSG","rho","mu","learning_rate"]]\
    #             .groupby(["algoName","rho","mu","learning_rate","period"]).mean()\
    #             .groupby(["algoName","rho","mu","learning_rate"]).sum()
    
    # df_valNoSG = df[['algoName','rho',"mu",'learning_rate','valNoSG_i','period','prosumers']]\
    #                 .drop_duplicates().drop(['prosumers'], axis=1)\
    #                 .groupby(['algoName','rho',"mu",'learning_rate','period']).sum()\
    #                 .groupby(['algoName','rho',"mu",'learning_rate']).sum()\
    #                 .rename(columns={"valNoSG_i":"ValNoSG"})
                    
    return df, df_valSG_T, df_valNoSG_T, df_valSG, df_valNoSG

def scenarioDataResult(df_rho_mu_epsilon_lambda: pd.DataFrame):
    """
    select ValSG and ValNoSG

    Parameters
    ----------
    df_rho_mu_epsilon_lambda : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ##### valSG and valnoSG by period
    cols_to_sel_valSG = ["algoName", "mu", "rho", "epsilon", "lambda_poisson", 
                         "learning_rate", "period", "M_exec_lri", "ValSG"]
    cols_100_valSG = cols_to_sel_valSG[:-2]
    df_valSG_T = df_rho_mu_epsilon_lambda[cols_to_sel_valSG].groupby(cols_100_valSG).mean().reset_index()
    
    cols_to_sel_valNoSG = ["algoName", "mu", "rho", "epsilon", "lambda_poisson", 
                           "learning_rate", "period", "M_exec_lri", "ValNoSG"]
    cols_100_valNoSG = cols_to_sel_valNoSG[:-2]
    df_valNoSG_T = df_rho_mu_epsilon_lambda[cols_to_sel_valNoSG].groupby(cols_100_valNoSG).mean().reset_index()
    
    ##### valSG by state and period
    cols_to_sel_valSG = ["algoName", "mu", "rho", "epsilon", "lambda_poisson", 
                         "learning_rate", "period", "state", "M_exec_lri", "ValSG"]
    cols_100_valSG = cols_to_sel_valSG[:-2]
    df_valSG_State_T = df_rho_mu_epsilon_lambda[cols_to_sel_valSG].groupby(cols_100_valSG).mean().reset_index()
    
    
    cols_valSG = df_valSG_T.columns.tolist()
    items_2_remove_ValSG = ["period", "ValSG"]
    cols_100_valSG = [col for col in cols_valSG if col not in items_2_remove_ValSG]
    df_valSG = df_valSG_T.groupby(cols_100_valSG).sum().reset_index()
    
    cols_valNoSG = df_valNoSG_T.columns.tolist()
    items_2_remove_VaNolSG = ["period", "ValNoSG"]
    cols_100_valNoSG = [col for col in cols_valNoSG if col not in items_2_remove_VaNolSG]
    df_valNoSG = df_valNoSG_T.groupby(cols_100_valNoSG).sum().reset_index()
    
    return df_valSG_T, df_valNoSG_T, df_valSG, df_valNoSG, df_valSG_State_T

###############################################################################
#                       load  files : FIN
###############################################################################

###############################################################################
#                   visu bar plot ValSG and ValNoSG : Debut
###############################################################################
def plot_performanceMeasures(df_valSG: pd.DataFrame, df_valNoSG: pd.DataFrame):
    """
    plot performances Measures for various learning rates for one value of rho, mu, epsilon

    Parameters
    ----------
    df_valSG : pd.DataFrame
        DESCRIPTION.
    df_valNoSG : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    plot_Perf_MeanLri_s = []
    
    learning_rates = df_valSG.learning_rate.unique()
    for learning_rate in learning_rates:
        
        df_lr_valSG = df_valSG[df_valSG.learning_rate == learning_rate]
        df_lr_valNoSG = df_valNoSG[ df_valNoSG.learning_rate == learning_rate]
        
        df_lr_valSG_NoSG = pd.merge(df_lr_valSG, df_lr_valNoSG, 
                              on=["algoName","mu","rho","epsilon","lambda_poisson",
                                  "learning_rate", "M_exec_lri", "period"], 
                              how="inner")
        
        data = { 
            'algoName': list(df_lr_valSG_NoSG['algoName']),
            'ValSG': list(df_lr_valSG_NoSG['ValSG']),
            'ValNoSG': list(df_lr_valSG_NoSG['ValNoSG'])
            }
        source = ColumnDataSource(data=data)
        
        plot_Perf_MeanLri = figure(x_range=data["algoName"], title=f"Performance Measures from learning rate {learning_rate}",
                    height=450, toolbar_location=None, tools="hover",
                    tooltips="$name @algoName: @$name")
    
        plot_Perf_MeanLri.vbar(x=dodge('algoName', -0.25, 
                                        range=plot_Perf_MeanLri.x_range), 
                                top='ValSG', source=source,
                                width=0.2, color="green", legend_label="ValSG")
    
        plot_Perf_MeanLri.vbar(x=dodge('algoName',  0.0,  
                                        range=plot_Perf_MeanLri.x_range), 
                                top='ValNoSG', source=source,
                                width=0.2, color="blue", legend_label="ValNoSG")
    
    
        plot_Perf_MeanLri.x_range.range_padding = 0.1
        plot_Perf_MeanLri.xgrid.grid_line_color = None
        plot_Perf_MeanLri.legend.location = "top_left"
        plot_Perf_MeanLri.legend.orientation = "horizontal"
        plot_Perf_MeanLri.legend.click_policy="mute"
    
        hover = HoverTool()
        hover.tooltips = [("Algorithm", "@algoName"), 
                          ("ValSG", "@ValSG"), ("ValNoSG", "@ValNoSG"), 
                          ]
        plot_Perf_MeanLri.add_tools(hover)
        plot_Perf_MeanLri.add_layout(plot_Perf_MeanLri.legend[0], 'right')
    
        plot_Perf_MeanLri_s.append(plot_Perf_MeanLri)
        
    return plot_Perf_MeanLri_s
        
###############################################################################
#                   visu bar plot ValSG and ValNoSG : Fin
###############################################################################

###############################################################################
#                   plot valSG and valNoSG over time : debut
###############################################################################
def plot_curve_valSGNoSG(df_valSG_T: pd.DataFrame, df_valNoSG_T: pd.DataFrame):
    """
    plot ValSG and ValNoSG over time for all algorithms

    Parameters
    ----------
    df_valSG_T : pd.DataFrame
        DESCRIPTION.
    df_valNoSG_T : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    plot_curve_lrs = []
    plotValSG_s, plotValNoSG_s = [], []
    
    learning_rates = df_valSG_T.learning_rate.unique()
    for learning_rate in learning_rates:
        
        df_lr_valSG_T = df_valSG_T[df_valSG_T.learning_rate == learning_rate]
        df_lr_valNoSG_T = df_valNoSG_T[ df_valNoSG_T.learning_rate == learning_rate]
        
        plotValSG = figure(
            title=f"nameScenario: show ValSG Value KPI for all algorithms,  Learning Rate = {learning_rate}",
            height=300,
            sizing_mode="stretch_width",  # use the full width of the parent element
            tooltips=TOOLTIPS_Val_SG_NoSG,
            output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
            tools="pan,box_zoom,reset,save",
            active_drag="box_zoom",  # enable box zoom by default
        )
        
        plotValNoSG = figure(
            title=f"nameScenario: show ValNoSG Value KPI for all algorithms Learning Rate = {learning_rate}",
            height=300,
            sizing_mode="stretch_width",  # use the full width of the parent element
            tooltips=TOOLTIPS_Val_SG_NoSG,
            output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
            tools="pan,box_zoom,reset,save",
            active_drag="box_zoom",  # enable box zoom by default
        )
        
        for algoName in df_lr_valSG_T.algoName.unique().tolist():
            df_lr_valSG_algo = df_lr_valSG_T[df_lr_valSG_T.algoName == algoName]
            plotValSG.line(x=df_lr_valSG_algo["period"], y=df_lr_valSG_algo["ValSG"], 
                           line_width=2, color=COLORS[algoName], alpha=0.8, 
                           legend_label=algoName)
            
            df_valNoSG_algo = df_lr_valNoSG_T[df_lr_valNoSG_T.algoName == algoName]
            plotValNoSG.line(x=df_valNoSG_algo["period"], y=df_valNoSG_algo["ValNoSG"], 
                           line_width=2, color=COLORS[algoName], alpha=0.8, 
                           legend_label=algoName)
            
            
            
        plotValSG.legend.location = "top_left"
        plotValSG.legend.click_policy = "hide"
        plotValNoSG.legend.location = "top_left"
        plotValNoSG.legend.click_policy = "hide"
        
        plotValSG_s.append( plotValSG )
        plotValNoSG_s.append(plotValNoSG)
        
    return plotValSG_s, plotValNoSG_s
###############################################################################
#                   plot valSG and valNoSG over time : Fin
###############################################################################

###############################################################################
#                   plot valSG over time and state : Debut
###############################################################################
def plot_barStack_valSG_over_statePeriod(df_valSG_State_T):
    """
    bar plot over time by state for one algorithm LRI_REPART 

    Parameters
    ----------
    df_valSG_State_T : pd.DataFrame
        list of columns :  ['algoName', 'mu', 'rho', 'epsilon', 'lambda_poisson', 
                            'learning_rate', 'period', 'state', 'M_exec_lri', 'ValSG']

    Returns
    -------
    None.

    """
    plot_ValSG_State_s = []
    
        
    learning_rates = df_valSG_State_T.learning_rate.unique()
    algoNames = df_valSG_State_T.algoName.unique().tolist()
    print(f"AlgoNames = {algoNames}, df_valSG_State_T={df_valSG_State_T.shape}")
    algoNames.remove('Bestie')
    for (learning_rate, algoName) in it.product(learning_rates, algoNames):
        
        df_algo_lr_valSG_State_T = df_valSG_State_T[
                                        (df_valSG_State_T.learning_rate == learning_rate) & 
                                        (df_valSG_State_T.algoName == algoName)]
        
        states = df_algo_lr_valSG_State_T.state.unique().tolist()
        df_pivot = df_algo_lr_valSG_State_T.pivot(index='period',columns='state',values='ValSG').reset_index()
        df_pivot['period'] = df_pivot['period'].astype("string")
        
        data = dict()
        data['periods'] = df_pivot.period.tolist()
        for state in states:
            data[state] = df_pivot[state].tolist()
            
        periods = df_pivot.period.tolist()
        
        
        # Création du plot
        p = figure(x_range=periods, height=550, 
                    title=f"ValSG par State puis Period: {algoName} lr={learning_rate}",
                    toolbar_location=None, tools="pan,box_zoom,reset,save,hover", 
                    tooltips="$name t=@periods: valSG=@$name", 
                    sizing_mode="stretch_width"
                    )
        
        # Barres
        p.vbar_stack(states, x='periods', width=0.9, 
                     color=Category10[10][:len(states)], 
                     source=data,
                     legend_label=states)
        
        # Configuration des axes
        p.y_range.start = 0
        p.x_range.range_padding = 0.1
        p.xaxis.major_label_orientation = 1
        p.xgrid.grid_line_color = None
        p.xaxis.group_label_orientation = 1
        
        plot_ValSG_State_s.append(p)
        
        
    return plot_ValSG_State_s
        
###############################################################################
#                   plot valSG over time and state : Fin
###############################################################################

###############################################################################
#                   plot cumulative valSG over time : Debut
###############################################################################
def plot_cumulative_ValSG(df_rho_mu_epsilon_lambda:pd.DataFrame):
    """
    cumulative ValSG over time

    Parameters
    ----------
    df_rho_mu_epsilon_lambda : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    plotValSG_cumul_s = []
    
    learning_rates = df_rho_mu_epsilon_lambda.learning_rate.unique()
    for learning_rate in learning_rates:
        
        df_lr_algos = df_rho_mu_epsilon_lambda[df_rho_mu_epsilon_lambda.learning_rate == learning_rate]
        
        df_ValSGs = df_lr_algos[["period","algoName","ValSG","M_exec_lri"]]\
                    .groupby(["algoName","period","M_exec_lri"])\
                        .mean().reset_index()\
                    .groupby(["algoName","period"]).sum().reset_index()
                    
        N_prosumers = len(df_lr_algos.prosumers.unique())
        
        mask_100Lri = ~df_ValSGs.astype(str).apply(lambda row: row.str.startswith("LRI")).any(axis=1)
        df_ValSGs_100lri = df_ValSGs[mask_100Lri]
        df_ValSGs_100lri["ValSG"] = df_ValSGs_100lri["ValSG"] * N_prosumers
        
        mask_Lri = df_ValSGs.astype(str).apply(lambda row: row.str.startswith("LRI")).any(axis=1)
        df_ValSGs_lri = df_ValSGs[mask_Lri]
        
        df_ValSGs_merge = pd.concat([df_ValSGs_100lri, df_ValSGs_lri])
        
        # set up the figure
        plotValSGs = figure(
            title=f" show cumulative ValSG for all algorithms from learning rate {learning_rate}",
            height=300,
            sizing_mode="stretch_width",  # use the full width of the parent element
            tooltips=TOOLTIPS_LCOST,
            output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
            tools="pan,box_zoom,reset,save",
            active_drag="box_zoom",  # enable box zoom by default
        )
        
        for algoName in df_ValSGs_merge.algoName.unique().tolist():
            df_ValSGs_algo = df_ValSGs_merge[df_ValSGs_merge.algoName == algoName]
            df_ValSGs_algo["ValSG_cumule"] = df_ValSGs_algo["ValSG"].cumsum()
            plotValSGs.line(x=df_ValSGs_algo["period"], y=df_ValSGs_algo["ValSG_cumule"], 
                             line_width=2, color=COLORS[algoName], alpha=0.8, 
                             legend_label=algoName)
            plotValSGs.scatter(x=df_ValSGs_algo["period"], y=df_ValSGs_algo["ValSG_cumule"],
                                size=2, color=COLORS[algoName], alpha=0.5)
            
        plotValSGs.legend.location = "top_left"
        plotValSGs.legend.click_policy = "hide"
        
        plotValSG_cumul_s.append(plotValSGs)
    
    return plotValSG_cumul_s
###############################################################################
#                   plot cumulative valSG over time : Fin
###############################################################################

###############################################################################
#                   plot QTStock all LRI, SSA, Bestie : debut
###############################################################################
def plot_sumQTStock(df_rho_mu_epsilon_lambda: pd.DataFrame):
    """
    plot Sum of QTStock for all algorithms for one value of rho, mu and epsilon

    Parameters
    ----------
    df_rho_mu_epsilon_lambda : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    plotQTstock_s = []
    
    learning_rates = df_rho_mu_epsilon_lambda.learning_rate.unique()
    for learning_rate in learning_rates:
        
        ### ------------ OLD version : DEBUT ---------------------
        # df_lr_algos = df_rho_mu_epsilon_lambda[df_rho_mu_epsilon_lambda.learning_rate == learning_rate]
        
        # df_QTStock = df_lr_algos[["period", "algoName", "QTStock", "scenarioName"]]\
        #                 .groupby(["algoName", "period"]).sum().reset_index()
                        
        # plotQTstock = figure(
        #     title=f" show QTStock KPI for all algorithms from Learning Rate {learning_rate}",
        #     height=300,
        #     sizing_mode="stretch_width",  # use the full width of the parent element
        #     tooltips=TOOLTIPS_LCOST,
        #     output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
        #     tools="pan,box_zoom,reset,save",
        #     active_drag="box_zoom",  # enable box zoom by default
        # )
        
        # for algoName in df_lr_algos.algoName.unique().tolist():
        #     df_qtstock_algo = df_QTStock[df_QTStock.algoName == algoName]
        #     plotQTstock.line(x=df_qtstock_algo["period"], y=df_qtstock_algo["QTStock"], 
        #                      line_width=2, color=COLORS[algoName], alpha=0.8, 
        #                      legend_label=algoName)
        #     plotQTstock.scatter(x=df_qtstock_algo["period"], y=df_qtstock_algo["QTStock"],
        #                         size=2, color="red", alpha=0.5)
        
        # plotQTstock.legend.location = "top_left"
        # plotQTstock.legend.click_policy = "hide"
        
        # plotQTstock_s.append(plotQTstock)
        ### ------------ OLD version : FIN ---------------------
        
        
        ### ------------ new version : DEBUT ---------------------
        df_lr_algos = df_rho_mu_epsilon_lambda[df_rho_mu_epsilon_lambda.learning_rate == learning_rate]
        
        df_QTStocks = df_lr_algos[["period","algoName","QTStock","M_exec_lri"]]\
                        .groupby(["algoName","period","M_exec_lri"])\
                            .mean().reset_index()\
                        .groupby(["algoName","period"]).sum().reset_index()
                    
        N_prosumers = len(df_lr_algos.prosumers.unique())
        
        mask_100Lri = ~df_QTStocks.astype(str).apply(lambda row: row.str.startswith("LRI")).any(axis=1)
        df_QTStocks_100lri = df_QTStocks[mask_100Lri]
        df_QTStocks_100lri["QTStock"] = df_QTStocks_100lri["QTStock"] * N_prosumers
        
        mask_Lri = df_QTStocks.astype(str).apply(lambda row: row.str.startswith("LRI")).any(axis=1)
        df_QTStocks_lri = df_QTStocks[mask_Lri]
        
        df_QTStocks_merge = pd.concat([df_QTStocks_100lri, df_QTStocks_lri])
        
        # set up the figure
        plotQTStocks = figure(
            title=f" show QTStock KPI for all algorithms from learning rate {learning_rate}",
            height=300,
            sizing_mode="stretch_width",  # use the full width of the parent element
            tooltips=TOOLTIPS_LCOST,
            output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
            tools="pan,box_zoom,reset,save",
            active_drag="box_zoom",  # enable box zoom by default
        )
        
        for algoName in df_QTStocks_merge.algoName.unique().tolist():
            df_QTStocks_algo = df_QTStocks_merge[df_QTStocks_merge.algoName == algoName]
            plotQTStocks.line(x=df_QTStocks_algo["period"], y=df_QTStocks_algo["QTStock"], 
                             line_width=2, color=COLORS[algoName], alpha=0.8, 
                             legend_label=algoName)
            plotQTStocks.scatter(x=df_QTStocks_algo["period"], y=df_QTStocks_algo["QTStock"],
                                size=2, color=COLORS[algoName], alpha=0.5)
            
        plotQTStocks.legend.location = "top_left"
        plotQTStocks.legend.click_policy = "hide"
        
        plotQTstock_s.append(plotQTStocks)
        ### ------------ new version : FIN ---------------------
    
    return plotQTstock_s
###############################################################################
#                   plot QTStock all LRI, SSA, Bestie : Fin
###############################################################################

###############################################################################
#                   plot sum storage all LRI, SSA, Bestie : debut
###############################################################################
def plot_sumStorage(df_rho_mu_epsilon_lambda:pd.DataFrame):
    """
    plot storage over time for all algorithms for one value of rho, mu, epsilon

    Parameters
    ----------
    df_rho_mu_epsilon_lambda : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """

    plotSis_s = []
    
    learning_rates = df_rho_mu_epsilon_lambda.learning_rate.unique()
    for learning_rate in learning_rates:
        
        df_lr_algos = df_rho_mu_epsilon_lambda[df_rho_mu_epsilon_lambda.learning_rate == learning_rate]
        
        df_Sis = df_lr_algos[["period","algoName","storage","M_exec_lri"]]\
                    .groupby(["algoName","period","M_exec_lri"])\
                        .mean().reset_index()\
                    .groupby(["algoName","period"]).sum().reset_index()
                    
        N_prosumers = len(df_lr_algos.prosumers.unique())
        
        mask_100Lri = ~df_Sis.astype(str).apply(lambda row: row.str.startswith("LRI")).any(axis=1)
        df_Sis_100lri = df_Sis[mask_100Lri]
        df_Sis_100lri["storage"] = df_Sis_100lri["storage"] * N_prosumers
        
        mask_Lri = df_Sis.astype(str).apply(lambda row: row.str.startswith("LRI")).any(axis=1)
        df_Sis_lri = df_Sis[mask_Lri]
        
        df_Sis_merge = pd.concat([df_Sis_100lri, df_Sis_lri])
        
        # set up the figure
        plotSis = figure(
            title=f" show Storage KPI for all algorithms from learning rate {learning_rate}",
            height=300,
            sizing_mode="stretch_width",  # use the full width of the parent element
            tooltips=TOOLTIPS_LCOST,
            output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
            tools="pan,box_zoom,reset,save",
            active_drag="box_zoom",  # enable box zoom by default
        )
        
        for algoName in df_Sis_merge.algoName.unique().tolist():
            df_Sis_algo = df_Sis_merge[df_Sis_merge.algoName == algoName]
            plotSis.line(x=df_Sis_algo["period"], y=df_Sis_algo["storage"], 
                             line_width=2, color=COLORS[algoName], alpha=0.8, 
                             legend_label=algoName)
            plotSis.scatter(x=df_Sis_algo["period"], y=df_Sis_algo["storage"],
                                size=2, color="red", alpha=0.5)
            
        plotSis.legend.location = "top_left"
        plotSis.legend.click_policy = "hide"
        
        plotSis_s.append(plotSis)
    
    return plotSis_s
    
###############################################################################
#                   plot sum storage all LRI, SSA, Bestie : Fin
###############################################################################

###############################################################################
#                   visu bar plot of actions(modes) : debut
###############################################################################
def plot_barModesBis(df_rho_mu_epsilon_lambda:pd.DataFrame):
    """
    plot distribution of value modes for all algorihtms for one value of rho, mu, epsilon

    Parameters
    ----------
    df_rho_mu_epsilon_lambda : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    plotBarModes = []
    learning_rates = df_rho_mu_epsilon_lambda.learning_rate.unique()
    for learning_rate in learning_rates:
        df_lr_algos = df_rho_mu_epsilon_lambda[df_rho_mu_epsilon_lambda.learning_rate == learning_rate]
        
        # value_counts
        df_res = df_lr_algos.groupby('algoName')[['period','mode']].value_counts().unstack(fill_value=0)
        
        #Affichage du résultat stochastique
        df_stoc = df_res.div(df_res.sum(axis=1), axis=0).reset_index()
    
        
        # rename columns 
        df_stoc = df_stoc.rename(columns={"Mode.CONSMINUS":"CONSMINUS", 
                                          "Mode.CONSPLUS":"CONSPLUS", 
                                          "Mode.DIS":"DIS", 
                                          "Mode.PROD":"PROD"})
        
        modes = ["CONSMINUS", "CONSPLUS", "DIS", "PROD"]
        df_stoc["period"] = df_stoc["period"].astype(str)
        factors = list(zip(df_stoc["algoName"], df_stoc["period"]))
        
        source = ColumnDataSource(data=dict(
                    x=factors,
                    CONSMINUS=df_stoc["CONSMINUS"].tolist(),
                    CONSPLUS=df_stoc["CONSPLUS"].tolist(),
                    DIS=df_stoc["DIS"].tolist(),
                    PROD=df_stoc["PROD"].tolist()
                    ))
        
        plotBarMode = figure(x_range=FactorRange(*factors), height=450,
                             toolbar_location=None, tools="", 
                             tooltips=TOOLTIPS_MODES)
    
        plotBarMode.vbar_stack(modes, x='x', width=0.9, alpha=0.5, 
                               color=["blue", "red", "yellow", "cyan"], 
                               source=source,
                               legend_label=modes)
        
        plotBarMode.y_range.start = 0
        plotBarMode.y_range.end = 1
        plotBarMode.x_range.range_padding = 0.1
        plotBarMode.xaxis.major_label_orientation = 1
        plotBarMode.xgrid.grid_line_color = None
        plotBarMode.legend.location = "top_center"
        plotBarMode.legend.orientation = "horizontal"
        plotBarMode.title = f" Distribution des strategies Modes from learning rate {learning_rate}"
        
        plotBarMode.add_layout(Legend(), 'right')
    
        plotBarModes.append(plotBarMode)
        
    return plotBarModes

def plot_barModesBis_refactoring(df_rho_mu_epsilon_lambda:pd.DataFrame):
    """
    refactoring 
    plot distribution of value modes for all algorihtms for one value of rho, mu, epsilon

    Parameters
    ----------
    df_rho_mu_epsilon_lambda : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    plotBarModes = []
    learning_rates = df_rho_mu_epsilon_lambda.learning_rate.unique()
    for learning_rate in learning_rates:
        df_lr_algos = df_rho_mu_epsilon_lambda[df_rho_mu_epsilon_lambda.learning_rate == learning_rate]
        
        df_lr_algos['state'] = df_lr_algos['state'].str.replace('State.', '', regex=False)
        df_lr_algos['mode'] = df_lr_algos['mode'].str.replace('Mode.', '', regex=False)
        
        
        counts = df_lr_algos.groupby(['algoName','period'])['mode'].value_counts().unstack(fill_value=0)
        # Calculer la somme par ligne
        row_sums = counts.sum(axis=1)
        
        # Diviser chaque valeur par la somme de sa ligne, puis multiplier par 100 pour obtenir un pourcentage
        counts_percent = counts.div(row_sums, axis=0) 
        modes = counts_percent.columns.tolist()
        
        # Optionnel : arrondir les valeurs à 2 décimales
        counts_percent = counts_percent.round(2)
        
        counts_percent = counts_percent.reset_index()
        
        
        counts_percent["period"] = counts_percent["period"].astype(str)
        factors = list(zip(counts_percent["algoName"], counts_percent["period"]))
        
        source = ColumnDataSource(data=dict(
                    x=factors,
                    period=counts_percent["period"].tolist(),
                    CONSMINUS=counts_percent["CONSMINUS"].tolist(),
                    CONSPLUS=counts_percent["CONSPLUS"].tolist(),
                    DIS=counts_percent["DIS"].tolist(),
                    PROD=counts_percent["PROD"].tolist()
                    ))
        
        plotBarMode = figure(x_range=FactorRange(*factors), height=450,
                             toolbar_location=None, tools="", 
                             tooltips=TOOLTIPS_STATE_MODES)
    
        
        plotBarMode.vbar_stack(modes, x='x', width=0.9, alpha=0.5, 
                               color=["blue", "red", "yellow", "cyan"], 
                               source=source,
                               legend_label=modes)
        
        plotBarMode.y_range.start = 0
        plotBarMode.y_range.end = 1
        plotBarMode.x_range.range_padding = 0.1
        plotBarMode.xaxis.major_label_orientation = 1
        plotBarMode.xgrid.grid_line_color = None
        plotBarMode.legend.location = "top_center"
        plotBarMode.legend.orientation = "horizontal"
        plotBarMode.title = f" Distribution des strategies Modes from learning rate {learning_rate}"
        
        plotBarMode.add_layout(Legend(), 'right')
    
        plotBarModes.append(plotBarMode)
        
    return plotBarModes
        
        
def plot_barStateModesBis_refactoring(df_rho_mu_epsilon_lambda:pd.DataFrame):
    """
    refactoring 
    plot distribution of value modes by state for all algorihtms for one value of rho, mu, epsilon

    Parameters
    ----------
    df_rho_mu_epsilon_lambda : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    plotBarStateModes = []
    learning_rates = df_rho_mu_epsilon_lambda.learning_rate.unique()
    for learning_rate in learning_rates:
        df_lr_algos = df_rho_mu_epsilon_lambda[df_rho_mu_epsilon_lambda.learning_rate == learning_rate]
        
        df_lr_algos['state'] = df_lr_algos['state'].str.replace('State.', '', regex=False)
        df_lr_algos['mode'] = df_lr_algos['mode'].str.replace('Mode.', '', regex=False)
        
        
        counts = df_lr_algos.groupby(['algoName','period', 'state'])['mode'].value_counts().unstack(fill_value=0)
        # Calculer la somme par ligne
        row_sums = counts.sum(axis=1)
        
        # Diviser chaque valeur par la somme de sa ligne, puis multiplier par 100 pour obtenir un pourcentage
        counts_percent = counts.div(row_sums, axis=0) 
        modes = counts_percent.columns.tolist()
        
        # Optionnel : arrondir les valeurs à 2 décimales
        counts_percent = counts_percent.round(2)
        
        
        modes = counts_percent.columns.tolist()
        counts_percent = counts_percent.reset_index()
        
        plotAlgoBarModes = []
        for algoName in counts_percent.algoName.unique():
            
            counts_percent_al = counts_percent[(counts_percent.algoName == algoName)]
            counts_percent_al["period"] = counts_percent_al["period"].astype(str)
            
            factors = list(zip(counts_percent_al["period"], counts_percent_al["state"]))
            
            source = ColumnDataSource(data=dict(
                        x=factors,
                        period=counts_percent_al["period"].tolist(),
                        CONSMINUS=counts_percent_al["CONSMINUS"].tolist(),
                        CONSPLUS=counts_percent_al["CONSPLUS"].tolist(),
                        DIS=counts_percent_al["DIS"].tolist(),
                        PROD=counts_percent_al["PROD"].tolist()
                        ))
            
            plotBarMode = figure(x_range=FactorRange(*factors), height=450,
                                 toolbar_location=None, 
                                 tools=TOOLS_MODES,
                                 #active_drag="box_zoom",  # enable box zoom by default
                                 tooltips=TOOLTIPS_STATE_MODES)
        
            
            plotBarMode.vbar_stack(modes, x='x', width=0.9, alpha=0.5, 
                                   color=["blue", "red", "yellow", "cyan"], 
                                   source=source,
                                   legend_label=modes)
            
            plotBarMode.y_range.start = 0
            plotBarMode.y_range.end = 1
            plotBarMode.x_range.range_padding = 0.1
            plotBarMode.xaxis.major_label_orientation = 1
            plotBarMode.xgrid.grid_line_color = None
            plotBarMode.legend.location = "top_center"
            plotBarMode.legend.orientation = "horizontal"
            plotBarMode.legend.click_policy = "hide"
            plotBarMode.title = f" Distribution des strategies Modes from {algoName}: lr = {learning_rate}"
            
            plotBarMode.add_layout(Legend(), 'right')
        
            plotAlgoBarModes.append([plotBarMode])
            
        plotBarStateModes.append([plotAlgoBarModes])
        
    return plotBarStateModes
###############################################################################
#                   visu bar plot of actions(modes) : Fin
###############################################################################

###############################################################################
#                   plot QTTepo_t_{plus,minus} all LRI, SSA, Bestie : DEBUT
###############################################################################
def plotQTTepo_t_minus_plus(df_rho_mu_epsilon_lambda:pd.DataFrame):
    """
    plot QttEpo_t_{minus, plus}

    Parameters
    ----------
    df_rho_mu_epsilon_lambda : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    plot_Qttepo_s : TYPE
        DESCRIPTION.

    """
    plot_Qttepo_s = []
    
    learning_rates = df_rho_mu_epsilon_lambda.learning_rate.unique()
    for learning_rate in learning_rates:
        
        ######################    OLD Version : start   ########################
        # df_lr_algos = df_rho_mu_epsilon_lambda[df_rho_mu_epsilon_lambda.learning_rate == learning_rate]
        
        # df_Qttepo = df_lr_algos[['period', 'algoName','prodit', 'consit', 'scenarioName']]\
        #                 .groupby(["algoName","period"]).sum().reset_index()
        # df_Qttepo.rename(columns={"prodit":"insg", "consit":"outsg"}, inplace=True)
        
        # df_Qttepo["Qttepo_t_minus"] = df_Qttepo["outsg"] - df_Qttepo["insg"]
        # df_Qttepo['Qttepo_t_minus'] = df_Qttepo['Qttepo_t_minus'].apply(lambda x: x if x>=0 else 0)
        
        # df_Qttepo["Qttepo_t_plus"] = df_Qttepo["insg"] - df_Qttepo["outsg"]
        # df_Qttepo['Qttepo_t_plus'] = df_Qttepo['Qttepo_t_plus'].apply(lambda x: x if x>=0 else 0)
        
        # plots_Qttepo = []
        # for algoName in df_lr_algos.algoName.unique().tolist():
        #     df_Qttepo_t_algo = df_Qttepo[df_Qttepo.algoName == algoName]
            
        #     plotQttepo_t_algo = figure(
        #         title=f"{algoName} show QttEpo_t^[+,-] KPI for lr={learning_rate}",
        #         height=300,
        #         sizing_mode="stretch_width",  # use the full width of the parent element
        #         tooltips=TOOLTIPS_LCOST,
        #         output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
        #         tools="pan,box_zoom,reset,save",
        #         active_drag="box_zoom",  # enable box zoom by default
        #     )
            
        #     plotQttepo_t_algo.line(x=df_Qttepo_t_algo["period"], 
        #                            y=df_Qttepo_t_algo["Qttepo_t_minus"], 
        #                            line_width=2, color="#00a933", alpha=0.8, 
        #                            legend_label="Qttepo_t_minus")
        #     plotQttepo_t_algo.scatter(x=df_Qttepo_t_algo["period"], 
        #                               y=df_Qttepo_t_algo["Qttepo_t_minus"],
        #                               size=5, color="#00a933", alpha=0.5)
            
        #     plotQttepo_t_algo.line(x=df_Qttepo_t_algo["period"], 
        #                            y=df_Qttepo_t_algo["Qttepo_t_plus"], 
        #                            line_width=2, color="#800080", alpha=0.8, 
        #                            legend_label="Qttepo_t_plus")
        #     plotQttepo_t_algo.scatter(x=df_Qttepo_t_algo["period"], 
        #                               y=df_Qttepo_t_algo["Qttepo_t_plus"],
        #                               size=5, color="#800080", alpha=0.5)
            
            
            
        #     plotQttepo_t_algo.legend.location = "top_left"
        #     plotQttepo_t_algo.legend.click_policy = "hide"
            
        #     plots_Qttepo.append(plotQttepo_t_algo)
            
        # plot_Qttepo_s.append(plots_Qttepo)
        ######################    OLD Version : END   ########################
    
        ######################    New Version : start   ########################
        df_lr_algos = df_rho_mu_epsilon_lambda[df_rho_mu_epsilon_lambda.learning_rate == learning_rate]
        
        
        cols = ["algoName","period","prosumers","prodit","consit","M_exec_lri"]
        df_prodcons = df_lr_algos[cols].groupby(['algoName','period','M_exec_lri'])[["prodit","consit"]].sum()
        df_prodcons.rename(columns={'prodit': 'insg', 'consit':'outsg'}, inplace=True)
        df_prodcons['Qttepo_t_minus'] = df_prodcons['outsg'] - df_prodcons['insg']
        df_prodcons['Qttepo_t_plus'] = df_prodcons['insg'] - df_prodcons['outsg']
        
        df_prodcons['Qttepo_t_minus'] = df_prodcons["Qttepo_t_minus"].where(df_prodcons['Qttepo_t_minus'] >= 0 , 0)
        df_prodcons['Qttepo_t_plus'] = df_prodcons["Qttepo_t_plus"].where(df_prodcons['Qttepo_t_plus'] >= 0 , 0)
        
        df_Qttepo = df_prodcons.groupby(["algoName","period"]).mean().reset_index()
        
        plots_Qttepo = []
        for algoName in df_lr_algos.algoName.unique().tolist():
            df_Qttepo_t_algo = df_Qttepo[df_Qttepo.algoName == algoName]
            
            plotQttepo_t_algo = figure(
                title=f"{algoName} show QttEpo_t^[+,-] KPI for lr={learning_rate}",
                height=300,
                sizing_mode="stretch_width",  # use the full width of the parent element
                tooltips=TOOLTIPS_LCOST,
                output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
                tools="pan,box_zoom,reset,save",
                active_drag="box_zoom",  # enable box zoom by default
            )
            
            plotQttepo_t_algo.line(x=df_Qttepo_t_algo["period"], 
                                   y=df_Qttepo_t_algo["Qttepo_t_minus"], 
                                   line_width=2, color="#00a933", alpha=0.8, 
                                   legend_label="Qttepo_t_minus")
            plotQttepo_t_algo.scatter(x=df_Qttepo_t_algo["period"], 
                                      y=df_Qttepo_t_algo["Qttepo_t_minus"],
                                      size=5, color="#00a933", alpha=0.5)
            
            plotQttepo_t_algo.line(x=df_Qttepo_t_algo["period"], 
                                   y=df_Qttepo_t_algo["Qttepo_t_plus"], 
                                   line_width=2, color="#800080", alpha=0.8, 
                                   legend_label="Qttepo_t_plus")
            plotQttepo_t_algo.scatter(x=df_Qttepo_t_algo["period"], 
                                      y=df_Qttepo_t_algo["Qttepo_t_plus"],
                                      size=5, color="#800080", alpha=0.5)
            
            
            
            plotQttepo_t_algo.legend.location = "top_left"
            plotQttepo_t_algo.legend.click_policy = "hide"
            
            plots_Qttepo.append(plotQttepo_t_algo)
            
        plot_Qttepo_s.append(plots_Qttepo)
        ######################    New Version : END    ########################
    
    return plot_Qttepo_s
###############################################################################
#                   plot QTTepo_t_{plus,minus} all LRI, SSA, Bestie : DEBUT
###############################################################################

###############################################################################
#                   plot som(LCOST/Price) for LRI : debut
###############################################################################
def plot_LcostPrice(df_rho_mu_epsilon_lambda:pd.DataFrame):
    """
    plot Lcost (learning cost) over the time 

    Parameters
    ----------
    df_rho_mu_epsilon_lambda : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    algoName = "LRI_REPART"
    plotLcosts = []
    
    learning_rates = df_rho_mu_epsilon_lambda.learning_rate.unique()
    for learning_rate in learning_rates:
        df_lr_algo = df_rho_mu_epsilon_lambda[(df_rho_mu_epsilon_lambda.learning_rate == learning_rate) & 
                                        (df_rho_mu_epsilon_lambda.algoName == algoName)]
        
        df_lr_algo['lcost_price'] = df_lr_algo['price'] - df_lr_algo['valStock_i']
        df_lr_algo['lcost_price'] = df_lr_algo[df_lr_algo['lcost_price'] < 0] = 0
        df_lr_algo['lcost_price'] = df_lr_algo['lcost_price'] / df_lr_algo['price'] 
        
        df_lcost_price = df_lr_algo[["period", "lcost_price"]].groupby("period").sum()
        df_lcost_price.reset_index(inplace=True)
        
        plotLcost = figure(
            title=f"ratio Lcost by time over time for LRI algorithms from learning rate = {learning_rate}",
            height=300,
            sizing_mode="stretch_width",  # use the full width of the parent element
            tooltips=TOOLTIPS_LCOST,
            output_backend="webgl",  # use webgl to speed up rendering (https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html)
            tools="pan,box_zoom,reset,save",
            active_drag="box_zoom",  # enable box zoom by default
        )
        
        plotLcost.line(x=df_lcost_price["period"], y=df_lcost_price['lcost_price'], 
                       line_width=2, color=COLORS[algoName], alpha=0.8, 
                       legend_label=algoName)
        
        plotLcost.scatter(x=df_lcost_price["period"], y=df_lcost_price['lcost_price'],
                          size=10, color="red", alpha=0.5)
        
        plotLcost.legend.location = "top_left"
    
        plotLcosts.append(plotLcost)
    return plotLcosts
###############################################################################
#                   plot som(LCOST/Price) for LRI : Fin
###############################################################################

###############################################################################
#               visu bar plot array nash equilibrium (NE) : DEBUT
###############################################################################
def plot_nashEquilibrium_byPeriod(df_rho_mu_epsilon_lambda:pd.DataFrame, folder_2_search_LRI:str):
    """
    TODO a refaire 
    plot distribution of nash equilibrium over time

    Parameters
    ----------
    df_rho_mu_epsilon_lambda : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """

    csv_files = glob.iglob(f'{folder_2_search_LRI}/**/*NE_brute.csv', recursive=True); 
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
                
        # print(f"Mu : {valeur_mu}, Rho : {valeur_rho}, Epsilon : {valeur_epsilon},"+
        #       f" Lambda : {valeur_lambda}, LR : {valeur_lr}, M_exec_lri : {valeur_M_exec_lri}")
        
        df = pd.read_csv(file, index_col=0)
        
        df["mu"] = float(valeur_mu); df["rho"] = float(valeur_rho) ; 
        df["epsilon"] = float(valeur_epsilon) ; 
        df["lambda_poisson"] = float(valeur_lambda) ; 
        df["learning_rate"] = float(valeur_lr) ; 
        df["M_exec_lri"] = int(valeur_M_exec_lri)
        
        dfs.append(df)
        
    dfs = pd.concat(dfs)
    
    rho = df_rho_mu_epsilon_lambda.rho.unique()[0]
    mu = df_rho_mu_epsilon_lambda.mu.unique()[0]
    epsilon = df_rho_mu_epsilon_lambda.epsilon.unique()[0]
    lambda_poisson = df_rho_mu_epsilon_lambda.lambda_poisson.unique()[0]
    
    
    dfs_Mexec = dfs[ (dfs.rho==rho) & (dfs.mu==mu) & (dfs.epsilon==epsilon) &
                    (dfs.lambda_poisson==lambda_poisson) ] # & (dfs.=) & (dfs.=) &
    
#    M_exec = len(dfs_Mexec.M_exec_lri.unique())
    
    prosumer_cols = [col for col in dfs_Mexec.columns if col.startswith('prosumer')];
    
    
    plot_NE_brute_s = []
    learning_rates = dfs_Mexec.learning_rate.unique()
    for learning_rate in learning_rates:
        dfs_Mexec_lr = dfs_Mexec[dfs_Mexec.learning_rate == learning_rate]
        dfs_Mexec_lr["compte_1"] = dfs_Mexec_lr[prosumer_cols].apply(lambda row: sum(row == 1), axis=1)
        
        M_exec = len(dfs_Mexec_lr.M_exec_lri.unique())
    
        df_res_lr_cpte1 = dfs_Mexec_lr[["period","compte_1"]].groupby("period").mean().reset_index()
        df_res_lr_cpte1["compte_1"] = df_res_lr_cpte1["compte_1"] / M_exec
        source = ColumnDataSource(df_res_lr_cpte1)
        
        plot_NE_brute = figure(title=f"LRI: Percent of Pure Nash Equilibrium per period for learning rate {learning_rate}",
                               x_axis_label='periods', y_axis_label='Percent', 
                               tooltips=TOOLTIPS_XY_ai)
        
        plot_NE_brute.vbar(x="period", top="compte_1", source=source, width=0.70)
    
        plot_NE_brute_s.append(plot_NE_brute)
    
    return plot_NE_brute_s
        
###############################################################################
#               visu bar plot array nash equilibrium (NE) : Fin
###############################################################################

###############################################################################
#                   visu all plots with mean LRI : debut
###############################################################################
def plot_all_figures_withMeanLRI_One_rhoMuEps(df_rho_mu_epsilon_lambda: pd.DataFrame, 
                                              scenarioCorePathDataViz: str,
                                              rho: int, mu: int, epsilon: int, 
                                              lambda_poisson:str,
                                              folder_2_search_LRI: str): 
    """
    plot all figures from requests of latex document for one value of rho, mu, epsilon, lambda

    Parameters
    ----------
    df_rho_mu_epsilon_lambda : pd.DataFrame
        DESCRIPTION.
        
    scenarioCorePathDataViz: str
        DESCRIPTION.
        
    Returns
    -------
    None.

    """
    filename="dataframes.csv"
    
    df_valSG_T, df_valNoSG_T, df_valSG, df_valNoSG, df_valSG_State_T \
        = scenarioDataResult(df_rho_mu_epsilon_lambda)
        
    plot_Perf_MeanLri_s = plot_performanceMeasures(df_valSG=df_valSG, 
                                                   df_valNoSG=df_valNoSG)
    
    plotValSG_s, plotValNoSG_s = plot_curve_valSGNoSG(df_valSG_T=df_valSG_T, 
                                                  df_valNoSG_T=df_valNoSG_T)
    
    plot_ValSG_State_s = plot_barStack_valSG_over_statePeriod(df_valSG_State_T=df_valSG_State_T)
    
    plotValSG_cumul_s = plot_cumulative_ValSG(df_rho_mu_epsilon_lambda=df_rho_mu_epsilon_lambda)
    
    #plotLcost_s = plot_LcostPrice(df_rho_mu_epsilon_lambda=df_rho_mu_epsilon_lambda)
    
    plotQTstock_s = plot_sumQTStock(df_rho_mu_epsilon_lambda=df_rho_mu_epsilon_lambda)
    
    plotSis_s = plot_sumStorage(df_rho_mu_epsilon_lambda=df_rho_mu_epsilon_lambda)
    
    # #plotBarMode = plot_barModes(df_prosumers, scenarioCorePathDataViz)
    
    plotBarModeBis = plot_barModesBis(df_rho_mu_epsilon_lambda=df_rho_mu_epsilon_lambda)
    
    plotBarModeBis_refact = plot_barModesBis_refactoring(df_rho_mu_epsilon_lambda=df_rho_mu_epsilon_lambda)
    
    plotBarStateModes = plot_barStateModesBis_refactoring(df_rho_mu_epsilon_lambda=df_rho_mu_epsilon_lambda)
    
    # plotQttepo = plotQTTepo(df_prosumers, scenarioCorePathDataViz)
    
    # plot_Perf = plot_performanceAlgo(df_prosumers, scenarioCorePathDataViz)
    
    plot_Qttepo_s = plotQTTepo_t_minus_plus(df_rho_mu_epsilon_lambda=df_rho_mu_epsilon_lambda)
    
    # ps_X_Y_ai = plot_X_Y_ai(scenarioCorePathDataViz)
    
    plot_NE_brute_s = plot_nashEquilibrium_byPeriod(df_rho_mu_epsilon_lambda=df_rho_mu_epsilon_lambda, 
                                                    folder_2_search_LRI=folder_2_search_LRI)
    
    # p_distr = plot_min_proba_distribution(df_prosumers, scenarioCorePathDataViz)
    
    # create a layout
    lyt = layout(
        [
            [plot_Perf_MeanLri_s], 
            [plotValSG_s], 
            [plotValNoSG_s], 
            [plot_ValSG_State_s],
            [plotValSG_cumul_s],
            # [plotLcost_s],
            [plotQTstock_s],
            [plotSis_s], 
            [plotBarModeBis], 
            [plotBarModeBis_refact],
            [plotBarStateModes],
            # [plotQttepo], 
            [plot_Qttepo_s], 
            # # ps_X_Y_ai, 
            plot_NE_brute_s, 
            # # p_distr
            # # # [p1, p2],  # the first row contains two plots, spaced evenly across the width of notebook
            # # # [p3],  # the second row contains only one plot, spanning the width of notebook
        ],
        sizing_mode="stretch_width",  # the layout itself stretches to the width of notebook
    )
    
    # set output to static HTML file
    filename = os.path.join(scenarioCorePathDataViz, f"plotCourbes_rho{rho}_mu{mu}_epsilon{epsilon}_lambda{lambda_poisson}.html")
    output_file(filename=filename, title="Static HTML file")
    
    save(lyt)
    
    
def plot_all_figures_withMeanLRI(df: pd.DataFrame, period_min: int,
                                 scenarioCorePathDataViz: str, 
                                 folder_2_search_LRI: str):
    """
    plot all figures from requests of latex document for all values of rhos, mus, epsilons, lambda_poissons

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    scenarioCorePathDataViz : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    mus = df.mu.unique()
    rhos = df.rho.unique()
    epsilons = df.epsilon.unique()
    lambda_poissons = df.lambda_poisson.unique()
    
    for (rho, mu, epsilon, lambda_poisson) in it.product(rhos, mus, epsilons, lambda_poissons):
        
        df_rho_mu_epsilon_lambda = df[(df.rho==rho) & (df.mu==mu) & 
                                      (df.epsilon==epsilon) & 
                                      (df.lambda_poisson == lambda_poisson) &
                                      (df.period>period_min)]
        plot_all_figures_withMeanLRI_One_rhoMuEps(df_rho_mu_epsilon_lambda=df_rho_mu_epsilon_lambda, 
                                                  scenarioCorePathDataViz=scenarioCorePathDataViz, 
                                                  folder_2_search_LRI=folder_2_search_LRI,
                                                  rho=rho, mu=mu, epsilon=epsilon, 
                                                  lambda_poisson=lambda_poisson)
        
###############################################################################
#                   visu all plots with mean LRI : FIN
###############################################################################


if __name__ == '__main__':

    ti = time.time()
    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate_test.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate_rho5_mu001.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomateMorePeriods.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate100Periods.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate50Periods.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate50PeriodsMultipleParams.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate100PeriodsMultipleParams.json"
    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate50PeriodsMultipleParams.json"
    
    # scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate150Periods.json"
    
    scenario = None
    with open(scenarioFile) as file:
        scenario = json.load(file)
        
    # scenario_dir = f"{scenario['scenarioName']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"
    # #scenario_dir = os.path.join(scenario["scenarioPath"], scenario_dir)
    # print(f"{scenario_dir}")
    
    scenario_dir = f"{scenario['scenarioName']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"
    folder_2_search = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataResult")
    folder_2_search_LRI = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "LRI_REPART")
    folder_2_save = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataViz")
    filename_csv = "dataframes.csv"
    print(f"{folder_2_search}")
    
    df = find_csvfile(folder_2_search=folder_2_search, folder_2_save=folder_2_save, filename_csv=filename_csv)
    
    scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataViz")
    
    plot_all_figures_withMeanLRI(df=df, 
                                 period_min=scenario["simul"]["period_min"],
                                 scenarioCorePathDataViz=scenarioCorePathDataViz, 
                                 folder_2_search_LRI=folder_2_search_LRI)
    
    print(f"runtime = {time.time() - ti}")