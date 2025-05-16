#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 17:41:39 2025

@author: willy
"""
import os
import glob
import json
import time
import pandas as pd


from bokeh.layouts import layout
from bokeh.plotting import figure, show, output_file, save
from bokeh.transform import factor_cmap
from bokeh.transform import dodge
from bokeh.palettes import Spectral5
from bokeh.models import ColumnDataSource
from bokeh.models import FactorRange
from bokeh.models import Legend
from bokeh.models import HoverTool


###############################################################################
#                   CONSTANTES: debut
###############################################################################
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,examine,help"
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
    
    assert df.shape == (151200, 60), "**** VERY BAD df.shape != (151200, 60) ****"
    
    return df


def scenarioDataResult(df, scenarioCorePathDataViz, filename="dataframes.csv"):
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

###############################################################################
#                       load  files : FIN
###############################################################################

###############################################################################
#                   visu bar plot ValSG and ValNoSG : debut
###############################################################################
def plot_performanceMeasures(df_valSG, df_valNoSG):
    """
    plot performance measures 

    Parameters
    ----------
    df_valSG : TYPE
        DESCRIPTION.
    df_valNoSG : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #--------------------------- version 2/04/2025 : debut ----------------------------
    df_valSG['mu_rho_epsilon_lambdaPoisson_lr'] = df_valSG.apply(lambda row: f"mu{row['mu']}_rho{row['rho']}_epsilon{row['epsilon']}_lambdaPoisson{row['lambda_poisson']}_lr{row['learning_rate']}", axis=1)
    #df_valSG['rho_mu_lr'] = df_valSG.apply(lambda row: f"rho{row['rho']}_mu{row['mu']}_lr{row['learning_rate']}", axis=1)
    # df_valNoSG['rho_mu_lr'] = df_valNoSG.apply(lambda row: f"rho{row['rho']}_mu{row['mu']}_lr{row['learning_rate']}", axis=1)
    
    df_valSG_NoSG = pd.merge(df_valSG, df_valNoSG, 
                              on=["algoName","mu","rho","epsilon","epsilon","lambda_poisson","learning_rate"], 
                              how="inner")
    
    pivot_df_ValSG = df_valSG_NoSG.pivot(index="algoName", columns="mu_rho_epsilon_lambdaPoisson_lr", values="ValSG")
    pivot_df_ValNoSG = df_valSG_NoSG.pivot(index="algoName", columns="mu_rho_epsilon_lambdaPoisson_lr", values="ValNoSG")
    
    pivot_df_ValSG_T = pivot_df_ValSG.T
    pivot_df_ValSG_T = pivot_df_ValSG_T.sort_values('LRI_REPART')
    pivot_df_ValSG = pivot_df_ValSG_T.T
    pivot_df_ValNoSG_T = pivot_df_ValNoSG.T
    pivot_df_ValNoSG_T = pivot_df_ValNoSG_T.sort_values('LRI_REPART')
    pivot_df_ValNoSG = pivot_df_ValNoSG_T.T
    
    algoNames = pivot_df_ValSG.index.tolist()
    cols = pivot_df_ValSG.columns.tolist()
    
    dico_valSG = dict(); dico_ValNoSG = dict()
    dico_valSG["cols"] = cols; dico_ValNoSG["cols"] = cols
    for algoName in algoNames:
        dico_valSG[algoName] = list(pivot_df_ValSG.loc[algoName,:])
        dico_ValNoSG[algoName] = list(pivot_df_ValNoSG.loc[algoName,:])
        
    source_valSG = ColumnDataSource(data=dico_valSG)
    source_valNoSG = ColumnDataSource(data=dico_ValNoSG)
    
    p_perf_valSG = figure(x_range=cols, title="ValSG", height=450, toolbar_location=None, tools=TOOLS)
    p_perf_valNoSG = figure(x_range=cols, title="ValNoSG", height=450, toolbar_location=None, tools=TOOLS)
    
    
    p_perf_valSG.add_layout(Legend(), 'right')
    p_perf_valNoSG.add_layout(Legend(), 'right')
    
    # Ajouter les barres pour rho, mu et valeur
    for i in range(len(algoNames)):
        p_perf_valSG.vbar(x=dodge("cols", -0.15*(-1+i*1), range=p_perf_valSG.x_range), 
                          top=algoNames[i], source=source_valSG,
                          width=0.10, color=COLORS[i], legend_label=algoNames[i])
        p_perf_valNoSG.vbar(x=dodge("cols", -0.15*(-1+i*1), range=p_perf_valSG.x_range), 
                          top=algoNames[i], source=source_valNoSG,
                          width=0.10, color=COLORS[i], legend_label=algoNames[i])
        
    
    
    # Personnalisation du graphique
    p_perf_valSG.xgrid.grid_line_color = None
    p_perf_valSG.xaxis.major_label_orientation = 1.2
    p_perf_valSG.legend.click_policy="hide"
    
    p_perf_valNoSG.xgrid.grid_line_color = None
    p_perf_valNoSG.xaxis.major_label_orientation = 1.2
    p_perf_valNoSG.legend.click_policy="hide"
    
    
    # Ajouter un outil de survol
    # hover = HoverTool(tooltips=[
    #     ("algoName", "@algoNames"),
    #     ("col", "@cols"),
    #     #("Valeur", "@value"),
    #     ("Valeur", "@y")
        
    #     # ("Rho", "@Rho"),
    #     # ("Mu", "@Mu"),
    #     # ("Learning Rate", "@LearningRate"),
    #     # ("Valeur", "@value_valSG"),
    # ])
    
    hover = HoverTool(tooltips=[
        ("algoName", "@algoName"),
        ("col", "@col"),
        ("Valeur", "@y"),
        # Ajout des valeurs mu, rho, epsilon, lambda_poisson, learning_rate
        ("Paramètres", "@cols"),  # Affiche le paramètre (ex: mu_rho_epsilon_lambdaPoisson_lr)
        # Pour afficher les valeurs spécifiques, il faudrait les inclure dans le ColumnDataSource
    ])
    
    p_perf_valSG.add_tools(hover)
    p_perf_valNoSG.add_tools(hover)
    
    return p_perf_valSG, p_perf_valNoSG
    
        
    ##--------------------------- version 2/04/2025: FIN ----------------------------
    
    
    # df_valSG = df_valSG.reset_index()
    # df_valSG['rho_mu_lr'] = df_valSG.apply(lambda row: f"rho{row['rho']}_mu{row['mu']}_lr{row['learning_rate']}", axis=1)
    # # df_valNoSG['rho_mu_lr'] = df_valNoSG.apply(lambda row: f"rho{row['rho']}_mu{row['mu']}_lr{row['learning_rate']}", axis=1)
    
    # df_valSG_NoSG = pd.merge(df_valSG, df_valNoSG, 
    #                          on=["algoName", "rho", "mu","learning_rate"], 
    #                          how="inner")
    
    # pivot_df_ValSG = df_valSG_NoSG.pivot(index="algoName", columns="rho_mu_lr", values="ValSG")
    # pivot_df_ValNoSG = df_valSG_NoSG.pivot(index="algoName", columns="rho_mu_lr", values="ValNoSG")
    
    # algoNames = pivot_df_ValSG.index.tolist()
    # cols = pivot_df_ValSG.columns.tolist()
    
    
    # dico = dict()
    # dico["cols"] = []
    # dico["algoNames"] = []
    # dico["value"] = []
    
    # for i, algoName in enumerate(algoNames):
    #     for col in cols:
    #         dico["cols"].append(col)
    #         dico["algoNames"].append(algoName)
    #         # Récupérer la valeur réelle pour cette combinaison
    #         value = pivot_df_ValSG.loc[algoName, col]
    #         dico["value"].append(value)
    
    # source = ColumnDataSource(data=dico)
    
    # p_perf = figure(x_range=cols, 
    #                 title="ValSG", height=450, toolbar_location=None, tools="")
    
    # for i in range(len(algoNames)):
    #     p_perf.vbar(x=dodge("cols", -0.15*(-1+i*1), range=p_perf.x_range), top=algoNames[i], source=source,
    #        width=0.10, color=COLORS[i], legend_label=algoNames[i])
    
    # # Personnalisation du graphique
    # p_perf.xgrid.grid_line_color = None
    # p_perf.xaxis.major_label_orientation = 1.2
    # p_perf.legend.click_policy="hide"
    
    
    
    
    # # Maintenant, votre HoverTool peut afficher les valeurs correctement
    # hover = HoverTool(tooltips=[
    #     ("algoName", "@algoNames"),
    #     ("col", "@cols"),
    #     ("Valeur", "@value")
    # ])
    
    # p_perf.add_tools(hover)
    
    ##--------------------------------- OLD Version -----------------------------------------
    
    # df_valSG = df_valSG.reset_index()
    # df_valSG['rho_mu_lr'] = df_valSG.apply(lambda row: f"rho{row['rho']}_mu{row['mu']}_lr{row['learning_rate']}", axis=1)
    # # df_valNoSG['rho_mu_lr'] = df_valNoSG.apply(lambda row: f"rho{row['rho']}_mu{row['mu']}_lr{row['learning_rate']}", axis=1)
    
    # df_valSG_NoSG = pd.merge(df_valSG, df_valNoSG, 
    #                           on=["algoName", "rho", "mu","learning_rate"], 
    #                           how="inner")
    
    # pivot_df_ValSG = df_valSG_NoSG.pivot(index="algoName", columns="rho_mu_lr", values="ValSG")
    # pivot_df_ValNoSG = df_valSG_NoSG.pivot(index="algoName", columns="rho_mu_lr", values="ValNoSG")
    
    # algoNames = pivot_df_ValSG.index.tolist()
    # cols = pivot_df_ValSG.columns.tolist()
    
    # dico_valSG = dict(); dico_ValNoSG = dict()
    # dico_valSG["cols"] = cols; dico_ValNoSG["cols"] = cols
    # for algoName in algoNames:
    #     dico_valSG[algoName] = list(pivot_df_ValSG.loc[algoName,:])
    #     dico_ValNoSG[algoName] = list(pivot_df_ValNoSG.loc[algoName,:])
        
    # source_valSG = ColumnDataSource(data=dico_valSG)
    # source_valNoSG = ColumnDataSource(data=dico_ValNoSG)
    
    # p_perf_valSG = figure(x_range=cols, title="ValSG", height=450, toolbar_location=None, tools=TOOLS)
    # p_perf_valNoSG = figure(x_range=cols, title="ValNoSG", height=450, toolbar_location=None, tools=TOOLS)
    
    
    # p_perf_valSG.add_layout(Legend(), 'right')
    # p_perf_valNoSG.add_layout(Legend(), 'right')
    
    # # Ajouter les barres pour rho, mu et valeur
    # for i in range(len(algoNames)):
    #     p_perf_valSG.vbar(x=dodge("cols", -0.15*(-1+i*1), range=p_perf_valSG.x_range), 
    #                       top=algoNames[i], source=source_valSG,
    #                       width=0.10, color=COLORS[i], legend_label=algoNames[i])
    #     p_perf_valNoSG.vbar(x=dodge("cols", -0.15*(-1+i*1), range=p_perf_valSG.x_range), 
    #                       top=algoNames[i], source=source_valNoSG,
    #                       width=0.10, color=COLORS[i], legend_label=algoNames[i])
        
    
    
    # # Personnalisation du graphique
    # p_perf_valSG.xgrid.grid_line_color = None
    # p_perf_valSG.xaxis.major_label_orientation = 1.2
    # p_perf_valSG.legend.click_policy="hide"
    
    # p_perf_valNoSG.xgrid.grid_line_color = None
    # p_perf_valNoSG.xaxis.major_label_orientation = 1.2
    # p_perf_valNoSG.legend.click_policy="hide"
    
    
    # # Ajouter un outil de survol
    # hover = HoverTool(tooltips=[
    #     ("algoName", "@algoNames"),
    #     ("col", "@cols"),
    #     #("Valeur", "@value"),
    #     ("Valeur", "@y")
        
    #     # ("Rho", "@Rho"),
    #     # ("Mu", "@Mu"),
    #     # ("Learning Rate", "@LearningRate"),
    #     # ("Valeur", "@value_valSG"),
    # ])
    
    # p_perf_valSG.add_tools(hover)
    # p_perf_valNoSG.add_tools(hover)
    
    # return p_perf_valSG, p_perf_valNoSG
    
        
    # ##------------------------------OLD Version --------------------------------------------
    
        
    # source = ColumnDataSource(data=dict(
    #             # algoNames=algoNames,
    #             # cols=cols,
    #             # value_valSG=pivot_df_ValSG.values,
    #             # value_valNoSG=pivot_df_ValNoSG.values,
    #             # Rho=df_valSG_NoSG["rho"].unique(),
    #             # Mu=df_valSG_NoSG["mu"].unique(),
    #             # LearningRate=df_valSG_NoSG["learning_rate"].unique()
                
    #             ))
    
    # p_perf = figure(x_range=cols, 
    #                 y_range=(0, 10), 
    #                 title="rho_mu_lr by algoName", height=350, toolbar_location=None, tools="")
    
    # # Ajouter les barres pour rho, mu et valeur
    # p_perf.vbar(x=dodge(cols, -0.25, range=p_perf.x_range), top=algoNames[0], source=source,
    #    width=0.2, color="#c9d9d3", legend_label="2015")

    # p_perf.vbar(x=dodge(cols,  0.0,  range=p_perf.x_range), top=algoNames[1], source=source,
    #        width=0.2, color="#718dbf", legend_label="2016")
    
    # p_perf.vbar(x=dodge(cols,  0.25, range=p_perf.x_range), top=algoNames[2], source=source,
    #        width=0.2, color="#e84d60", legend_label="2017")
    
    
    # # Ajouter un outil de survol
    # hover = HoverTool(tooltips=[
    #     ("algoName", "@algoNames"),
    #     ("col", "@col"),
    #     ("Valeur", "@value_valSG"),
        
    #     # ("Rho", "@Rho"),
    #     # ("Mu", "@Mu"),
    #     # ("Learning Rate", "@LearningRate"),
    #     # ("Valeur", "@value_valSG"),
    # ])
    
    # p_perf.add_tools(hover)
        
    
    # return p_perf
    
    
#     # Extraire les données pour ValNoSG
#     data_valnsg = df_valNoSG.to_dict()['ValNoSG'] # dico['ValNoSG']
    
#     # Créer des listes pour stocker les valeurs
#     names = []
#     rhos = []
#     mus = []
#     learning_rates = []
#     values = []
    
#     # Itérer sur les clés et valeurs
#     for key, value in data_valnsg.items():
#         name = key[0]
#         rho = key[2]
#         mu = key[3]
#         learning_rate = key[1]
#         names.append(name)
#         rhos.append(rho)
#         mus.append(mu)
#         learning_rates.append(learning_rate)
#         values.append(value)
    
#     # Créer un ColumnDataSource
#     source = ColumnDataSource(data=dict(
#         Name=names,
#         Rho=rhos,
#         Mu=mus,
#         LearningRate=learning_rates,
#         Value=values
#     ))
    
#     # Créer le diagramme en batons
#     p = figure(x_range=names, title="Diagramme en batons pour ValNoSG", x_axis_label='Nom', y_axis_label='Valeur')
    
#     # Ajouter les barres pour rho, mu et valeur
#     p.vbar(x='Name', top='Value', width=0.2, color='blue', legend_label='Valeur', source=source)
# #    p.vbar(x='Name', top=[float(r) for r in rhos], width=0.2, color='red', legend_label='Rho', source=source, offset=-0.1)
# #    p.vbar(x='Name', top=[float(m) for m in mus], width=0.2, color='green', legend_label='Mu', source=source, offset=0.1)
    
    # # Ajouter un outil de survol
    # hover = HoverTool(tooltips=[
    #     ("Nom", "@Name"),
    #     ("Rho", "@Rho"),
    #     ("Mu", "@Mu"),
    #     ("Learning Rate", "@LearningRate"),
    #     ("Valeur", "@Value"),
    # ])
    
    # p.add_tools(hover)
        
    
    # return p
    
###############################################################################
#                   visu bar plot ValSG and ValNoSG : Fin
###############################################################################


###############################################################################
#                   visu all plots with mean LRI : debut
###############################################################################
def plot_all_figures_withMeanLRI(df: pd.DataFrame, scenarioCorePathDataViz: str): 
    """
    plot all figures from requests of latex document

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
        
    scenarioCorePathDataViz: str
        DESCRIPTION.
        
    Returns
    -------
    None.

    """
    filename="dataframes.csv"
    
    df, df_valSG_T, df_valNoSG_T, df_valSG, df_valNoSG \
        = scenarioDataResult(df, scenarioCorePathDataViz, filename=filename)
        
    plot_perf_valSG, plot_perf_valNoSG = plot_performanceMeasures(df_valSG, df_valNoSG)
    
    # plotValSG, plotValNoSG = plot_curve_valSGNoSG(df_prosumers, scenarioCorePathDataViz)
    
    # plotLcost = plot_LcostPrice(df_prosumers, scenarioCorePathDataViz)
    
    # plotQTstock = plot_sumQTStock(df_prosumers, scenarioCorePathDataViz)
    
    # plotSis = plot_sumStorage(df_prosumers, scenarioCorePathDataViz)
    
    # #plotBarMode = plot_barModes(df_prosumers, scenarioCorePathDataViz)
    
    # plotBarModeBis = plot_barModesBis(df_prosumers, scenarioCorePathDataViz)
    
    # plotQttepo = plotQTTepo(df_prosumers, scenarioCorePathDataViz)
    
    # plot_Perf = plot_performanceAlgo(df_prosumers, scenarioCorePathDataViz)
    
    # plots_list = plotQTTepo_t_minus_plus(df_prosumers, scenarioCorePathDataViz)
    
    # plot_Perf_MeanLri = plot_performanceAlgo_meanLRI(scenarioCorePathDataViz)
    
    # ps_X_Y_ai = plot_X_Y_ai(scenarioCorePathDataViz)
    
    # plot_NE_brute = plot_nashEquilibrium_byPeriod(scenarioCorePathDataViz, M_execution_LRI)
    
    # p_distr = plot_min_proba_distribution(df_prosumers, scenarioCorePathDataViz)
    
    # create a layout
    lyt = layout(
        [
            [plot_perf_valSG], 
            [plot_perf_valNoSG],
            # [plotValSG], 
            # [plotValNoSG], 
            # [plotLcost],
            # [plotQTstock],
            # [plotSis], 
            # # [plotBarMode], 
            # [plotBarModeBis], 
            # [plotQttepo], 
            # # [plot_Perf], 
            # [plot_Perf_MeanLri],
            # plots_list, 
            # ps_X_Y_ai, 
            # plot_NE_brute, 
            # p_distr
            # # [p1, p2],  # the first row contains two plots, spaced evenly across the width of notebook
            # # [p3],  # the second row contains only one plot, spanning the width of notebook
        ],
        sizing_mode="stretch_width",  # the layout itself stretches to the width of notebook
    )
    
    # set output to static HTML file
    filename = os.path.join(scenarioCorePathDataViz, "plotCourbes_mus_lrs_rhos.html")
    output_file(filename=filename, title="Static HTML file")
    
    save(lyt)
    
###############################################################################
#                   visu all plots with mean LRI : FIN
###############################################################################

if __name__ == '__main__':

    
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate.json"
    scenarioFile = "./data_scenario_JeuDominique/dataFromQuentinAutomate_test.json"
    
    scenario = None
    with open(scenarioFile) as file:
        scenario = json.load(file)
        
    # scenario_dir = f"{scenario['scenarioName']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"
    # #scenario_dir = os.path.join(scenario["scenarioPath"], scenario_dir)
    # print(f"{scenario_dir}")
    
    scenario_dir = f"{scenario['scenarioName']}_N{scenario['instance']['N_actors']}T{scenario['simul']['nbPeriod']}K{scenario['algo']['LRI_REPART']['maxstep']}"
    folder_2_search = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataResult")
    folder_2_save = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataViz")
    filename_csv = "dataframes.csv"
    print(f"{folder_2_search}")
    
    df = find_csvfile(folder_2_search=folder_2_search, folder_2_save=folder_2_save, filename_csv=filename_csv)
    
    scenarioCorePathDataViz = os.path.join(scenario["scenarioPath"], scenario_dir, "datas", "dataViz")
    
    plot_all_figures_withMeanLRI(df=df, scenarioCorePathDataViz=scenarioCorePathDataViz)