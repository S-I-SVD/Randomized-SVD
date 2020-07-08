import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import svd_tools as svdt
import data_analysis as da

DATA_PATH = '../data/covid data/'

df_covid_deaths = pd.read_csv(DATA_PATH + 'covid_deaths_usafacts.csv', sep=',')
df_fl_deaths = df_covid_deaths[df_covid_deaths['State'] == 'FL']
df_fl_counties = df_covid_deaths['County Name']

fl_counties = df_fl_counties.values
fl_deaths = df_fl_deaths.loc[:, '1/22/20':].values


def plot_fl_deaths_surfaces():
    da.plot_svd_surfaces(
            data   = fl_deaths, 
            title  = 'FL Daily COVID-19 Deaths by County',
            xlabel = 'Days since 1/22/20',
            ylabel = 'County',
            zlabel = 'Total Deaths'
            )

