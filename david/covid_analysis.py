import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import svd_tools as svdt
import data_analysis as da

DATA_PATH = '../data/covid data/'

'''
Returns a matrix giving the total deaths over days for each county in a a state 
and an array with the corresponding county names
'''
def get_state_deaths(state):
    df_covid_deaths = pd.read_csv(DATA_PATH + 'covid_deaths_usafacts.csv', sep=',')
    df_state_deaths = df_covid_deaths[df_covid_deaths['State'] == state.upper()]
    df_state_counties = df_state_deaths['County Name']

    state_counties = df_state_counties.values
    state_deaths = df_state_deaths.loc[:, '1/22/20':].values

    return state_deaths, state_counties


def state_deaths_svd_plots(state, centering='s', style='lines'):
    state_deaths, county_names = get_state_deaths(state)
    
    da.svd_plots(
        data      = state_deaths, 
        title     = '%s Daily COVID-19 Deaths by County' % state.upper(),
        xlabel    = 'Days since 1/22/20',
        ylabel    = 'Total Deaths',
        centering = centering,
        style     = style,
        axis      = 'rows',
        labels    = county_names
    )
