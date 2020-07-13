import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import svd_tools as svdt
import data_analysis as da

DATA_PATH = '../data/covid data/'

df_covid_deaths = pd.read_csv(DATA_PATH + 'covid_deaths_usafacts.csv', sep=',')

no_dc = df_covid_deaths[df_covid_deaths['State'] != 'DC']
states = list(map(lambda x: x.lower(), no_dc['State'].drop_duplicates().values))
print(states)

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

    return state_deaths, list(map((lambda x: x.replace(' County', '')), state_counties)) 

def state_deaths_plot(ax, state, chosen=None, labels=False):
    state_deaths, county_names = get_state_deaths(state)
    
    da.lines_plot(
        ax        = ax,
        data      = state_deaths, 
        title     = '%s Daily COVID-19 Deaths by County' % state.upper(),
        xlabel    = 'Days since 1/22/20',
        ylabel    = 'Total Deaths',
        axis      = 'rows',
        labels    = da.filter_labels(county_names, chosen) if labels else None
    )


def state_deaths_svd_plots(fig, state, centering='s', style='lines', chosen=None, labels=False):
    state_deaths, county_names = get_state_deaths(state)
    
    da.svd_plots(
        fig       = fig,
        data      = state_deaths, 
        title     = '%s Daily COVID-19 Deaths by County' % state.upper(),
        xlabel    = 'Days since 1/22/20',
        ylabel    = 'Total Deaths',
        centering = centering,
        style     = style,
        axis      = 'rows',
        labels    = da.filter_labels(county_names, chosen) if labels else None
    )

def state_deaths_svd_plots_four(fig, state, centering='s', style='lines', chosen=None, labels=False):
    state_deaths, county_names = get_state_deaths(state)
    
    da.svd_plots_four(
        fig       = fig,
        data      = state_deaths, 
        title     = '%s Daily COVID-19 Deaths by County' % state.upper(),
        xlabel    = 'Days since 1/22/20',
        ylabel    = 'Total Deaths',
        centering = centering,
        style     = style,
        axis      = 'rows',
        labels    = da.filter_labels(county_names, chosen) if labels else None
    )

def state_deaths_svd_plots_four_resid(fig, state, centering='s', style='lines', chosen=None, labels=False):
    state_deaths, county_names = get_state_deaths(state)
    
    da.svd_plots_four_resid(
        fig       = fig,
        data      = state_deaths, 
        title     = '%s Daily COVID-19 Deaths by County' % state.upper(),
        xlabel    = 'Days since 1/22/20',
        ylabel    = 'Total Deaths',
        centering = centering,
        style     = style,
        axis      = 'rows',
        labels    = da.filter_labels(county_names, chosen) if labels else None
    )


def state_deaths_svd_plots_three(fig, state, centering='s', style='lines', chosen=None, labels=False):
    state_deaths, county_names = get_state_deaths(state)
    
    da.svd_plots_three(
        fig       = fig,
        data      = state_deaths, 
        title     = '%s Daily COVID-19 Deaths by County' % state.upper(),
        xlabel    = 'Days since 1/22/20',
        ylabel    = 'Total Deaths',
        centering = centering,
        style     = style,
        axis      = 'rows',
        labels    = da.filter_labels(county_names, chosen) if labels else None
    )

def state_deaths_svd_plots_five(fig, state, centering='s', style='lines', chosen=None, labels=False):
    state_deaths, county_names = get_state_deaths(state)
    
    da.svd_plots_five(
        fig       = fig,
        data      = state_deaths, 
        title     = '%s Daily COVID-19 Deaths by County' % state.upper(),
        xlabel    = 'Days since 1/22/20',
        ylabel    = 'Total Deaths',
        centering = centering,
        style     = style,
        axis      = 'rows',
        labels    = da.filter_labels(county_names, chosen) if labels else None
    )



def generate_state_svd_plots_four():
    # Drop DC
    no_dc = df_covid_deaths[df_covid_deaths['State'] != 'DC']
    
    states = list(map(lambda x: x.lower(), no_dc['State'].drop_duplicates().values))
    print(states)
    for state in states:
        print(state)
        fig = plt.figure()
        state_deaths_svd_plots_four(fig=fig, state=state)
        fig.set_size_inches(18, 5)
        fig.tight_layout()
        plt.savefig('out/covid/%s_deaths_svds.png' % state, dpi=192, bbox_inches='tight')
        plt.close()

def generate_state_deaths_plots():
    # Drop DC
    no_dc = df_covid_deaths[df_covid_deaths['State'] != 'DC']
    
    states = list(map(lambda x: x.lower(), no_dc['State'].drop_duplicates().values))
    for state in states:
        print(state)
        fig, ax = plt.subplots()
        state_deaths_plot(ax=ax, state=state)
        fig.tight_layout()
        plt.savefig('out/covid/%s_deaths.png' % state, dpi=192, bbox_inches='tight')
        plt.close()


# -----------------------------------------------
# Exploration

nyc_counties = ['Kings', 'Queens', 'Bronx', 'New York', 'Richmond']
fl_counties_special = ['Miami-Dade', 'Broward', 'Palm Beach', 'Pinellas']

# Clearer plot for Florida
