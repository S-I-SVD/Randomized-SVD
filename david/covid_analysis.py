import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import svd_tools as svdt
import data_analysis as da

DATA_PATH = '../data/covid data/'

df_covid_deaths = pd.read_csv(DATA_PATH + 'covid_deaths_usafacts.csv', sep=',')
df_covid_cases = pd.read_csv(DATA_PATH + 'covid_deaths_usafacts.csv', sep=',')
dates = df_covid_cases.loc[:, '1/22/20':].columns.values

#no_dc = df_covid_deaths[df_covid_deaths['State'] != 'DC']
#states = list(map(lambda x: x.lower(), no_dc['State'].drop_duplicates().values))

'''
Returns a matrix giving the total deaths over days for each county in a a state 
and an array with the corresponding county names
'''
def get_state_deaths(state, days_offset=0, new=False):
    df_covid_deaths = pd.read_csv(DATA_PATH + 'covid_deaths_usafacts.csv', sep=',')
    df_state_deaths = df_covid_deaths[df_covid_deaths['State'] == state.upper()]
    df_state_counties = df_state_deaths['County Name']

    state_counties = df_state_counties.values
    state_deaths_ = df_state_deaths.loc[:, '1/22/20':].values

    if new:
        state_deaths = da.consecutive_differences(state_deaths_)
    else:
        state_deaths = state_deaths_

    state_deaths = state_deaths[:, days_offset:]

    return state_deaths, list(map((lambda x: x.replace(' County', '')), state_counties)) 

def get_state_cases(state, days_offset=0, new=False):
    df_covid_cases = pd.read_csv(DATA_PATH + 'covid_confirmed_usafacts.csv', sep=',')
    df_state_cases = df_covid_cases[df_covid_cases['State'] == state.upper()]
    df_state_counties = df_state_cases['County Name']

    state_counties = df_state_counties.values
    state_cases_ = df_state_cases.loc[:, '1/22/20':].values

    if new:
        state_cases = da.consecutive_differences(state_cases_)
    else:
        state_cases = state_cases_

    state_cases = state_cases[:, days_offset:]

    return state_cases, list(map((lambda x: x.replace(' County', '')), state_counties)) 



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


def state_deaths_svd_plots(fig, state, centering='s', style='lines', chosen=None, labels=False, days_offset=0):
    state_deaths, county_names = get_state_deaths(state, days_offset)
    
    da.svd_plots(
        fig       = fig,
        data      = state_deaths, 
        title     = '%s Daily COVID-19 Deaths by County' % state.upper(),
        #xlabel    = 'Days since 1/22/20',
        xlabel    = 'Days since %s' % dates[days_offset],
        ylabel    = 'Total Deaths',
        centering = centering,
        style     = style,
        axis      = 'rows',
        labels    = da.filter_labels(county_names, chosen) if labels else None
    )

    ax_list = fig.axes
    for ax in ax_list:
        for date in fl_closure_dates.keys():
            ax.axvline(x = date, color='red', alpha=.75, linestyle = '--')
        for date in fl_opening_dates.keys():
            ax.axvline(x = date, color='blue', alpha=.75, linestyle = '--')

def state_cases_svd_plots(fig, state, centering='s', style='lines', chosen=None, labels=False, days_offset=0, new=False, scale=None):
    state_cases_, county_names = get_state_cases(state, days_offset)

    if new:
        ylabel = 'New Cases'
        state_cases = da.consecutive_differences(state_cases_, axis='c')
    else:
        ylabel = 'Total Cases'
        state_cases = state_cases_
        
    print(state_cases.shape)

    if scale == 'log':
        state_cases = np.log(state_cases - np.min(state_cases) + 1)
        ylabel = r"$\log(1 + %s)$" % ylabel

    da.svd_plots(
        fig       = fig,
        data      = state_cases, 
        title     = '%s Daily COVID-19 Cases by County' % state.upper(),
        #xlabel    = 'Days since 1/22/20',
        xlabel    = 'Days since %s' % dates[days_offset],
        ylabel    = ylabel,
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
fl_counties_beaches = ['Miami-Dade', 'Broward', 'Palm Beach']
fl_beaches = ['Miami-Dade', 'Broward', 'Palm Beach']
fl_counties_2 = ['Miami-Dade', 'Broward', 'Palm Beach', 'Orange', 'Hillsborough']

fl_closure_dates = {
        68 : 'Broward, Miami-Dade, Palm Beach, Monroe SAH orders', # Mar 30
        70 : 'State-wide SAH order', # Apr 1
        156 : 'Shut down bars' # Jun 26
        }

fl_opening_dates = {
        100 : 'Open trails, some beaches (limited)', # May 1
        133 : 'Bars, movie theaters, etc open at half capacity', # Jun 3
        141 : 'Communities can reopen education systems' # Jun 11
        }

# Scree plots for paper
def paper_cases_scree_plots():
    fig, axs = plt.subplots(1, 2)
    cases_cumulative, _ = get_state_cases(state='fl', days_offset=14, new=False)
    cases_new, _ = get_state_cases(state='fl', days_offset=14, new=False)
    print(cases_cumulative)
    svdt.scree_plot(ax=axs[0], mat=cases_cumulative, num_sv=10, 
            title='Scree Plot: Daily Cumulative FL COVID-19 Cases')
    svdt.scree_plot(ax=axs[1], mat=cases_new, num_sv=10, 
            title='Scree Plot: Daily New FL COVID-19 Cases')

    fig.set_size_inches(10, 5)
    fig.tight_layout()
    fig.savefig('out/covid/cases/fl_cases_scree_plots.png', bbox_inches='tight', dpi=192, 
            layout='landscape')
    fig.show()

def paper_fl_cases_cumulative_svd_plots():
    fig = plt.figure()
    state_cases_svd_plots(fig=fig, state='fl', days_offset=14, labels=True,
            chosen=fl_counties_2)
    fig.set_size_inches(15, 10)
    fig.tight_layout()
    fig.savefig('out/covid/cases/fl_cases_cumulative_svd_plots.png', bbox_inches='tight', 
            dpi=192, layout='landscape')
    fig.show()

def paper_fl_cases_svd_plots(fig, centering='s', new=False, scale=None):
    #fig = plt.figure()
    state_cases_svd_plots(fig=fig, state='fl', days_offset=0, labels=True,
            chosen=fl_beaches, scale=scale, centering=centering, new=new)
    ax_list = fig.axes
    for ax in ax_list:
        for date in fl_closure_dates.keys():
            ax.axvline(x = date, color='red', alpha=.75, linestyle = '--')
        for date in fl_opening_dates.keys():
            ax.axvline(x = date, color='blue', alpha=.75, linestyle = '--')
    #fig.set_size_inches(15, 10)
    #fig.savefig('out/covid/cases/fl_log_cases_cumulative_svd_plots.png', bbox_inches='tight', dpi=192, layout='landscape')
    #fig.show()

def log_cases_scree_plot(state, scale=None, new=False):
    fig, ax = plt.subplots()
    cases, _ = get_state_cases(state, new=new)
    svdt.scree_plot(ax, cases, 10, title='Scree Plot: %s COVID-19 Cases' % state.upper(), scale=scale)
    fig.set_size_inches(5, 5)
    fig.tight_layout()
    #fig.savefig('out/covid/cases/fl_log_cases_cumulative_svd_plots_shift_7.png', bbox_inches='tight', dpi=192, layout='landscape')
    fig.show()


def paper_fl_log_cases_scree_plot():
    fig, ax = plt.subplots()
    fl_cases, _ = get_state_cases('fl')
    svdt.scree_plot(ax, fl_cases, 10, title='Scree Plot: Florida COVID-19 Log Cumulative Known Cases', scale='log')
    fig.set_size_inches(5, 5)
    fig.tight_layout()
    fig.savefig('out/covid/cases/fl_log_cases_scree.png', bbox_inches='tight', dpi=192, layout='landscape')
    fig.show()

def paper_fl_cases_scree_plot():
    fig, ax = plt.subplots()
    fl_cases, _ = get_state_cases('fl')
    svdt.scree_plot(ax, fl_cases, 10, 
            title='Scree Plot: FL COVID-19 Cumulative Known Cases', 
            scale='')
    fig.set_size_inches(5, 5)
    fig.tight_layout()
    fig.savefig('out/covid/cases/fl_cases_scree_plot.png', bbox_inches='tight', dpi=192, layout='landscape')
    fig.show()
    
def paper_fl_deaths_scree_plot():
    fig, ax = plt.subplots()
    fl_cases, _ = get_state_deaths('fl')
    svdt.scree_plot(ax, fl_cases, 10, 
            title='Scree Plot: FL COVID-19 Cumulative Deaths', 
            scale='')
    fig.set_size_inches(5, 5)
    fig.tight_layout()
    fig.savefig('out/covid/fl_deaths_scree_plot.png', bbox_inches='tight', dpi=192, layout='landscape')
    fig.show()





def paper_fl_cases_new_svd_plots():
    fig = plt.figure()
    state_cases_svd_plots(fig=fig, state='fl', days_offset=14, labels=True,
            chosen=fl_counties_2, new=True)
    fig.set_size_inches(15, 10)
    fig.tight_layout()
    fig.savefig('out/covid/cases/fl_cases_new_svd_plots.png', bbox_inches='tight', 
            dpi=192, layout='landscape')
    fig.show()
