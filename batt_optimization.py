"""
Author: Tari Jung
Project: RRA 2022
Battery resource dispatch using Unit and RIIA data

"""
#%%
import numpy as np
import pandas as pd
import pyomo.environ as pyomo
import os
import datetime
import csv

import matplotlib
matplotlib.use('Agg')  # Need this for PyCharm to turn off interactive mode
import matplotlib.pyplot as plt

# fpath="C:\\Users\\m02520\\Documents\\PythonScripts\\RRA2022\\"
fpath="./"
os.chdir(fpath)

#%%
#############
## Read CSV files

for sy in [2031,2041]:
# for sy in [2031]:
# Read Unit Data
    unit_data = pd.read_excel('RRA_%d_battinput.xlsx' % sy, sheet_name='SampleUnit')
    unit_data = unit_data.iloc[1:] # drop top row
    unit_list = unit_data['Site'].to_list() # Get list of unit names

    # solar_power = pd.read_csv('SampleUnit_MEC_profile.csv')
    # solar_power['DateTime'] = pd.to_datetime(solar_power['DateTime'])

    riia_prices = pd.read_csv('RIIA_Price_Final.csv')

    ## Create a CSV file to save all results
    header = ['date', 'time_interval', 'unit_name', 'soc', 'BtoG', 'GtoB','prices']
    all_results_file = 'RRA%d_RRF_MISO_bat_results_all.csv' % sy

    with open(all_results_file, 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)     # write the header

    #########
    ## Parameter choices
    wthr_yr = 2018 # Select weather year
    sel_area = 'MEC' # Select area

    ren_level_col = 'WgtPrice_30' # select price column for RIIA prices

    # create list of days to simulate
    # sim_days = ['2040-01-01', '2040-06-01', '2040-09-01']
    sim_days = [d.strftime('%Y-%m-%d') for d in pd.date_range('%d-01-01' % sy,'%d-12-31' % sy)]
    # sim_days = [d.strftime('%Y-%m-%d') for d in pd.date_range('%d-06-01' % sy,'%d-06-07' % sy)]

    ###############
    ## Battery assumptions

    ## https://atb.nrel.gov/electricity/2021/utility-scale_battery_storage
    ## O&M costs are 2.5% of capital costs
    ## Capital costs are $382/kWh (assuming 4 hour lithium ion battery)
    ## Thus O&M costs are 0.025 * 381/1000 in $/MWh = 0.009525
    Ecost = 0.009525 # Battery O&M cost

    ## https://www.mdpi.com/2079-9292/9/1/90/htm
    Bcost = 0.04 # Battery degradation cost

    eta_c = round(np.sqrt(0.86)+0.005,3) # Charging efficiency
    eta_d = round(np.sqrt(0.86)-0.005,3) # Discharging efficiency

    n = 24 # number of time intervals or hours

    ###############
    ## Loop over all given units
    for unit_name in unit_list:
        print(unit_name)

        # Get parameters for selected unit
        Emin = unit_data.loc[unit_data['Site'] == unit_name].iloc[0]['Battery Min SoC (4 hour) (MWh)']  # pmax* 0.1 * 4 / 0.86
        Emax = unit_data.loc[unit_data['Site'] == unit_name].iloc[0]['Battery Max SoC (4 hour) (MWh)']  # pmax * 0.9 * 4 / 0.86
        pmax = unit_data.loc[unit_data['Site'] == unit_name].iloc[0][
            'Battery Max Capacity (MW)']  # pmax = Pi * 1.25 (accounting for ILR)
        Pi = unit_data.loc[unit_data['Site'] == unit_name].iloc[0][
            'InterConnectionLimit (MW)']  # Pi = Power limit at point of Interconnection; From EGEAS

        # take average of solar MW to get hourly solar production
        # mean_solar_power = solar_power1[['hourendingest', unit_name]].groupby('hourendingest').mean()
        # mean_solar_power[unit_name] = round(mean_solar_power[unit_name], 3)

        # convert to dict
        # solar_dict = mean_solar_power.to_dict()
        # Ps = solar_dict.get(unit_name)

        # Placeholder for results
        

        # Loop over the days and run hybrid model
        E_last = Emax
        for sel_day in sim_days:

            # create list with selected dates repeated, so it can be reported in results
            date_list=[sel_day for i in range(n)]
            E=[]
            # Psb_list=[]
            # Psg_list=[]
            Pbg_list=[]
            Pgb_list=[]

            sel_day_dt = datetime.datetime.strptime(sel_day, '%Y-%m-%d')
            sel_day_day = pd.to_datetime(sel_day_dt).day
            sel_day_month = pd.to_datetime(sel_day_dt).month

            # solar_power1 = solar_power.loc[solar_power['MarketDate'] == sel_day] # Filter to one day
            # solar_power1 = solar_power1.loc[solar_power1['weatheryear'] == wthr_yr] # Filter to one weather year

            # Filter dataframe to selections
            riia_prices1 = riia_prices.loc[riia_prices['area'] == sel_area]
            riia_prices1 = riia_prices1.loc[riia_prices1['month'] == sel_day_month]
            riia_prices1 = riia_prices1.loc[riia_prices1['day'] == sel_day_day]
            riia_prices1['markethourend'] = riia_prices1['hour']+1
            riia_prices1['markethourend'] = riia_prices1['markethourend'].astype('int')

            # select price column corresponding to assumed penetration level
            prices = riia_prices1[['markethourend', ren_level_col]]
            prices = prices.set_index('markethourend')

            prices_dict = prices.to_dict()
            lmps = prices_dict.get(ren_level_col)


            ####################
            ## Model

            def bat_dispatch(n, lmps,Elast, Emin, Emax, Ecost, Bcost, pmax, eta_c, eta_d, Pi):
                model = pyomo.ConcreteModel()  # Instance of the model

                ## Index sets
                model.T = pyomo.RangeSet(1, n)

                # LMP data
                model.lmp = pyomo.Param(model.T, initialize=lmps)

                # Solar production data
                # model.Ps = pyomo.Param(model.T, initialize=Ps)

                ## Decision variables
                model.E = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)  # SOC of battery
                model.Pbg = pyomo.Var(model.T, bounds=(0, pmax))  # Power: battery to grid i.e. discharging
                model.Pgb = pyomo.Var(model.T, bounds=(0, pmax))  # Power: grid to battery i.e. charging

                ## Objective
                revenue = sum((model.Pbg[t]) * model.lmp[t]  for t in model.T)
                cost = sum(model.E[t] * Ecost for t in model.T) + sum(model.Pbg[t] * Bcost for t in model.T) + sum(
                           model.Pgb[t] * Bcost for t in model.T) + sum(model.Pgb[t]*model.lmp[t] for t in model.T)
                model.profit = pyomo.Objective(expr=revenue - cost, sense=pyomo.maximize)

                ## Battery SOC limits
                model.socmin = pyomo.Constraint(model.T, rule=lambda model, t: model.E[t] >= Emin) # Discharge limit
                model.socmax = pyomo.Constraint(model.T, rule=lambda model, t: model.E[t] <= Emax) # Charge limit

                ## Solar production equation
                # model.solar = pyomo.Constraint(model.T, rule=lambda model, t: model.Psg[t] + model.Psb[t] == Ps[t])

                ## Power injection limit
                # model.injection = pyomo.Constraint(model.T, rule=lambda model, t: model.Psg[t] + model.Pbg[t] <= Pi)
                model.injection = pyomo.Constraint(model.T, rule=lambda model, t: model.Pbg[t] <= Pi)
                model.injection2 = pyomo.Constraint(model.T, rule=lambda model, t: model.Pgb[t] <= Pi)

                ## Charge and discharge non-comp
                model.noncomp = pyomo.Constraint(model.T, rule=lambda model, t: (model.Pbg[t]/pmax + model.Pgb[t]/pmax) <= 1)

                ## SOC inter-temporal equations
                model.cons = pyomo.ConstraintList()

                for t in model.T:
                    if t == 1:
                        model.cons.add(model.E[t] == Elast - (1 / eta_d) * model.Pbg[t] + eta_c*model.Pgb[t]) # Discharge and Charge

                    if t > 1:
                        model.cons.add(model.E[t] == model.E[t-1] - (1 / eta_d) * model.Pbg[t]+ eta_c*model.Pgb[t]) # Discharge and Charge

                        ## Call the solver
                solver = pyomo.SolverFactory('glpk')
                model.dual = pyomo.Suffix(direction=pyomo.Suffix.IMPORT)

                # solver.solve(model) # Doesn't return the solver output to terminal
                solver.solve(model).write()  # Writes the solver output to terminal

                print('Results')

                for n in lmps.keys():
                    #print('Time interval', n, ': Energy ', model.E[n](), 'MW')
                    #print('Time interval', n, ': S to B', model.Psb[n](), 'MW')
                    #print('Time interval', n, ': S to G', model.Psg[n](), 'MW')
                    #print('Time interval', n, ': B to G', model.Pbg[n](), 'MW')
                    E.append(model.E[n]())
                    # print("Hourly:", n, model.E[n]())
                    # Psb_list.append(model.Psb[n]())
                    # Psg_list.append(model.Psg[n]())
                    Pbg_list.append(model.Pbg[n]())
                    Pgb_list.append(model.Pgb[n]())
                # return model.E[n]

            #######
            ###BELOW UNTESTED###
            #######
            dispatch = bat_dispatch(n, lmps, E_last,Emin, Emax, Ecost, Bcost, pmax, eta_c, eta_d, Pi)
            E_last = E[-1]
            # print(E)
            # print("ELast:", E_last)
            new_list=[]
            unit_name_cleaned = unit_name.replace(':', '').replace('-', '')
            new_list.extend([unit_name_cleaned for i in range(n)])

            # Save results to dataframe
            results_df = pd.DataFrame(
                {'date':date_list,
                'time_interval': range(1,n+1),
                'unit_name':new_list,
                'soc': E,
                # 'solar': list(Ps.values()),
                # 'StoB': Psb_list,
                # 'StoG': Psg_list,
                'BtoG': Pbg_list,
                'GtoB': Pgb_list,
                'prices': list(lmps.values())
                })

            results_df['soc'] = results_df['soc'].round(3)
            # results_df['solar'] = results_df['solar'].round(3)
            # results_df['StoB'] = results_df['StoB'].round(3)
            # results_df['StoG'] = results_df['StoG'].round(3)
            results_df['BtoG'] = results_df['BtoG'].round(3)
            results_df['GtoB'] = results_df['GtoB'].round(3)
            results_df['prices'] = results_df['prices'].round(2)

            # Save dispatch results to individual csv
            #res_file_name = unit_name_cleaned + '_' + sel_day + '_results.csv'
            #results_df.to_csv(res_file_name, index=False)

            # append results to CSV file
            results_df.to_csv(all_results_file, mode='a', index=False, header=False)

            # fig = plt.figure(figsize=(20, 10), dpi=300)

            # Add plots
            # ax1 = fig.add_subplot(4, 1, 1)
            # ax1.plot(results_df['time_interval'], results_df['soc'])
            # ax1.set_xlabel('Time', fontsize=12)
            # ax1.set_ylabel('SOC', fontsize=12)
            # ax2 = fig.add_subplot(6, 1, 2)
            # ax2.plot(results_df['time_interval'], results_df['solar'], 'y')
            # ax2.set_xlabel('Time', fontsize=12)
            # ax2.set_ylabel('solar', fontsize=12)
            # ax3 = fig.add_subplot(6, 1, 3)
            # ax3.plot(results_df['time_interval'], results_df['StoB'],'k')
            # ax3.set_xlabel('Time', fontsize=12)
            # ax3.set_ylabel('StoB', fontsize=12)
            # ax4 = fig.add_subplot(4, 1, 2)
            # ax4.plot(results_df['time_interval'], results_df['GtoB'], 'r')
            # ax4.set_xlabel('Time', fontsize=12)
            # ax4.set_ylabel('GtoB', fontsize=12)
            # ax5 = fig.add_subplot(4, 1, 3)
            # ax5.plot(results_df['time_interval'], results_df['BtoG'], 'g')
            # ax5.set_xlabel('Time', fontsize=12)
            # ax5.set_ylabel('BtoG', fontsize=12)
            # ax6 = fig.add_subplot(4, 1, 4)
            # ax6.plot(results_df['time_interval'], results_df['prices'], 'm')
            # ax6.set_xlabel('Time', fontsize=12)
            # ax6.set_ylabel('prices', fontsize=12)
            # fig_fname = "freeSOC_"+unit_name_cleaned + '_' + sel_day + '.png'
            # fig.savefig("2022_batt_dispatch_results/"+fig_fname)



# %%
