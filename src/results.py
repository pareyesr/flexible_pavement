#TODO Responder la pregunta de fondo: ¿Cual es la mejor politica de diseño? 
#TODO grafica de demanda vs capacidad. SN vs trafico proyc vs simulado.
import numpy as np
import pandas as pd
from Logica import *
import matplotlib.pyplot as plt

def calculate_sn_projected(acumulated:np.array,params:dict):
    """
    transit: array with acumulated traffic
    params: dictionary with all the parameters
    return: array with projected SN
    """
    sn_projected = np.zeros(len(acumulated))
    for i in range(len(acumulated)):
        try:
            sn_projected[i] = solve_sn(params["Reliavility"], params["Standard_Deviation"], params["Delta_PSI"], params["Mr"],acumulated[i])
        except:
            sn_projected[i] = sn_projected[i-1]
    return sn_projected

def plot_simulated_function(params:dict):
    """
    Plot the mu_function with sigma_function bands around it.
    
    Shows:
    - Mean growth function (mu_function)
    - Standard deviation bands (sigma_function) around the mean
    
    Args:
        params: Dictionary containing simulation parameters
    """
    # Create time periods array
    time_periods = np.arange(params["n"])
    
    # Calculate mu_function values for each time period
    mu_values = np.array([params["mu_function"](t) for t in time_periods])
    
    # Calculate sigma_function values for each time period
    sigma_values = np.array([params["sigma_function"](t) for t in time_periods])
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Plot the mean function
    plt.plot(time_periods, mu_values, 'r-', linewidth=2, label='Mean Growth Rate (μ)')
    
    # Plot the standard deviation bands
    plt.fill_between(time_periods, 
                     mu_values - sigma_values, 
                     mu_values + sigma_values, 
                     alpha=0.3, color='lightblue', label='±1σ Range')
    
    # Plot the 2-sigma bands
    plt.fill_between(time_periods, 
                     mu_values - 2*sigma_values, 
                     mu_values + 2*sigma_values, 
                     alpha=0.2, color='lightblue', label='±2σ Range')
    
    # Add labels and legend
    plt.xlabel('Time Period (months)')
    plt.ylabel('Growth Rate')
    plt.title('Growth Rate Function with Standard Deviation Bands')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def plot_simulated_transit(params:dict):
    """
    Plot the behavior of the monthly traffic growth using the formula:
    res[sim, month] = initial_monthly_trips * (1 + grow_rates[sim, month])
    
    Shows two subplots:
    1. Monthly traffic with linear scale
    2. Cumulative traffic with log scale
    
    Each subplot shows:
    - Individual simulation runs of traffic
    - Mean of all simulations
    - Range (min to max) of all simulations
    
    Args:
        params: Dictionary containing simulation parameters
    """
    # Extract parameters
    TPD = params["TPD"]
    vc = params["vc"]
    cd = params["cd"]
    size = params["size"]
    n = params["n"]
    seedint = params["seedint"]
    f_mean = params["mu_function"]
    f_std = params["sigma_function"]
    
    # Set up random number generator
    rng = np.random.default_rng(seed=seedint)
    
    # Generate growth rates
    grow_rates = np.zeros((size, n))
    for i in range(size):
        for j in range(n):
            grow_rates[i,j] = rng.normal(loc=f_mean(j), scale=f_std(j))
    
    # Calculate initial monthly trips
    initial_monthly_trips = TPD * 365 / 12 * vc * cd
    
    # Initialize arrays for monthly traffic and cumulative traffic
    res = np.zeros((size, n))
    cum_res = np.zeros((size, n))
    
    # Apply the growth formula and calculate cumulative traffic
    for sim in range(size):
        res[sim, 0] = initial_monthly_trips
        cum_res[sim, 0] = initial_monthly_trips
        for month in range(1, n):
            # Apply compounded growth
            res[sim, month] = initial_monthly_trips * (1 + grow_rates[sim, month])
            # Calculate cumulative traffic
            cum_res[sim, month] = cum_res[sim, month-1] + res[sim, month]
    
    # Calculate statistics for monthly traffic
    time_periods = np.arange(n)
    mean_traffic = np.mean(res, axis=0)
    min_traffic = np.min(res, axis=0)
    max_traffic = np.max(res, axis=0)
    
    # Calculate statistics for cumulative traffic
    mean_cum = np.mean(cum_res, axis=0)
    min_cum = np.min(cum_res, axis=0)
    max_cum = np.max(cum_res, axis=0)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot monthly traffic (linear scale)
    ax1.fill_between(time_periods, min_traffic, max_traffic, 
                    alpha=0.2, color='lightgray', label='Min-Max Range')
    
    # Plot a sample of individual simulations
    num_to_show = min(10, size)
    random_indices = np.random.choice(size, num_to_show, replace=False)
    
    for i, idx in enumerate(random_indices):
        ax1.plot(time_periods, res[idx], alpha=0.7, linewidth=0.8, 
                label=f'Simulation {idx+1}' if i < 5 else "_nolegend_")
    
    ax1.plot(time_periods, mean_traffic, 'k-', linewidth=2, label='Mean')
    ax1.set_xlabel('Time Period (months)')
    ax1.set_ylabel('Monthly Traffic')
    ax1.set_title('Monthly Traffic Growth Simulation (Linear Scale)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Plot cumulative traffic (log scale)
    # Ensure no zero or negative values for log scale
    min_cum_plot = np.maximum(min_cum, 1e-10)  # Set minimum value to avoid log(0)
    ax2.fill_between(time_periods, min_cum_plot, max_cum, 
                    alpha=0.2, color='lightblue', label='Min-Max Range')
    
    for i, idx in enumerate(random_indices):
        ax2.plot(time_periods, cum_res[idx], alpha=0.7, linewidth=0.8, 
                label=f'Simulation {idx+1}' if i < 5 else "_nolegend_")
    
    ax2.plot(time_periods, mean_cum, 'k-', linewidth=2, label='Mean')
    ax2.set_xlabel('Time Period (months)')
    ax2.set_ylabel('Cumulative Traffic')
    ax2.set_title('Cumulative Traffic (Log Scale)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_yscale('log')
    # Set y-axis limits to ensure visibility of the range
    ax2.set_ylim(min_cum_plot.min(), max_cum.max() * 1.1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    plt.show()