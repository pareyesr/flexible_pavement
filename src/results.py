#TODO Responder la pregunta de fondo: ¿Cual es la mejor politica de diseño? 
#TODO grafica de demanda vs capacidad. SN vs trafico proyc vs simulado.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import functions based on how this module is being used
try:
    # Try relative imports first (when imported as a module)
    from .Logica import *
except ImportError:
    # Fall back to absolute imports (when run directly)
    from Logica import *

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
    min_cum_safe = np.maximum(min_cum, 1)  # Use 1 instead of 1e-10 for better visibility
    max_cum_safe = np.maximum(max_cum, 1)  # Ensure max is also positive
    
    # Plot the fill_between first (so it appears behind other elements)
    ax2.fill_between(time_periods, min_cum_safe, max_cum_safe, 
                    alpha=0.3, color='lightblue', label='Min-Max Range')
    
    # Plot individual simulations
    for i, idx in enumerate(random_indices):
        cum_safe = np.maximum(cum_res[idx], 1)  # Ensure no zero values
        ax2.plot(time_periods, cum_safe, alpha=0.6, linewidth=0.8, 
                label=f'Simulation {idx+1}' if i < 5 else "_nolegend_")
    
    # Plot mean last (so it appears on top)
    mean_cum_safe = np.maximum(mean_cum, 1)
    ax2.plot(time_periods, mean_cum_safe, 'r-', linewidth=3, label='Mean')
    
    ax2.set_xlabel('Time Period (months)')
    ax2.set_ylabel('Cumulative Traffic')
    ax2.set_title('Cumulative Traffic (Log Scale)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_yscale('log')
    
    # Set y-axis limits with proper margin for log scale
    y_min = max(min_cum_safe.min() * 0.8, 1)
    y_max = max_cum_safe.max() * 1.2
    ax2.set_ylim(y_min, y_max)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    plt.show()

def plot_traditional_design(params:dict, DF):
    """
    Plot the results of a traditional pavement design, showing traffic progression and SN capacity over time.
    
    Parameters:
    -----------
    params : dict
        Dictionary containing design parameters including:
        - TPD: Traffic per day
        - vc: Directional distribution (usually 0.5)
        - cd: Design lane factor (usually 1.0)
        - n: Design period in months
        - Reliavility: Reliability level
        - Standard_Deviation: Standard deviation
        - Delta_PSI: Allowable serviceability loss
        - Mr: Resilient modulus
        - grade: Grade adjustment
        - emb: Embankment cost
        - excv: Excavation cost
        - mu_function: Traffic growth function
    DF : pd.DataFrame
        DataFrame containing material properties
    
    Returns:
    --------
    tuple
        (figure, axes) containing the plot
    """
    # Get design results
    dis_sect, sn_design, m = traditional_design(params, DF)
    
    # Use same logic as traditional_design function from Logica.py
    initial_monthly_trips = params['TPD'] * 365 / 12 * params['vc'] * params['cd']
    
    # Generate monthly traffic using growth rates (same as make_simulated_transit)
    monthly_traffic = np.zeros(params['n'])
    for month in range(params['n']):
        growth_rate = params['mu_function'](month)
        monthly_traffic[month] = initial_monthly_trips * (1 + growth_rate)
    
    # Convert monthly traffic to annual traffic
    mean_traffic = np.zeros(params['n']//12)
    for i in range(params['n']//12):
        annual_traffic = 0
        for j in range(12):
            month_idx = i * 12 + j
            if month_idx < len(monthly_traffic):
                annual_traffic += monthly_traffic[month_idx]
        mean_traffic[i] = annual_traffic
    
    # Calculate accumulated traffic from annual mean_traffic
    acumulated_traffic = np.cumsum(mean_traffic)
    
    # Print design information
    print(dis_sect.info())
    print(f"Design SN: {sn_design:.2f}")
    
    # Calculate projected SN over time
    sn_projected = calculate_sn_projected(acumulated_traffic, params)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot 1: Monthly Traffic progression
    months = np.arange(len(monthly_traffic))
    ax1.plot(months, monthly_traffic, 'b-', label='Monthly Traffic')
    ax1.set_xlabel('Months')
    ax1.set_ylabel('Monthly Traffic (ESAL)')
    ax1.set_title('Monthly Traffic Progression Over Time')
    ax1.grid(True)
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Plot 2: Annual Traffic and Cumulative Traffic
    years = np.arange(len(mean_traffic))
    ax2_twin = ax2.twinx()
    
    line1 = ax2.bar(years, mean_traffic, alpha=0.6, color='lightblue', label='Annual Traffic')
    line2 = ax2_twin.plot(years, acumulated_traffic, 'r-', linewidth=2, label='Cumulative Traffic')
    
    ax2.set_xlabel('Years')
    ax2.set_ylabel('Annual Traffic (ESAL)', color='blue')
    ax2_twin.set_ylabel('Cumulative Traffic (ESAL)', color='red')
    ax2.set_title('Annual and Cumulative Traffic Progression')
    ax2.grid(True)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax2_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Plot 3: SN capacity vs projected SN
    ax3.plot(years, [sn_design] * len(years), 'r--', linewidth=2, label='Design SN')
    ax3.plot(years, sn_projected, 'g-', linewidth=2, label='Projected SN')
    ax3.set_xlabel('Years')
    ax3.set_ylabel('Structural Number (SN)')
    ax3.set_title('SN Capacity vs Projected SN Over Time')
    ax3.grid(True)
    ax3.legend()
    
    # Add a shaded area where projected SN exceeds design SN
    if np.any(sn_projected > sn_design):
        exceed_mask = sn_projected > sn_design
        ax3.fill_between(years, sn_design, sn_projected, 
                        where=exceed_mask, color='red', alpha=0.3,
                        label='SN Exceedance')
        ax3.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, (ax1, ax2, ax3)
