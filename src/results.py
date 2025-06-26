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

def read_flexibility_csvs(base_filename: str, src_path: str = "src/") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the two CSV files generated by evaluate_flexibility function and return DataFrames.
    
    Parameters:
    -----------
    base_filename : str
        Base name of the CSV files (without extension)
        e.g., "datos_res" will read "datos_res.csv" and "datos_res.csv_acumulated.csv"
    src_path : str
        Path to the source directory containing the CSV files
        
    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        - results_df: DataFrame with section design information for each simulation and period
        - accumulated_sn_df: DataFrame with accumulated SN values for each simulation over time
    """
    import os
    
    # Construct file paths
    results_file = os.path.join(src_path, f"{base_filename}.csv")
    accumulated_file = os.path.join(src_path, f"{base_filename}.csv_acumulated.csv")
    
    # Check if files exist
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    if not os.path.exists(accumulated_file):
        raise FileNotFoundError(f"Accumulated SN file not found: {accumulated_file}")
    
    # Read the results DataFrame
    results_df = pd.read_csv(results_file)
    
    # Read the accumulated SN data
    accumulated_sn_df = pd.read_csv(accumulated_file, header=0)
    
    print(f"Successfully loaded flexibility analysis results:")
    print(f"- Results DataFrame: {results_df.shape[0]} entries with {results_df.shape[1]} columns")
    print(f"- Accumulated SN DataFrame: {accumulated_sn_df.shape[0]} simulations x {accumulated_sn_df.shape[1]} time periods")
    
    return results_df, accumulated_sn_df

def plot_flexible_design_sn(results_df: pd.DataFrame, accumulated_sn_df: pd.DataFrame, 
                            simulation_id: int = None, show_envelope: bool = False, 
                            show_all_sims: bool = False, max_sims_display: int = 10) -> tuple:
    """
    Plot SN projected vs SN designed for flexible pavement design, showing reconstruction events.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with section design information from evaluate_flexibility
    accumulated_sn_df : pd.DataFrame 
        DataFrame with accumulated SN values over time (simulations x time periods)
    simulation_id : int, optional
        Specific simulation to highlight. If None, shows first simulation
    show_envelope : bool
        Whether to show the envelope (min/max) of all simulations
    show_all_sims : bool
        Whether to show all individual simulations (faded)
    max_sims_display : int
        Maximum number of individual simulations to display when show_all_sims=True
        
    Returns:
    --------
    tuple
        (figure, axes) containing the plot
    """
    
    # Determine which simulation to focus on
    if simulation_id is None:
        simulation_id = results_df['simulation'].min()
    
    # Get the total time periods
    n_periods = accumulated_sn_df.shape[1]
    time_periods = np.arange(n_periods)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot envelope if requested
    if show_envelope:
        sn_max = accumulated_sn_df.max(axis=0)
        sn_min = accumulated_sn_df.min(axis=0)
        ax.fill_between(time_periods, sn_min, sn_max, 
                       alpha=0.2, color='lightgray', label='SN Requirement Envelope (All Sims)')
    
    # Plot individual simulations if requested
    if show_all_sims:
        n_sims_to_show = min(max_sims_display, len(accumulated_sn_df))
        sim_indices = np.linspace(0, len(accumulated_sn_df)-1, n_sims_to_show, dtype=int)
        
        for i, sim_idx in enumerate(sim_indices):
            if sim_idx != simulation_id:  # Don't duplicate the main simulation
                sn_projected = accumulated_sn_df.iloc[sim_idx].values
                ax.plot(time_periods, sn_projected, 'gray', alpha=0.3, linewidth=0.5)
        
        # Add legend entry for individual sims
        ax.plot([], [], 'gray', alpha=0.3, linewidth=0.5, label=f'Individual Simulations (sample of {n_sims_to_show})')
    
    # Plot the main simulation's projected SN
    main_sim_projected = accumulated_sn_df.iloc[simulation_id].values
    ax.plot(time_periods, main_sim_projected, 'r-', linewidth=2, 
           label=f'SN Projected (Sim {simulation_id})')
    
    # Get reconstruction events for the main simulation
    sim_events = results_df[results_df['simulation'] == simulation_id].copy()
    sim_events = sim_events.sort_values('period')
    
    # Build the SN capacity timeline for the main simulation
    sn_capacity_timeline = np.zeros(n_periods)
    capacity_periods = []
    capacity_values = []
    
    if len(sim_events) > 0:
        # Process each reconstruction event
        for idx, event in sim_events.iterrows():
            period = int(event['period'])
            total_sn = event['total_sn']
            startover = event['startover']
            
            # Store the capacity change point
            capacity_periods.append(period)
            capacity_values.append(total_sn)
            
            # Fill capacity from this period until next event (or end)
            next_period = n_periods
            next_events = sim_events[sim_events['period'] > period]
            if len(next_events) > 0:
                next_period = int(next_events.iloc[0]['period'])
            
            # Calculate capacity degradation if applicable
            if startover > 0:  # Partial reconstruction - base deteriorates
                # Linear degradation of base between reconstructions
                base_initial = startover
                base_final = max(0, base_initial * 0.8)  # Assume 20% degradation
                top_sn = total_sn - base_initial
                
                for t in range(period, min(next_period, n_periods)):
                    if next_period > period:
                        degradation_factor = (t - period) / (next_period - period)
                        current_base = base_initial - (base_initial - base_final) * degradation_factor
                    else:
                        current_base = base_initial
                    sn_capacity_timeline[t] = top_sn + current_base
            else:  # Complete reconstruction - constant capacity
                sn_capacity_timeline[period:min(next_period, n_periods)] = total_sn
    
    # Plot the SN capacity as a step function
    if len(capacity_periods) > 0:
        # Create step plot data
        step_periods = [0]  # Start from beginning
        step_values = [0]   # Start with zero capacity
        
        for i, (period, value) in enumerate(zip(capacity_periods, capacity_values)):
            # Add point just before the jump
            step_periods.append(period)
            step_values.append(step_values[-1])
            
            # Add the jump
            step_periods.append(period)
            step_values.append(value)
            
            # Add degradation if applicable
            if i < len(capacity_periods) - 1:
                next_period = capacity_periods[i + 1]
                current_event = sim_events.iloc[i]
                
                if current_event['startover'] > 0:
                    # Add degradation points
                    degradation_points = np.linspace(period + 1, next_period - 1, 
                                                   max(1, (next_period - period) // 12))
                    for deg_period in degradation_points:
                        step_periods.append(int(deg_period))
                        step_values.append(sn_capacity_timeline[int(deg_period)])
                else:
                    # Constant capacity
                    step_periods.append(next_period)
                    step_values.append(value)
            else:
                # Last event - extend to end
                step_periods.append(n_periods - 1)
                step_values.append(sn_capacity_timeline[min(period, n_periods - 1)])
        
        ax.plot(step_periods, step_values, 'b-', linewidth=2, 
               label=f'SN Capacity (Sim {simulation_id})')
        
        # Mark reconstruction events
        for period, value in zip(capacity_periods, capacity_values):
            event_info = sim_events[sim_events['period'] == period].iloc[0]
            marker_color = 'red' if event_info['startover'] == 0 else 'orange'
            marker_style = '^' if event_info['startover'] == 0 else 'o'
            label_suffix = 'Complete Rebuild' if event_info['startover'] == 0 else 'Top Layer Rebuild'
            
            ax.plot(period, value, marker=marker_style, color=marker_color, 
                   markersize=8, markeredgecolor='black', markeredgewidth=1)
            
            # Add annotation for reconstruction events
            ax.annotate(f'{label_suffix}\nSN={value:.2f}', 
                       xy=(period, value), xytext=(10, 10), 
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Add shaded areas where capacity is exceeded
    exceeded_mask = main_sim_projected > sn_capacity_timeline
    if np.any(exceeded_mask):
        ax.fill_between(time_periods, sn_capacity_timeline, main_sim_projected,
                       where=exceeded_mask, color='red', alpha=0.3, 
                       label='SN Capacity Exceeded')
    
    # Formatting
    ax.set_xlabel('Time Period (months)')
    ax.set_ylabel('Structural Number (SN)')
    ax.set_title(f'Flexible Pavement Design: SN Projected vs SN Capacity\n'
                f'Simulation {simulation_id} with Reconstruction Events')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    # Add secondary x-axis in years
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    year_ticks = np.arange(0, n_periods, 12)
    ax2.set_xticks(year_ticks)
    ax2.set_xticklabels([f'{t//12}' for t in year_ticks])
    ax2.set_xlabel('Time (years)')
    
    plt.tight_layout()
    return fig, ax


def plot_multiple_simulations_comparison(results_df: pd.DataFrame, accumulated_sn_df: pd.DataFrame,
                                       simulation_ids: list = None, max_sims: int = 5) -> tuple:
    """
    Plot multiple simulations in separate subplots for detailed comparison.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with section design information from evaluate_flexibility
    accumulated_sn_df : pd.DataFrame 
        DataFrame with accumulated SN values over time
    simulation_ids : list, optional
        List of simulation IDs to plot. If None, selects first max_sims simulations
    max_sims : int
        Maximum number of simulations to plot
        
    Returns:
    --------
    tuple
        (figure, axes) containing the plots
    """
    
    if simulation_ids is None:
        available_sims = results_df['simulation'].unique()
        simulation_ids = available_sims[:min(max_sims, len(available_sims))]
    
    n_sims = len(simulation_ids)
    n_cols = min(2, n_sims)
    n_rows = (n_sims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
    if n_sims == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, sim_id in enumerate(simulation_ids):
        ax = axes[i] if n_sims > 1 else axes[0]
        
        # Get data for this simulation
        n_periods = accumulated_sn_df.shape[1]
        time_periods = np.arange(n_periods)
        sim_projected = accumulated_sn_df.iloc[sim_id].values
        
        # Plot projected SN
        ax.plot(time_periods, sim_projected, 'r-', linewidth=2, label='SN Projected')
        
        # Get and plot capacity
        sim_events = results_df[results_df['simulation'] == sim_id].copy()
        sim_events = sim_events.sort_values('period')
        
        if len(sim_events) > 0:
            # Build capacity timeline (simplified version)
            capacity_periods = [0]
            capacity_values = [0]
            
            for _, event in sim_events.iterrows():
                period = int(event['period'])
                total_sn = event['total_sn']
                
                # Add step change
                capacity_periods.extend([period, period])
                capacity_values.extend([capacity_values[-1], total_sn])
            
            # Extend to end
            capacity_periods.append(n_periods - 1)
            capacity_values.append(capacity_values[-1])
            
            ax.plot(capacity_periods, capacity_values, 'b-', linewidth=2, label='SN Capacity')
            
            # Mark events
            for _, event in sim_events.iterrows():
                period = int(event['period'])
                total_sn = event['total_sn']
                marker = '^' if event['startover'] == 0 else 'o'
                color = 'red' if event['startover'] == 0 else 'orange'
                ax.plot(period, total_sn, marker=marker, color=color, markersize=6)
        
        ax.set_title(f'Simulation {sim_id}')
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('SN')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide extra subplots
    for i in range(n_sims, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig, axes

def plot_sn_progression(datos_res, datos_res_accumulated, simulation_id, return_fig=False):
    """
    Plots the accumulated SN and design capacity progression for a specific simulation.
    
    Parameters:
        datos_res (pd.DataFrame): DataFrame from 'datos_res.csv' with redesign events.
        datos_res_accumulated (pd.DataFrame): DataFrame with accumulated SN over time.
                                            Format: simulations as rows, time periods as columns
        simulation_id (int): ID of the simulation to plot.
        return_fig (bool): If True, returns the figure instead of showing it.
    
    Returns:
        matplotlib.figure.Figure: The created figure (if return_fig=True).
    """
    # Filter redesign events for the specified simulation
    redesign_events = datos_res[datos_res['simulation'] == simulation_id].copy()
    
    # Sort redesign events by period first
    redesign_events = redesign_events.sort_values('period').reset_index(drop=True)
    
    # Use top_sn and startover columns directly from the CSV data
    # design_capacity = top_sn (capacity of top layers)
    # total_capacity = startover + top_sn (total structural capacity)
    redesign_events['design_capacity'] = redesign_events['top_sn']
    redesign_events['total_capacity'] = redesign_events['startover'] + redesign_events['top_sn']
    
    # Get accumulated SN data for the specific simulation
    # accumulated_sn_df structure: rows=simulations, columns=time_periods
    if simulation_id >= len(datos_res_accumulated):
        raise ValueError(f"Simulation {simulation_id} not found. Available simulations: 0 to {len(datos_res_accumulated)-1}")
    
    accumulated_sn_values = datos_res_accumulated.iloc[simulation_id].values
    time_periods = np.arange(len(accumulated_sn_values))
    
    # Create figure and plot accumulated SN
    fig = plt.figure(figsize=(12, 6))
    plt.plot(time_periods, accumulated_sn_values, 
             label='Accumulated SN Demand', color='blue', linewidth=2)
    
    # Plot design capacity as a step function
    if not redesign_events.empty:
        periods = redesign_events['period'].tolist()
        design_capacities = redesign_events['design_capacity'].tolist()  # top_sn values
        total_capacities = redesign_events['total_capacity'].tolist()     # startover + top_sn values
        
        # Create step function for design capacity (top_sn)
        capacity_timeline = np.zeros(len(time_periods))
        
        for i, (period, capacity) in enumerate(zip(periods, design_capacities)):
            # Set capacity from this period to the next (or end)
            start_period = int(period)
            end_period = len(time_periods)
            
            if i < len(periods) - 1:
                end_period = int(periods[i + 1])
            
            if start_period < len(capacity_timeline):
                capacity_timeline[start_period:end_period] = capacity
        
        plt.plot(time_periods, capacity_timeline, 
                 label='Design Capacity (Top Layers)', color='red', linewidth=2, linestyle='--')
        
        # Create step function for total capacity (startover + top_sn)
        total_capacity_timeline = np.zeros(len(time_periods))
        
        for i, (period, total_capacity) in enumerate(zip(periods, total_capacities)):
            # Set total capacity from this period to the next (or end)
            start_period = int(period)
            end_period = len(time_periods)
            
            if i < len(periods) - 1:
                end_period = int(periods[i + 1])
            
            if start_period < len(total_capacity_timeline):
                total_capacity_timeline[start_period:end_period] = total_capacity
        
        plt.plot(time_periods, total_capacity_timeline, 
                 label='Total SN Capacity (Base + Top)', color='purple', linewidth=2, linestyle='-')
        
        # Add vertical lines at redesign events
        for i, period in enumerate(periods):
            label = 'Redesign Event' if i == 0 else None
            plt.axvline(x=period, color='green', linestyle='--', alpha=0.7, label=label)
            
        # Add shaded area where demand exceeds total capacity
        exceed_mask = accumulated_sn_values > total_capacity_timeline
        if np.any(exceed_mask):
            plt.fill_between(time_periods, total_capacity_timeline, accumulated_sn_values,
                           where=exceed_mask, color='red', alpha=0.3, 
                           label='SN Total Capacity Exceeded')
    
    # Add labels and title
    plt.xlabel('Time Period (months)')
    plt.ylabel('Structural Number (SN)')
    plt.title(f'SN Progression and Redesign Events (Simulation {simulation_id})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add secondary x-axis in years
    ax2 = plt.gca().twiny()
    ax2.set_xlim(plt.gca().get_xlim())
    year_ticks = np.arange(0, len(time_periods), 12)
    ax2.set_xticks(year_ticks)
    ax2.set_xticklabels([f'{t//12}' for t in year_ticks])
    ax2.set_xlabel('Time (years)')
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()

def analyze_traditional_vs_simulated(params: dict, accumulated_traffic_data: np.ndarray, 
                                   evaluation_periods: list = None, return_fig: bool = False):
    """
    Analyze how well the traditional design approach predicts traffic growth compared to simulated data.
    Shows frequency of underestimation vs overestimation.
    
    Parameters:
    -----------
    params : dict
        Dictionary containing design parameters (TPD, vc, cd, n, mu_function, etc.)
    accumulated_traffic_data : np.ndarray
        2D array with accumulated traffic data (simulations x time_periods)
    evaluation_periods : list, optional
        List of time periods (in months) to evaluate. If None, uses [60, 120, 180, 240, 300, 360]
    return_fig : bool
        If True, returns the figure instead of showing it
        
    Returns:
    --------
    dict
        Dictionary containing analysis results and optionally the figure
    """
    
    if evaluation_periods is None:
        evaluation_periods = [60, 120, 180, 240, 300, min(360, params['n']-1)]
    
    # Remove periods that exceed the simulation length
    evaluation_periods = [p for p in evaluation_periods if p < params['n']]
    
    # Calculate traditional design prediction using linear growth rate
    initial_monthly_trips = params['TPD'] * 365 / 12 * params['vc'] * params['cd']
    
    # Generate traditional prediction using mean growth function
    traditional_monthly_traffic = np.zeros(params['n'])
    for month in range(params['n']):
        growth_rate = params['mu_function'](month)
        traditional_monthly_traffic[month] = initial_monthly_trips * (1 + growth_rate)
    
    # Calculate traditional accumulated traffic
    traditional_accumulated = np.cumsum(traditional_monthly_traffic)
    
    # Alternatively, use the W18_i_regression approach for comparison
    # Convert to annual and calculate regression-based growth rate
    annual_periods = params['n'] // 12
    traditional_annual = np.zeros(annual_periods)
    for year in range(annual_periods):
        year_traffic = 0
        for month in range(12):
            month_idx = year * 12 + month
            if month_idx < len(traditional_monthly_traffic):
                year_traffic += traditional_monthly_traffic[month_idx]
        traditional_annual[year] = year_traffic
    
    # Calculate average growth rate from traditional approach
    if len(traditional_annual) > 1:
        traditional_growth_rate = W18_i_regression(traditional_annual)
    else:
        traditional_growth_rate = 0.047  # Default
    
    # Create W18-based prediction
    w18_prediction = np.zeros(params['n'])
    for month in range(params['n']):
        year_fraction = month / 12.0
        annual_traffic = pred_W18(params['TPD'], params['vc'], params['cd'], traditional_growth_rate, year_fraction)
        w18_prediction[month] = annual_traffic
    
    # Analysis results storage
    results = {
        'evaluation_periods': evaluation_periods,
        'underestimation_frequency': {},
        'overestimation_frequency': {},
        'mean_error': {},
        'rmse': {},
        'prediction_errors': {}
    }
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Sample trajectories comparison
    n_samples = min(10, accumulated_traffic_data.shape[0])
    sample_indices = np.random.choice(accumulated_traffic_data.shape[0], n_samples, replace=False)
    
    time_periods = np.arange(params['n'])
    
    for i, idx in enumerate(sample_indices):
        alpha = 0.3 if i > 0 else 0.8
        linewidth = 0.8 if i > 0 else 2
        label = 'Simulated Traffic' if i == 0 else None
        ax1.plot(time_periods, accumulated_traffic_data[idx], 'b-', alpha=alpha, 
                linewidth=linewidth, label=label)
    
    ax1.plot(time_periods, traditional_accumulated, 'r-', linewidth=3, 
            label='Traditional Prediction (μ function)')
    ax1.plot(time_periods, w18_prediction, 'g--', linewidth=2, 
            label='W18 Regression Prediction')
    
    ax1.set_xlabel('Time (months)')
    ax1.set_ylabel('Accumulated Traffic')
    ax1.set_title('Traditional vs Simulated Traffic Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution at specific periods
    colors = plt.cm.viridis(np.linspace(0, 1, len(evaluation_periods)))
    
    for i, period in enumerate(evaluation_periods):
        if period >= accumulated_traffic_data.shape[1]:
            continue
            
        simulated_values = accumulated_traffic_data[:, period]
        traditional_value = traditional_accumulated[period]
        w18_value = w18_prediction[period]
        
        # Calculate errors
        traditional_errors = (simulated_values - traditional_value) / traditional_value * 100
        w18_errors = (simulated_values - w18_value) / w18_value * 100
        
        # Store results
        results['underestimation_frequency'][period] = np.sum(traditional_errors > 0) / len(traditional_errors) * 100
        results['overestimation_frequency'][period] = np.sum(traditional_errors < 0) / len(traditional_errors) * 100
        results['mean_error'][period] = np.mean(traditional_errors)
        results['rmse'][period] = np.sqrt(np.mean(traditional_errors**2))
        results['prediction_errors'][period] = traditional_errors
        
        # Plot error distribution
        ax2.hist(traditional_errors, bins=30, alpha=0.6, color=colors[i], 
                label=f'Month {period}', density=True)
    
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Perfect Prediction')
    ax2.set_xlabel('Prediction Error (%)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Prediction Errors\n(Positive = Underestimation)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Frequency of under/over estimation over time
    periods_list = list(results['underestimation_frequency'].keys())
    under_freq = [results['underestimation_frequency'][p] for p in periods_list]
    over_freq = [results['overestimation_frequency'][p] for p in periods_list]
    
    x_pos = np.arange(len(periods_list))
    width = 0.35
    
    ax3.bar(x_pos - width/2, under_freq, width, label='Underestimation', color='red', alpha=0.7)
    ax3.bar(x_pos + width/2, over_freq, width, label='Overestimation', color='blue', alpha=0.7)
    
    ax3.set_xlabel('Evaluation Period (months)')
    ax3.set_ylabel('Frequency (%)')
    ax3.set_title('Frequency of Under vs Over Estimation')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([str(p) for p in periods_list])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: RMSE and Mean Error over time
    rmse_values = [results['rmse'][p] for p in periods_list]
    mean_errors = [results['mean_error'][p] for p in periods_list]
    
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(periods_list, rmse_values, 'bo-', label='RMSE', linewidth=2, markersize=6)
    line2 = ax4_twin.plot(periods_list, mean_errors, 'ro-', label='Mean Error', linewidth=2, markersize=6)
    
    ax4.set_xlabel('Evaluation Period (months)')
    ax4.set_ylabel('RMSE (%)', color='blue')
    ax4_twin.set_ylabel('Mean Error (%)', color='red')
    ax4.set_title('Prediction Accuracy Metrics Over Time')
    ax4.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Print summary statistics
    print("=" * 60)
    print("TRADITIONAL DESIGN vs SIMULATED TRAFFIC ANALYSIS")
    print("=" * 60)
    
    for period in evaluation_periods:
        if period in results['underestimation_frequency']:
            print(f"\nPeriod {period} months ({period//12:.1f} years):")
            print(f"  Underestimation frequency: {results['underestimation_frequency'][period]:.1f}%")
            print(f"  Overestimation frequency:  {results['overestimation_frequency'][period]:.1f}%")
            print(f"  Mean prediction error:     {results['mean_error'][period]:+.1f}%")
            print(f"  RMSE:                      {results['rmse'][period]:.1f}%")
    
    # Overall assessment
    overall_under = np.mean(list(results['underestimation_frequency'].values()))
    overall_over = np.mean(list(results['overestimation_frequency'].values()))
    overall_rmse = np.mean(list(results['rmse'].values()))
    
    print(f"\nOVERALL ASSESSMENT:")
    print(f"  Average underestimation frequency: {overall_under:.1f}%")
    print(f"  Average overestimation frequency:  {overall_over:.1f}%")
    print(f"  Average RMSE:                      {overall_rmse:.1f}%")
    
    if overall_under > overall_over:
        bias = "CONSERVATIVE (tends to underestimate)"
    elif overall_over > overall_under:
        bias = "AGGRESSIVE (tends to overestimate)"
    else:
        bias = "BALANCED"
    
    print(f"  Design bias:                       {bias}")
    print("=" * 60)
    
    if return_fig:
        results['figure'] = fig
        return results
    else:
        plt.show()
        return results