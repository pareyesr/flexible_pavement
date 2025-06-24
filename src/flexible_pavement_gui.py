import tkinter as tk
from tkinter import ttk
import os
import numpy as np
import pandas as pd
from tkinter import messagebox
from tkinter import filedialog
from scipy.optimize import curve_fit
import tkinter.font as tkfont

# Import matplotlib components with error handling
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Set backend before importing pyplot
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
    from matplotlib.figure import Figure
    # Only import key_press_handler if needed
    try:
        from matplotlib.backend_bases import key_press_handler
    except ImportError:
        key_press_handler = None
except ImportError as e:
    print(f"Matplotlib import error: {e}")
    # Create fallback classes if matplotlib fails
    class Figure:
        def __init__(self, *args, **kwargs):
            pass
    class FigureCanvasTkAgg:
        def __init__(self, *args, **kwargs):
            pass
    class NavigationToolbar2Tk:
        def __init__(self, *args, **kwargs):
            pass
    plt = None

def show_copyable_message(title, message):
    """Show a message dialog with copyable text"""
    try:
        import pyperclip
        pyperclip.copy(message)
        messagebox.showinfo(title, message + "\n\n(Text has been copied to clipboard)")
    except ImportError:
        # If pyperclip is not available, just show the message
        messagebox.showinfo(title, message)

def create_params_dict(app_instance, 
                      include_traffic=True, 
                      include_design=True, 
                      include_simulation=True, 
                      include_flexible=True,
                      include_functions=True):
    """
    Centralized function to create parameter dictionaries for various operations.
    
    Args:
        app_instance: The App instance to get values from
        include_traffic: Include traffic parameters (TPD, vc, cd)
        include_design: Include design parameters (reliability, SN, etc.)
        include_simulation: Include simulation parameters (size, n, seedint)
        include_flexible: Include flexibility parameters (step, capas, etc.)
        include_functions: Include growth functions (mu_function, sigma_function)
    
    Returns:
        dict: Parameter dictionary with requested parameters
    """
    params = {}
    
    if include_traffic:
        params.update({
            "TPD": float(app_instance.tpd.get() or 402.39),
            "vc": float(app_instance.vc.get() or 0.5),
            "cd": float(app_instance.cd.get() or 1.0),
        })
    
    if include_design:
        params.update({
            "Reliavility": float(app_instance.confianza_entry.get() or 0.9),
            "Standard_Deviation": float(app_instance.desviacion_entry.get() or 0.45),
            "Delta_PSI": float(app_instance.delta_psi_entry.get() or 2.0),
            "Mr": float(app_instance.modulo_resiliente_entry.get() or 3000),
            "grade": float(app_instance.grade.get() or 0.0),
            "emb": float(app_instance.emb.get() or 0.0),
            "excv": float(app_instance.exc.get() or 0.0),
        })
    
    if include_simulation:
        params.update({
            "size": int(app_instance.size.get() or 5000),
            "n": int(app_instance.n.get() or 360),
            "seedint": int(app_instance.seedint.get() or 63442967),
        })
    
    if include_flexible:
        # Handle capas parameter - could be from different widgets depending on context
        capas_value = 3  # default
        if hasattr(app_instance, 'layer_count'):
            capas_value = int(app_instance.layer_count.get() or 3)
        elif hasattr(app_instance, 'capas'):
            capas_value = int(app_instance.capas.get() or 3)
            
        params.update({
            "rate": float(app_instance.rate.get() or 0.05),
            "step": round(float(app_instance.intervention_interval_value.get()) * (12 if app_instance.intervention_interval_unit.get() == "Años" else 1)),
            "capas": capas_value,
            "factor": float(app_instance.life_factor.get() or 1.0),
            "cost_rb": app_instance.calculate_rb_cost(),
        })
    
    if include_functions:
        # Set default functions if not available
        if not hasattr(app_instance, 'mean_func'):
            app_instance.mean_func = lambda x: 0.047
        if not hasattr(app_instance, 'std_func'):
            app_instance.std_func = lambda x: 0.057
            
        params.update({
            "mu_function": app_instance.mean_func,
            "sigma_function": app_instance.std_func,
        })
    
    return params

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Method AASHTO")
        
        # Initialize configuration fields
        self.exc = None
        self.emb = None
        self.grade = None
        self.mat_result_label = None
        self.cruta = None
        self.save_button = None
        self.entrys = []
        self.dict_params = {}
        self.acumulated = None
        self.data_series = None
        
        # Bind the close event
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.create_widgets()

    def on_closing(self):
        """Handle cleanup when the application is closing"""
        try:
            # Close all matplotlib figures
            plt.close('all')
            
            # Destroy the root window
            self.master.destroy()
            
            # Force exit if there are any remaining processes
            import sys
            sys.exit(0)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            import sys
            sys.exit(1)

    def create_widgets(self):
        # Create a notebook widget
        notebook = ttk.Notebook(self.master)
        notebook.pack(fill='both', expand=True)

        # Create the tab for calculating SN
        sn_tab = ttk.Frame(notebook)
        notebook.add(sn_tab, text='Calcular SN')
        self.create_sn_widgets(sn_tab)
        
        # Create the tab for create or upload materials.
        mat_tab = ttk.Frame(notebook)
        notebook.add(mat_tab, text='Cargar Materiales')
        self.mat_tab = mat_tab
        DF=self.create_mat_widgets(mat_tab)

        # Create the tab for solution.
        sol_tab = ttk.Frame(notebook)
        notebook.add(sol_tab, text='Solución')
        self.create_sol_widgets(sol_tab)
        
        # Create the tab for design decisions
        design_tab = ttk.Frame(notebook)
        notebook.add(design_tab, text='Decisiones de diseño')
        self.create_graph_widgets(design_tab)

        # Create the tab for traditional design visualization
        trad_design_tab = ttk.Frame(notebook)
        notebook.add(trad_design_tab, text='Diseño Tradicional')
        self.create_trad_design_widgets(trad_design_tab)

        # Pack all tabs
        notebook.pack(expand=1, fill='both')

    def create_traffic_params_widgets(self, tab, notebook):
        # Create a frame for traffic parameters
        traffic_params_frame = ttk.LabelFrame(tab, text="Parámetros de Tráfico", padding="10")
        traffic_params_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Etiquetas y cajas de entrada para recalcular capas
        ttk.Label(traffic_params_frame, text="TPD:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.tpd = ttk.Entry(traffic_params_frame)
        self.tpd.insert(0, "402.39")
        self.tpd.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        ttk.Label(traffic_params_frame, text="archivo npy:").grid(row=0, column=3, padx=10, pady=5, sticky="w")
        self.arr_ruta = ttk.Entry(traffic_params_frame)
        self.arr_ruta.insert(0, "datos_res")
        self.arr_ruta.grid(row=0, column=4, padx=10, pady=5, sticky="w") 
        
        ttk.Label(traffic_params_frame, text="vc:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.vc = ttk.Entry(traffic_params_frame)
        self.vc.insert(0, "0.5")
        self.vc.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        ttk.Label(traffic_params_frame, text="cd:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.cd = ttk.Entry(traffic_params_frame)
        self.cd.insert(0, "1.0")
        self.cd.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        
        ttk.Label(traffic_params_frame, text="size:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.size = ttk.Entry(traffic_params_frame)
        self.size.insert(0, "5000")
        self.size.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        
        ttk.Label(traffic_params_frame, text="n:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.n = ttk.Entry(traffic_params_frame)
        self.n.insert(0, "360")
        self.n.grid(row=4, column=1, padx=10, pady=5, sticky="w")
        
        ttk.Label(traffic_params_frame, text="rate:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.rate = ttk.Entry(traffic_params_frame)
        self.rate.insert(0, "0.05")
        self.rate.grid(row=5, column=1, padx=10, pady=5, sticky="w")
        
        ttk.Label(traffic_params_frame, text="cost_rb:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.cost_rb = ttk.Entry(traffic_params_frame)
        self.cost_rb.insert(0, "1000")
        self.cost_rb.grid(row=6, column=1, padx=10, pady=5, sticky="w")
        
        ttk.Label(traffic_params_frame, text="seedint:").grid(row=9, column=0, padx=10, pady=5, sticky="w")
        self.seedint = ttk.Entry(traffic_params_frame)
        self.seedint.insert(0, "63442967")
        self.seedint.grid(row=9, column=1, padx=10, pady=5, sticky="w")
        
        # Add Load button
        ttk.Button(traffic_params_frame, text="Simular Tráfico", command=lambda: self.create_rand_graph_widgets(notebook)).grid(row=0, column=5, columnspan=2, pady=10)
        
        # Add a description label
        description = "Estos parámetros se utilizan para simular el tráfico y evaluar el comportamiento del pavimento a lo largo del tiempo."
        ttk.Label(traffic_params_frame, text=description, wraplength=400).grid(row=10, column=0, columnspan=6, padx=10, pady=10, sticky="w")

    def create_param_rand_widgets(self, tab, notebook):
        self.create_traffic_params_widgets(tab, notebook)

    def create_rand_graph_widgets(self,notebook):
        """Create random graph widgets with current parameters"""
        # Load materials
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, str(self.cruta.get()) + ".csv")
        DF = cargar_materiales(csv_path)
            
        if len(DF) == 0:
            messagebox.showerror("Error", "No materials available for calculation")
            return
                
        # Get current solution
        solutions = solve(DF, self.calcular_sn(), float(self.grade.get() or 0.0), 
                           float(self.emb.get() or 0.0), float(self.exc.get() or 0.0))
        if not solutions:
            messagebox.showerror("Error", "No valid solutions found")
            return
                
        # Store all solutions and update the selector
        self.all_solutions = solutions
        self.solution_selector['values'] = list(range(len(solutions)))
        
        # Use selected solution or default to first
        if hasattr(self, 'solution_index'):
            index = min(self.solution_index.get(), len(solutions)-1)
            self.solution_index.set(index)
            sect = solutions[index]
        else:
            self.solution_index.set(0)
            sect = solutions[0]
            
        # Calculate flexibility or load from file
        ruta_arr = os.path.join(script_dir, str(self.arr_ruta.get()) + ".csv")
        result = None
        # Check if results file exists
        if os.path.exists(ruta_arr):
            try:
                # Try to read existing results
                results_df = pd.read_csv(ruta_arr)
                # Use the loaded DataFrame directly instead of recreating
                result = results_df
                print(f"Loaded {len(results_df)} simulation results from {ruta_arr}")
            except Exception as e:
                show_copyable_message("Warning", f"Failed to read results from {ruta_arr}:\n{str(e)}")
                result = None
        else:
            # If no results exist or failed to read, run simulation
            if result is None:
                # Update parameters dictionary using centralized function
                self.dict_params = create_params_dict(self, 
                                                    include_traffic=True,
                                                    include_design=True, 
                                                    include_simulation=True,
                                                    include_flexible=True,
                                                    include_functions=True)
                
                # Add specific parameters for this operation
                self.dict_params.update({
                    "sn_design": self.calcular_sn() if "sn_design" not in self.dict_params.keys() else self.dict_params['sn_design'],
                    "sect": sect,
                })
                
                result, self.acumulated = evaluate_flexibility(self.dict_params,DF)
                # Save results
                try:
                    # Save the full results DataFrame directly
                    result.to_csv(ruta_arr, index=False)
                    pd.DataFrame(self.acumulated).to_csv(ruta_arr + "_acumulated.csv", index=False)
                except Exception as e:
                    show_copyable_message('Warning', f'Failed to save results to {ruta_arr}: {str(e)}')
            else:
                # This else block seems unreachable now, but keeping for safety
                try:
                    results_df = pd.read_csv(ruta_arr)
                    result = results_df
                except Exception as e:
                    show_copyable_message("Warning", f"Failed to read results from {ruta_arr}:\n{str(e)}")
                    result = None
        
        # Create new tab for results
        rand_tab = ttk.Frame(notebook)
        notebook.add(rand_tab, text='Random Results')
            
        # Create figure with two subplots
        fig = Figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)  # Top subplot for SN vs Time
        ax2 = fig.add_subplot(212)  # Bottom subplot for Accumulated Traffic
        
        # Plot SN vs Time for each simulation
        # Check if result is the DataFrame format with columns or the matrix format
        if hasattr(result, 'columns') and 'simulation' in result.columns:
            # DataFrame format with simulation column
            for sim in range(min(10, result['simulation'].max() + 1)):
                # Get all periods for this simulation
                sim_data = result[result['simulation'] == sim]
                periods = sim_data['period'].values
                sn_values = sim_data['total_sn'].values
                
                ax1.plot(periods, sn_values, alpha=0.5, label=f'Sim {sim+1}')
        else:
            # Matrix format or other structure - use alternative approach
            try:
                # If result is the original DataFrame from evaluate_flexibility
                for sim in range(min(10, len(result))):
                    if hasattr(result, 'iloc'):
                        sim_data = result.iloc[sim]
                        if hasattr(sim_data, 'period'):
                            periods = sim_data['period'].values if hasattr(sim_data['period'], 'values') else [0]
                            sn_values = sim_data['total_sn'].values if hasattr(sim_data['total_sn'], 'values') else [0]
                        else:
                            # Fallback - create dummy data for display
                            periods = list(range(10))
                            sn_values = [3.0 + sim * 0.1] * 10
                    else:
                        # Fallback - create dummy data for display
                        periods = list(range(10))
                        sn_values = [3.0 + sim * 0.1] * 10
                    
                    ax1.plot(periods, sn_values, alpha=0.5, label=f'Sim {sim+1}')
            except Exception as e:
                # Ultimate fallback - show placeholder message
                ax1.text(0.5, 0.5, f'Data visualization error: {str(e)}', 
                        transform=ax1.transAxes, ha='center', va='center')
        
        ax1.set_title('Structural Number (SN) Over Time')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('SN')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend()
        
        # Plot Accumulated Traffic
        for i in range(min(10, len(self.acumulated))):  # Plot first 10 simulations
            ax2.plot(range(len(self.acumulated[i])), self.acumulated[i], 
                    alpha=0.5, label=f'Sim {i+1}')
        
        ax2.set_title('Accumulated Traffic Over Time')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Accumulated Traffic')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend()
        
        # Adjust layout
        fig.tight_layout()
            
        canvas = FigureCanvasTkAgg(fig, master=rand_tab)
        canvas.draw()
            
        # Add navigation toolbar
        toolbar = NavigationToolbar2Tk(canvas, rand_tab)
        toolbar.update()
            
        # Pack widgets
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add export to Excel button
        def export_to_excel():
            try:
                # Save the full results DataFrame directly
                file_path = filedialog.asksaveasfilename(
                    defaultextension='.xlsx',
                    filetypes=[('Excel files', '*.xlsx')],
                    title='Save Results as Excel File'
                )
                if file_path:
                    with pd.ExcelWriter(file_path) as writer:
                        result.to_excel(writer, sheet_name='Section Results', index=False)
                        pd.DataFrame(self.acumulated).to_excel(writer, sheet_name='Accumulated Traffic', index=False)
                    messagebox.showinfo('Success', 'Results exported successfully!')
            except Exception as e:
                show_copyable_message('Error', f'Failed to export results: {str(e)}')
        
        export_btn = ttk.Button(rand_tab, text='Export to Excel', command=export_to_excel)
        export_btn.pack(side=tk.BOTTOM, pady=5)
            
        # Switch to new tab
        notebook.select(rand_tab)
            
    def create_resol_widgets(self,tab):
        """Create widgets for the resolve tab"""
        # Clear existing widgets
        for widget in tab.grid_slaves():
            widget.destroy()
            
        # Add spinbox for solution selection
        ttk.Label(tab, text="Select solution number:").grid(row=0, column=0, padx=10, pady=5)
        self.n_sect = ttk.Spinbox(tab, from_=0, to=4, width=5)
        self.n_sect.grid(row=0, column=1, padx=10, pady=5)
        self.n_sect.set(0)  # Default to first solution
        
        # Add spinbox for number of layers to modify
        ttk.Label(tab, text="Number of layers to modify:").grid(row=1, column=0, padx=10, pady=5)
        self.n_capas = ttk.Spinbox(tab, from_=0, to=10, width=5)
        self.n_capas.grid(row=1, column=1, padx=10, pady=5)
        self.n_capas.set(0)  # Default to all layers
        
        # Add result label
        self.resol_result_label = ttk.Label(tab, text="")
        self.resol_result_label.grid(row=4, column=0, columnspan=4, pady=10)
        
        # Calculate and display initial solution
        self.recalcular_sol()
        
        # Add recalculate button
        ttk.Button(tab, text="Re-Calculate layers", command=self.recalcular_sol).grid(row=3, column=0, columnspan=2, pady=10)
        
    def recalcular_sol(self):
        """Recalculate solution with current parameters"""
        # Load materials
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, str(self.cruta.get()) + ".csv")
        DF = cargar_materiales(csv_path)
            
        if len(DF) == 0:
            self.resol_result_label.config(text="No materials available for calculation")
            return
                
        # Get current parameters
        n_sect = int(self.n_sect.get() or 0)
        n_capas = int(self.n_capas.get() or 0)
        grade = float(self.grade.get() or 0.0)
        emb = float(self.emb.get() or 0.0)
        exc = float(self.exc.get() or 0.0)
            
        # Calculate solution
        all_solutions = solve(DF, self.calcular_sn(), grade, emb, exc)
        if not all_solutions:
            self.resol_result_label.config(text="No valid solutions found")
            return
                
        if n_sect >= len(all_solutions):
            self.resol_result_label.config(text=f"Solution {n_sect} not available. Maximum is {len(all_solutions)-1}")
            return
                
        sect = all_solutions[n_sect]
        resolve(DF, sect, self.calcular_sn(), n_capas, grade, emb, exc)
            
        # Format results
        result_text = "Modified Solution:\n"
        for layer in sect:  
            result_text += f"{layer.name}: {layer.thickness:.2f} inches\n"
        result_text += f"Total Cost: {sect.totalCost:.2f}"
            
        self.resol_result_label.config(text=result_text)
        return sect
            
    def create_sol_widgets(self, tab):
        """Create widgets for the solution tab"""
        # Clear existing widgets
        for widget in tab.grid_slaves():
            widget.destroy()
            
        # Create a frame for the solution list
        solution_list_frame = ttk.Frame(tab)
        solution_list_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
            
        # Calculate solutions
        combined_data = self.calcular_sol()
        
        if not combined_data:  # If no solutions available
            ttk.Label(solution_list_frame, text="No solutions available. Please add materials first.").grid(
                row=0, column=0, columnspan=4, padx=10, pady=20)
        else:
            # Create headers
            headers = ["Material", "Thickness", "Cost", "SN"]
            for i, header in enumerate(headers):
                ttk.Label(solution_list_frame, text=header, font=('Arial', 10, 'bold')).grid(row=0, column=i, padx=10, pady=5)
                
            # Calculate base row for each solution
            current_row = 1
                
            # Display solutions
            for i, solution in enumerate(combined_data, 1):
                # Solution header with spacing
                ttk.Label(solution_list_frame, text=f"Solution {i}", font=('Arial', 10, 'bold')).grid(
                    row=current_row, column=0, columnspan=4, pady=(20,5))
                current_row += 1
                
                # Iterate directly over the Section object
                for layer in solution:
                    # Display layer information
                    ttk.Label(solution_list_frame, text=layer.name).grid(row=current_row, column=0, padx=10, pady=2, sticky='w')
                    ttk.Label(solution_list_frame, text=f"{layer.thickness:.2f}").grid(row=current_row, column=1, padx=10, pady=2)
                    ttk.Label(solution_list_frame, text=f"{layer.cost:.2f}").grid(row=current_row, column=2, padx=10, pady=2)
                    ttk.Label(solution_list_frame, text=f"{layer.sn:.2f}").grid(row=current_row, column=3, padx=10, pady=2)
                    current_row += 1
                
                # Add total cost with a separator line above
                separator = ttk.Frame(solution_list_frame, height=2, relief="groove")
                separator.grid(row=current_row, column=0, columnspan=4, sticky='ew', pady=(5,5))
                current_row += 1
                
                ttk.Label(solution_list_frame, text=f"Total Cost: {solution.totalCost:.2f}", font=('Arial', 10, 'bold')).grid(
                    row=current_row, column=0, columnspan=4, pady=(0,10))
                current_row += 1
                
        # Botón para calcular la solución de capas - place at the top
        ttk.Button(solution_list_frame, text="Calcular nuevas capas", command=self.calcular_sol).grid(
            row=0, column=4, padx=20, pady=5, sticky='ne')

        # Create pavement design frame
        pavement_frame = ttk.LabelFrame(tab, text="Diseño de Pavimento", padding="10")
        pavement_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Add solution selector
        solution_frame = ttk.Frame(pavement_frame)
        solution_frame.pack(fill='x', pady=5)
        
        ttk.Label(solution_frame, text="Solución:").pack(side='left', padx=5)
        self.solution_index = tk.IntVar(value=0)
        self.solution_selector = ttk.Combobox(solution_frame, textvariable=self.solution_index, state='readonly', width=5)
        self.solution_selector.pack(side='left', padx=5)
        self.solution_selector.bind('<<ComboboxSelected>>', self.on_solution_change)
        
        ttk.Button(solution_frame, text="Actualizar", command=self.update_solutions_list).pack(side='right', padx=5)
        
        # Create figure and canvas for pavement design
        self.pavement_fig = Figure(figsize=(3, 4), dpi=100)
        self.pavement_ax = self.pavement_fig.add_subplot(111)
        self.pavement_canvas = FigureCanvasTkAgg(self.pavement_fig, master=pavement_frame)
        self.pavement_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Configure grid weights to make both frames expand properly
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)
        
    def calcular_sol(self):
        """Calcula la solución y devuelve los datos para mostrar en la tabla"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, str(self.cruta.get()) + ".csv")
        DF = cargar_materiales(csv_path)
        if len(DF) == 0:
            self.mat_result_label.config(text="No materials available for calculation")
            return []
                
        lst = solve(DF, self.calcular_sn(), float(self.grade.get() or 0.0), 
                       float(self.emb.get() or 0.0), float(self.exc.get() or 0.0))[:5]
        return lst

    def create_mat_widgets(self, tab):
        # Store the tab reference
        self.mat_tab = tab
        
        # Initialize result label
        self.mat_result_label = ttk.Label(tab, text="")
        self.mat_result_label.grid(row=0, column=7, columnspan=2, pady=10)
        
        # Material file selection
        ttk.Label(tab, text="Cargar datos de materiales de un archivo csv").grid(row=0, column=0, padx=0, pady=5, sticky="w")
        self.cruta = ttk.Entry(tab, width=10)
        self.cruta.insert(0, "default")  # Default value
        self.cruta.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text=".csv").grid(row=0, column=2, padx=0, pady=5, sticky="w")
        
        # Column headers
        titulos = ['mat_name', 'SN', 'min', 'max', 'density', 'cost', 'unit', 'surface', 'subgrade', 'alkaline']
        for i, titulo in enumerate(titulos):
            ttk.Label(tab, text=titulo).grid(row=1, column=i, padx=0, pady=5)

        # Define options for dropdowns
        unit_options = ['ton', 'cyd', 'sqyd']
        bool_options = ['False', 'True']
        
        # Create dictionaries to store the comboboxes
        self.unit_combo = ttk.Combobox(tab, values=unit_options, width=7, state='readonly')
        self.unit_combo.set(unit_options[0])  # Set default value
        self.unit_combo.grid(row=2, column=6, padx=0, pady=5)

        self.surface_combo = ttk.Combobox(tab, values=bool_options, width=7, state='readonly')
        self.surface_combo.set(bool_options[0])
        self.surface_combo.grid(row=2, column=7, padx=0, pady=5)

        self.subgrade_combo = ttk.Combobox(tab, values=bool_options, width=7, state='readonly')
        self.subgrade_combo.set(bool_options[0])
        self.subgrade_combo.grid(row=2, column=8, padx=0, pady=5)

        self.alkaline_combo = ttk.Combobox(tab, values=bool_options, width=7, state='readonly')
        self.alkaline_combo.set(bool_options[0])
        self.alkaline_combo.grid(row=2, column=9, padx=0, pady=5)
        
        # Load materials from default.csv
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ruta = os.path.join(script_dir, str(self.cruta.get()) + ".csv")
        return self.cargar_mat(tab, ruta)
    
    def crear_materiales(self,):
        # Create a new material
        lst_entrys = self.entrys
        new_mat = []
        all_empty = True  # Track if all fields are empty
        
        for i in range(len(lst_entrys)):
            # Handle different types of input
            val = lst_entrys[i].get().strip()  # Remove whitespace
            if val:  # If any field has a value, not all empty
                all_empty = False
                
            if i == 0 or i == 6:  # name and unit are strings
                new_mat.append(val)
            elif i in [7, 8, 9]:  # surface, subgrade, alkaline are booleans
                new_mat.append(val.lower() == "true" or val == "1")
            else:  # rest are floats
                new_mat.append(float(val) if val else 0.0)
        
        # Add new material to DataFrame only if not all fields are empty
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, str(self.cruta.get()) + ".csv")
        
        if all_empty:
            self.mat_result_label.config(text="No material data entered")
            # Just reload the current file
            self.cargar_mat(self.mat_tab, csv_path)
            return None
            
        # Use cargar_materiales to ensure file exists with proper headers
        df = cargar_materiales(csv_path)
        new_row = pd.DataFrame([new_mat], columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_path, index=False)
        self.mat_result_label.config(text="Material added successfully")
        
        # Refresh the display
        self.cargar_mat(self.mat_tab, csv_path)
        return df

    def cargar_mat(self,tab,ruta):
        # Clear existing widgets except for permanent ones
        preserved_widgets = []
        if hasattr(self, 'mat_result_label') and self.mat_result_label:
            preserved_widgets.append(self.mat_result_label)
        if hasattr(self, 'cruta') and self.cruta:
            preserved_widgets.append(self.cruta)
            
        for widget in tab.grid_slaves():
            if isinstance(widget, (ttk.Label, ttk.Entry, ttk.Combobox)) and widget not in preserved_widgets:
                widget.destroy()
        
        # Mostrar el resultado de cargar el material en la interfaz
        DF_mat = cargar_materiales(ruta)
        
        # If the DataFrame is empty (new file), create a template row with empty values
        if len(DF_mat) == 0:
            self.mat_result_label.config(text="Created new material file template")
            # Create empty DataFrame with correct columns
            DF_mat = pd.DataFrame(columns=['mat_name','SN','min','max','density','cost','unit','surface','subgrade','alkaline'])
        
        # Re-add title labels
        titulos=['mat_name','SN','min','max','density','cost','unit','surface','subgrade','alkaline']
        for i in range(len(titulos)):
            ttk.Label(tab, text=titulos[i]).grid(row=1, column=i, padx=0, pady=5)
            
        # Add material data
        for i in range(len(DF_mat)):
            for j in range(len(DF_mat.iloc[i])):
                ttk.Label(tab, text=str(DF_mat.iloc[i].iloc[j])).grid(row=i+2, column=j, padx=10, pady=1)
        
        # Calculate the row for new entries (after the last material)
        entry_row = len(DF_mat) + 2
        
        # Define options for dropdowns
        unit_options = ['ton', 'cyd', 'sqyd']
        bool_options = ['False', 'True']
        
        # Create or update entry fields for new material
        self.entrys = []
        for k in range(len(titulos)):
            if titulos[k] == 'unit':
                entry = ttk.Combobox(tab, values=unit_options, width=7, state='readonly')
                entry.set(unit_options[0])
            elif titulos[k] in ['surface', 'subgrade', 'alkaline']:
                entry = ttk.Combobox(tab, values=bool_options, width=7, state='readonly')
                entry.set(bool_options[0])
            else:
                entry = ttk.Entry(tab, width=10)
            entry.grid(row=entry_row, column=k, padx=10, pady=5, sticky="w")
            self.entrys.append(entry)
                
        # Add save button if it doesn't exist
        if not hasattr(self, 'save_button') or not self.save_button:
            self.save_button = ttk.Button(tab, text="Guardar material", command=self.crear_materiales)
            self.save_button.grid(row=0, column=5, columnspan=2, pady=10)
            
        # Add additional configuration fields
        config_start_row = entry_row + 1
        
        # Header for configuration section
        ttk.Label(tab, text="Configuration", font=('Arial', 10, 'bold')).grid(row=config_start_row, column=0, columnspan=2, pady=(20,5), sticky="w")
        
        # Excavation cost field
        ttk.Label(tab, text="Excavation Cost ($/cyd)").grid(row=config_start_row+1, column=0, padx=0, pady=0, sticky="w")        
        self.exc = ttk.Entry(tab, width=10)
        if not hasattr(self, 'exc_value'):
            self.exc_value = "20"
        self.exc.insert(0, self.exc_value)
        self.exc.grid(row=config_start_row+1, column=1, padx=10, pady=5, sticky="w")
        
        # Embankment cost field
        ttk.Label(tab, text="Embankment Cost ($/cyd)").grid(row=config_start_row+2, column=0, padx=0, pady=0, sticky="w")        
        self.emb = ttk.Entry(tab, width=10)
        if not hasattr(self, 'emb_value'):
            self.emb_value = "10"
        self.emb.insert(0, self.emb_value)
        self.emb.grid(row=config_start_row+2, column=1, padx=10, pady=0, sticky="w")
        
        # Grade field
        ttk.Label(tab, text="Grade (in)").grid(row=config_start_row+3, column=0, padx=0, pady=0, sticky="w")        
        self.grade = ttk.Entry(tab, width=10)
        if not hasattr(self, 'grade_value'):
            self.grade_value = "0.0"
        self.grade.insert(0, self.grade_value)
        self.grade.grid(row=config_start_row+3, column=1, padx=10, pady=5, sticky="w")
            
        return DF_mat

    def create_sn_widgets(self, tab):
        """Create widgets for the SN calculation tab"""
        # Create a frame for better organization
        frame = ttk.Frame(tab, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input fields
        ttk.Label(frame, text="Nivel de Confianza:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.confianza_entry = ttk.Entry(frame)
        self.confianza_entry.insert(0, "0.95")
        self.confianza_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(frame, text="Desviación Estándar:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.desviacion_entry = ttk.Entry(frame)
        self.desviacion_entry.insert(0, "0.35")
        self.desviacion_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(frame, text="NESE:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.W18_entry = ttk.Entry(frame)
        self.W18_entry.insert(0, "5000000")
        self.W18_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(frame, text="ΔPSI:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.delta_psi_entry = ttk.Entry(frame)
        self.delta_psi_entry.insert(0, "1.9")
        self.delta_psi_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(frame, text="Módulo Resiliente (PSI):").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.modulo_resiliente_entry = ttk.Entry(frame)
        self.modulo_resiliente_entry.insert(0, "5000")
        self.modulo_resiliente_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        # Calculate button
        ttk.Button(frame, text="Calcular SN", command=self.calcular_sn).grid(row=5, column=0, columnspan=2, pady=10)

        # Result label
        self.sn_result_label = ttk.Label(frame, text="", font=('Arial', 12, 'bold'))
        self.sn_result_label.grid(row=6, column=0, columnspan=2, pady=10)
    def calcular_sn(self):
        """Calculate SN value with current parameters"""
        confianza = float(self.confianza_entry.get() or 0.9)  # Default 0.9 if empty
        desviacion = float(self.desviacion_entry.get() or 0.45)  # Default 0.45 if empty
        delta_psi = float(self.delta_psi_entry.get() or 2.0)  # Default 2.0 if empty
        modulo_resiliente = float(self.modulo_resiliente_entry.get() or 3000)  # Default 3000 if empty
        esal = float(self.W18_entry.get() or 1000000)  # Default 1M if empty
            
        if modulo_resiliente <= 0:
            self.mat_result_label.config(text="Error: Resilient modulus must be positive")
            return 0.0
            
        sn = solve_sn(confianza, desviacion, delta_psi, modulo_resiliente, esal)
        if sn is None:
            self.sn_result_label.config(text="Error: Could not calculate SN. Please check input ranges:\n" +
                                           "Confianza: 0.5-0.999\n" +
                                           "Desviación: 0.3-0.5\n" +
                                           "ΔPSI: 1.0-3.0", foreground="red")
            return 0.0
            
        # Show the result in the interface
        self.sn_result_label.config(text=f"SN = {sn:.2f}", foreground="dark green")
        print(f"El valor calculado de SN es: {sn:.2f}")
        self.dict_params['sn_design'] = sn
        return sn

    def create_graph_widgets(self, parent):
        # Create a notebook widget for the design decisions tab
        design_notebook = ttk.Notebook(parent)
        design_notebook.pack(fill='both', expand=True)
        
        # Create the main growth parameters tab
        growth_tab = ttk.Frame(design_notebook)
        design_notebook.add(growth_tab, text='Parámetros de Crecimiento')
        
        # Create the traffic parameters tab
        traffic_tab = ttk.Frame(design_notebook)
        design_notebook.add(traffic_tab, text='Parámetros de Tráfico')
        
        # Add traffic parameters to the traffic tab
        self.create_traffic_params_widgets(traffic_tab, design_notebook)
        
        # Initialize global parameters
        self.global_params = {
            'x_start': tk.DoubleVar(value=0.0),
            'x_end': tk.DoubleVar(value=12.0),
            'noise': tk.DoubleVar(value=0.0),
            'generate_std': tk.BooleanVar(value=False),
            'std_manual': tk.StringVar(value="")
        }

        # Initialize function parameters
        self.function_params = {
            'linear': {
                'slope': tk.DoubleVar(value=1.0),
                'intercept': tk.DoubleVar(value=0.0)
            },
            'logarithmic': {
                'a': tk.DoubleVar(value=1.0),
                'b': tk.DoubleVar(value=0.0)
            },
            'exponential': {
                'a': tk.DoubleVar(value=1.0),
                'b': tk.DoubleVar(value=0.1)
            },
            'polynomial': {
                'a': tk.DoubleVar(value=1.0),
                'b': tk.DoubleVar(value=0.0),
                'c': tk.DoubleVar(value=0.0)
            }
        }

        param_frame = ttk.LabelFrame(growth_tab, text="Parámetros de crecimiento de tráfico", padding="10")
        param_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Fitting type selection
        ttk.Label(param_frame, text="Tipo de Ajuste:").grid(row=0, column=0, padx=5, pady=5)
        self.fit_type = tk.StringVar(value="manual")
        ttk.Radiobutton(param_frame, text="Manual", variable=self.fit_type, value="manual").grid(row=0, column=1, padx=5, pady=5)
        ttk.Radiobutton(param_frame, text="Autoajuste", variable=self.fit_type, value="auto").grid(row=0, column=2, padx=5, pady=5)

        # Create container for dynamic parameters
        self.manual_params_container = ttk.LabelFrame(param_frame, text="Parámetros de Ajuste Manual", padding="10")
        self.auto_params_container = ttk.LabelFrame(param_frame, text="Parámetros de Autoajuste", padding="10")
        
        # Pack containers
        self.manual_params_container.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        self.auto_params_container.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        
        # Initially show manual parameters
        self.auto_params_container.grid_remove()
        
        # Bind fit type change
        self.fit_type.trace('w', self.update_parameter_ui)
        
        # Cargar datos button
        ttk.Button(param_frame, text="Cargar datos de crecimiento", command=self.load_data_series).grid(row=2, column=0, padx=5, pady=5)
        self.data_label = ttk.Label(param_frame, text="Archivo: Ninguno")
        self.data_label.grid(row=2, column=1, padx=5, pady=5)
        
        # Add button to plot simulated transit with tooltip
        transit_btn = ttk.Button(param_frame, text="Simular y Graficar Tránsito", command=self.plot_transit_simulation)
        transit_btn.grid(row=2, column=2, padx=5, pady=5)
        
        # Add a label to explain the simulation plots
        transit_info = ttk.Label(param_frame, text="Genera gráficos detallados del tránsito acumulado y mensual", 
                                 font=("Arial", 8), foreground="gray")
        transit_info.grid(row=3, column=0, columnspan=3, padx=5, pady=2, sticky="w")

        # Create plot frame
        graph_frame = ttk.LabelFrame(growth_tab, text="Gráfico", padding="10")
        graph_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Create figure and canvas
        self.graph_fig = Figure(figsize=(6, 4), dpi=100)
        self.graph_ax = self.graph_fig.add_subplot(111)
        self.graph_canvas = FigureCanvasTkAgg(self.graph_fig, master=graph_frame)
        self.graph_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Initialize selection
        self.selected_functions = []
        self.update_parameter_ui()

        # Intervalos de Intervención Frame
        interval_frame = ttk.LabelFrame(growth_tab, text="Intervalos de Intervención", padding="10")
        interval_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Input parameters
        ttk.Label(interval_frame, text="Intervalo de Intervención:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.intervention_interval_value = tk.DoubleVar(value=10)
        self.intervention_interval_unit = tk.StringVar(value="Años")
        ttk.Entry(interval_frame, textvariable=self.intervention_interval_value, width=10).grid(row=0, column=1, padx=5, pady=5)
        ttk.Combobox(interval_frame, textvariable=self.intervention_interval_unit, values=["Años", "Meses"], state="readonly", width=10).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(interval_frame, text="Factor de Vida Útil (0-1):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.life_factor = tk.DoubleVar(value=0.85)
        ttk.Entry(interval_frame, textvariable=self.life_factor, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(interval_frame, text="Capas de pavimento \na intervenir en el periodo:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.capas = tk.IntVar(value=2)
        ttk.Entry(interval_frame, textvariable=self.capas, width=10).grid(row=2, column=1, padx=5, pady=5)

    def update_parameter_ui(self, *args):
        # Clear existing widgets
        for widget in self.manual_params_container.winfo_children():
            widget.destroy()
        for widget in self.auto_params_container.winfo_children():
            widget.destroy()

        if self.fit_type.get() == "manual":
            # Show manual parameters
            self.auto_params_container.grid_remove()
            self.manual_params_container.grid()
            
            # Manual fitting parameters
            ttk.Label(self.manual_params_container, text="Funciones").grid(row=0, column=0, padx=5, pady=5)
            self.function_listbox = tk.Listbox(self.manual_params_container, selectmode=tk.MULTIPLE, height=3)
            self.function_listbox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            for func in ['linear', 'logarithmic', 'exponential', 'polynomial']:
                self.function_listbox.insert(tk.END, func)
            self.function_listbox.bind('<<ListboxSelect>>', self.on_function_select)

            # Add parameter controls for each selected function
            row = 1
            for func_name in self.selected_functions:
                ttk.Label(self.manual_params_container, text=f"Parámetros {func_name}").grid(row=row, column=0, columnspan=2, padx=5, pady=5)
                row += 1
                
                params = self.function_params[func_name]
                for param_name, param_var in params.items():
                    ttk.Label(self.manual_params_container, text=param_name).grid(row=row, column=0, padx=5, pady=2)
                    ttk.Entry(self.manual_params_container, textvariable=param_var).grid(row=row, column=1, padx=5, pady=2)
                    row += 1

            # Add X-range and noise controls only if functions are selected
            if self.selected_functions:
                ttk.Label(self.manual_params_container, text="X Inicial").grid(row=row, column=0, padx=5, pady=5)
                ttk.Entry(self.manual_params_container, textvariable=self.global_params['x_start']).grid(row=row, column=1, padx=5, pady=5)
                row += 1
                
                ttk.Label(self.manual_params_container, text="X Final").grid(row=row, column=0, padx=5, pady=5)
                ttk.Entry(self.manual_params_container, textvariable=self.global_params['x_end']).grid(row=row, column=1, padx=5, pady=5)
                row += 1

                # Noise controls
                ttk.Label(self.manual_params_container, text="Nivel de Ruido").grid(row=row, column=0, padx=5, pady=5)
                ttk.Entry(self.manual_params_container, textvariable=self.global_params['noise']).grid(row=row, column=1, padx=5, pady=5)
                row += 1

                # Std deviation controls
                ttk.Checkbutton(self.manual_params_container, text="Generar desviación con datos", variable=self.global_params['generate_std'], command=self.update_parameter_ui).grid(row=row, column=0, columnspan=2, padx=5, pady=5)
                row += 1
                if not self.global_params['generate_std'].get():
                    ttk.Label(self.manual_params_container, text="Función de Desviación").grid(row=row, column=0, padx=5, pady=5)
                    ttk.Entry(self.manual_params_container, textvariable=self.global_params['std_manual']).grid(row=row, column=1, padx=5, pady=5)
                    row += 1

                # Add Update Graph button
                ttk.Button(self.manual_params_container, text="Actualizar Gráfico", command=self.update_graph).grid(row=row, column=0, columnspan=2, pady=5)

        else:  # auto
            # Show auto parameters
            self.manual_params_container.grid_remove()
            self.auto_params_container.grid()
            
            # Auto fitting parameters
            ttk.Label(self.auto_params_container, text="Tipo de Ajuste Automático:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            fit_types = ['linear', 'log', 'exp', 'poly']
            self.auto_fit_type = tk.StringVar(value="linear")
            ttk.Combobox(self.auto_params_container, textvariable=self.auto_fit_type, values=fit_types, state='readonly').grid(row=0, column=1, padx=5, pady=5)
            
            ttk.Button(self.auto_params_container, text="Ajustar Automáticamente", command=self.auto_fit).grid(row=1, column=0, columnspan=2, pady=5)

    def auto_fit(self):
        if self.data_series is None:
            messagebox.showerror("Error", "Por favor, cargue datos primero")
            return

        try:
            # Extract data
            x_data = self.data_series[:, 0]
            y_data = self.data_series[:, 1]
            
            # Get selected fit type
            fit_type = self.auto_fit_type.get()
            
            # Perform fitting
            if fit_type == 'linear':
                popt, _ = curve_fit(lambda x, a, b: a * x + b, x_data, y_data)
                self.mean_func = lambda x: popt[0] * x + popt[1]
                self.std_func = lambda x: np.std(y_data - self.mean_func(x_data))
                
                # Update manual parameters if linear is selected
                if 'linear' in self.selected_functions:
                    self.function_params['linear']['slope'].set(popt[0])
                    self.function_params['linear']['intercept'].set(popt[1])
                    
            elif fit_type == 'log':
                popt, _ = curve_fit(lambda x, a, b: a * np.log(x + 1) + b, x_data, y_data)
                self.mean_func = lambda x: popt[0] * np.log(x + 1) + popt[1]
                self.std_func = lambda x: np.std(y_data - self.mean_func(x_data))
                
                # Update manual parameters if logarithmic is selected
                if 'logarithmic' in self.selected_functions:
                    self.function_params['logarithmic']['a'].set(popt[0])
                    self.function_params['logarithmic']['b'].set(popt[1])
                    
            elif fit_type == 'exp':
                popt, _ = curve_fit(lambda x, a, b: a * np.exp(b * x), x_data, y_data)
                self.mean_func = lambda x: popt[0] * np.exp(popt[1] * x)
                self.std_func = lambda x: np.std(y_data - self.mean_func(x_data))
                
                # Update manual parameters if exponential is selected
                if 'exponential' in self.selected_functions:
                    self.function_params['exponential']['a'].set(popt[0])
                    self.function_params['exponential']['b'].set(popt[1])
                    
            else:  # poly
                popt = np.polyfit(x_data, y_data, 2)
                self.mean_func = np.poly1d(popt)
                self.std_func = lambda x: np.std(y_data - self.mean_func(x_data))
                
                # Update manual parameters if polynomial is selected
                if 'polynomial' in self.selected_functions:
                    self.function_params['polynomial']['a'].set(popt[0])
                    self.function_params['polynomial']['b'].set(popt[1])
                    self.function_params['polynomial']['c'].set(popt[2])
            
            # Update graph
            self.update_graph()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en el ajuste automático:\n{str(e)}")
        
        self.update_pavement_design()

    def update_graph(self):
        try:
            if self.data_series is None:
                return

            # Clear previous plot
            self.graph_ax.clear()
            
            # Plot data series as bars
            x_data = self.data_series[:, 0]
            y_data = self.data_series[:, 1]
            self.graph_ax.bar(x_data, y_data, alpha=0.3, 
                            color='gray', width=0.8, 
                            label='Datos Experimentales', 
                            edgecolor='black')
            self.graph_ax.scatter(x_data, y_data, color='red', label='Datos usuario')

            # Plot selected manual functions
            if self.fit_type.get() == "manual" and self.selected_functions:
                x = np.linspace(min(x_data), max(x_data), 100)
                y_total = np.zeros_like(x)
                for func_name in self.selected_functions:
                    params = self.function_params[func_name]
                    if func_name == 'linear':
                        y = params['slope'].get() * x + params['intercept'].get()
                    elif func_name == 'logarithmic':
                        y = params['a'].get() * np.log(x + 1) + params['b'].get()
                    elif func_name == 'exponential':
                        y = params['a'].get() * np.exp(params['b'].get() * x)
                    else:  # polynomial
                        y = params['a'].get() * x**2 + params['b'].get() * x + params['c'].get()
                    y_total += y

                self.graph_ax.plot(x, y_total, color='blue', label='Función Combinada')

                # Calculate and display standard deviation
                if hasattr(self, 'std_func'):
                    y_std = self.std_func(x)
                    y_upper = y_total + 2 * y_std
                    y_lower = y_total - 2 * y_std
                    self.graph_ax.fill_between(x, y_lower, y_upper, color='skyblue', alpha=0.3, label='±2 Desv. Est.')

            # Plot fitted function if available
            elif hasattr(self, 'mean_func') and hasattr(self, 'std_func'):
                x = np.linspace(min(x_data), max(x_data), 100)
                y_mean = self.mean_func(x)
                y_std = self.std_func(x)
                y_upper = y_mean + 2 * y_std
                y_lower = y_mean - 2 * y_std

                self.graph_ax.plot(x, y_mean, color='blue', label='Media estimada')
                self.graph_ax.fill_between(x, y_lower, y_upper, color='skyblue', alpha=0.3, label='±2 Desv. Est.')

            # Configure plot
            self.graph_ax.set_title("Crecimiento del Tráfico")
            self.graph_ax.set_xlabel("Período")
            self.graph_ax.set_ylabel("Crecimiento (%)")
            self.graph_ax.grid(True, linestyle='--', alpha=0.7)
            self.graph_ax.legend(loc='upper left')

            self.graph_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Error actualizando gráfico:\n{str(e)}")
        
        self.update_pavement_design()

    def update_pavement_design(self):
        """Update the pavement design visualization"""
        # Check if we have a solution
        sect = None
        
        if hasattr(self, 'dict_params') and 'sect' in self.dict_params:
            sect = self.dict_params['sect']
        elif hasattr(self, 'all_solutions') and self.all_solutions:
            # Use the currently selected solution
            index = self.solution_index.get()
            if 0 <= index < len(self.all_solutions):
                sect = self.all_solutions[index]
        else:
            # Try to get a solution
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(script_dir, str(self.cruta.get()) + ".csv")
                DF = cargar_materiales(csv_path)
                
                if len(DF) == 0:
                    return
                    
                solutions = solve(DF, self.calcular_sn(), float(self.grade.get() or 0.0), 
                                float(self.emb.get() or 0.0), float(self.exc.get() or 0.0))
                if not solutions:
                    return
                
                # Store all solutions and update the selector
                self.all_solutions = solutions
                self.solution_selector['values'] = list(range(len(solutions)))
                self.solution_index.set(0)
                
                sect = solutions[0]
            except Exception as e:
                print(f"Error getting pavement design: {str(e)}")
                return
        
        if not sect:
            return
            
        # Clear the plot
        self.pavement_ax.clear()
        
        # Create a stacked bar chart for the pavement layers
        bottom = 0
        y_positions = []
        layer_names = []
        colors = plt.cm.tab10.colors  # Use a colormap for different layers
        
        # Process layers from bottom to top (reverse order for visualization)
        for i, layer in enumerate(reversed(sect)):
            thickness = layer.thickness
            y_positions.append(bottom + thickness/2)  # For label positioning
            self.pavement_ax.bar(0, thickness, bottom=bottom, width=0.6, 
                              color=colors[i % len(colors)], 
                              edgecolor='black', linewidth=1)
            bottom += thickness
            layer_names.append(f"{layer.name}\n{thickness:.1f} {layer.unit}")
        
        # Add layer labels
        for i, (y, name) in enumerate(zip(y_positions, layer_names)):
            self.pavement_ax.text(0, y, name, ha='center', va='center', 
                               fontsize=8, fontweight='bold')
        
        # Set axis properties
        self.pavement_ax.set_xlim(-0.5, 0.5)
        self.pavement_ax.set_xticks([])
        self.pavement_ax.set_ylabel('Espesor (pulgadas)')
        self.pavement_ax.set_title('Diseño de Pavimento')
        
        # Add total thickness and SN
        total_thickness = sum(layer.thickness for layer in sect)
        total_sn = sum(layer.thickness * layer.sn for layer in sect)
        self.pavement_ax.text(0, -0.1, f"Espesor total: {total_thickness:.1f}\nSN total: {total_sn:.2f}", 
                           ha='center', transform=self.pavement_ax.transAxes)
        
        # Redraw the canvas
        self.pavement_canvas.draw()

    def load_data_series(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo de datos",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                # Read raw data as strings
                raw_data = np.loadtxt(file_path, dtype=str, )
                # Convert commas to periods and parse to floats
                y_values = [float(item.replace(',', '.')) for item in raw_data]
                # Generate x values as indices starting from 0
                x_values = np.arange(len(y_values))
                # Create 2D array with x indices and y values
                self.data_series = np.column_stack((x_values, y_values))
                self.data_label.config(text=f"Archivo: {os.path.basename(file_path)}")
                self.update_graph()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando datos:\n{str(e)}")
                self.data_series = None
                self.data_label.config(text="Archivo: Ninguno")

    def on_solution_change(self, event=None):
        """Handle solution selection change"""
        if hasattr(self, 'all_solutions') and self.all_solutions:
            index = self.solution_index.get()
            if 0 <= index < len(self.all_solutions):
                # Update the selected solution
                if hasattr(self, 'dict_params'):
                    self.dict_params['sect'] = self.all_solutions[index]
                # Update the visualization
                self.update_pavement_design()
    
    def update_solutions_list(self):
        """Update the solution selector with available solutions"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, str(self.cruta.get()) + ".csv")
            DF = cargar_materiales(csv_path)
            
            if len(DF) == 0:
                messagebox.showerror("Error", "No materials available for calculation")
                return
                
            solutions = solve(DF, self.calcular_sn(), float(self.grade.get() or 0.0), 
                            float(self.emb.get() or 0.0), float(self.exc.get() or 0.0))
            
            if not solutions:
                messagebox.showerror("Error", "No valid solutions found")
                return
            
            # Store solutions and update selector
            self.all_solutions = solutions
            self.solution_selector['values'] = list(range(len(solutions)))
            
            # Set current selection to first solution
            self.solution_index.set(0)
            
            # Update dict_params if it exists
            if hasattr(self, 'dict_params') and 'sect' in self.dict_params:
                self.dict_params['sect'] = solutions[0]
            
            # Update visualization
            self.update_pavement_design()
            
            messagebox.showinfo("Success", f"Found {len(solutions)} valid solutions")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error updating solutions list: {str(e)}")

    def on_function_select(self, event):
        selected_indices = self.function_listbox.curselection()
        self.selected_functions = [self.function_listbox.get(i) for i in selected_indices]
        self.update_parameter_ui()
        self.update_graph()
        
    def calculate_rb_cost(self):
        """
        Calculate the rubble (rb) cost
        """
        # This is a placeholder - implement the actual calculation based on your requirements
        return 0.0

    def plot_transit_simulation(self):
        """
        Plot the simulated transit using the parameters from the GUI
        """
        try:
            # Show a message indicating that plots are being generated
            messagebox.showinfo("Generando Gráficos", 
                               "Se generarán dos gráficos:"
                               "\n1. Función de crecimiento con bandas de desviación estándar"
                               "\n2. Simulación de tráfico mensual con crecimiento compuesto")
            
            # Define default growth functions if not already defined
            if not hasattr(self, 'mean_func'):
                # Default mean function - constant growth rate
                self.mean_func = lambda x: 0.047
            
            if not hasattr(self, 'std_func'):
                # Default standard deviation function
                self.std_func = lambda x: 0.057
            
            # Create or update the parameters dictionary using centralized function
            self.dict_params = create_params_dict(self, 
                                                 include_traffic=True,
                                                 include_design=False, 
                                                 include_simulation=True,
                                                 include_flexible=False,
                                                 include_functions=True)
        
            # First plot: Growth function with standard deviation bands
            plot_simulated_function(self.dict_params)
            
            # Second plot: Monthly traffic simulation with compounded growth
            plot_simulated_transit(self.dict_params)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting simulated transit: {str(e)}")
            import traceback
            traceback.print_exc()

    def create_trad_design_widgets(self, tab):
        """
        Create widgets for the traditional design visualization tab
        """
        # Create a frame for parameters
        param_frame = ttk.LabelFrame(tab, text="Parámetros de Diseño")
        param_frame.pack(fill='x', padx=5, pady=5)

        # Add a description label
        description = "Esta visualización muestra el progreso del tráfico y cómo el SN proyectado se compara con el SN de diseño a lo largo del tiempo."
        ttk.Label(param_frame, text=description, wraplength=400).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        # Create a frame for the plot
        plot_frame = ttk.LabelFrame(tab, text="Visualización")
        plot_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Create matplotlib figure
        self.trad_fig = Figure(figsize=(8, 6))
        self.trad_canvas = FigureCanvasTkAgg(self.trad_fig, master=plot_frame)
        self.trad_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Add toolbar
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill='x')
        toolbar = NavigationToolbar2Tk(self.trad_canvas, toolbar_frame)
        toolbar.update()

        # Add plot button
        ttk.Button(param_frame, text="Generar Gráfico", 
                   command=self.plot_traditional_design).grid(row=1, column=0, 
                                                            columnspan=2, pady=10)

    def plot_traditional_design(self):
        """
        Generate and display the traditional design plot using parameters from other tabs
        """
        try:
            # Get parameters from existing entries in other tabs using centralized function
            params = create_params_dict(self, 
                                       include_traffic=True,
                                       include_design=True, 
                                       include_simulation=True,
                                       include_flexible=False,
                                       include_functions=True)

            # Clear previous plot
            self.trad_fig.clear()

            # Generate new plot
            fig, (ax1, ax2, ax3) = plot_traditional_design(params, DF)
            
            # Copy the plots to our figure
            for i, ax in enumerate([ax1, ax2, ax3]):
                self.trad_fig.add_subplot(3, 1, i+1)
                for line in ax.get_lines():
                    self.trad_fig.axes[i].plot(line.get_xdata(), line.get_data()[1], 
                                             color=line.get_color(), 
                                             linestyle=line.get_linestyle(),
                                             label=line.get_label())
                self.trad_fig.axes[i].set_xlabel(ax.get_xlabel())
                self.trad_fig.axes[i].set_ylabel(ax.get_ylabel())
                self.trad_fig.axes[i].set_title(ax.get_title())
                self.trad_fig.axes[i].grid(True)
                self.trad_fig.axes[i].legend()

            # Add shaded area for SN exceedance if it exists
            if hasattr(ax3, 'collections'):
                for collection in ax3.collections:
                    if collection.get_label() == 'SN Exceedance':
                        self.trad_fig.axes[2].fill_between(collection.get_paths()[0].vertices[:, 0],
                                                         collection.get_paths()[0].vertices[:, 1],
                                                         color='red', alpha=0.3,
                                                         label='SN Exceedance')
                        self.trad_fig.axes[2].legend()

            self.trad_fig.tight_layout()
            self.trad_canvas.draw()
            
            # Close the temporary figure
            plt.close(fig)

        except Exception as e:
            messagebox.showerror("Error", f"Error al generar el gráfico: {str(e)}")

# Import functions based on how this module is being used
try:
    # Try relative imports first (when imported as a module)
    from .Logica import solve_sn, cargar_materiales, solve, resolve, evaluate_flexibility
    from .results import plot_simulated_function, plot_simulated_transit, plot_traditional_design
except ImportError:
    # Fall back to absolute imports (when run directly)
    from Logica import solve_sn, cargar_materiales, solve, resolve, evaluate_flexibility
    from results import plot_simulated_function, plot_simulated_transit, plot_traditional_design

# Load default materials
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "default.csv")
DF = cargar_materiales(csv_path)

# Crear la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()