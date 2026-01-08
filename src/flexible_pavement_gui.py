import tkinter as tk
from tkinter import ttk
import os
import sys
import numpy as np
import pandas as pd
from tkinter import messagebox
from tkinter import filedialog
from scipy.optimize import curve_fit
import tkinter.font as tkfont
import importlib.util
import threading

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

def show_interactive_layer_modification_dialog(parent, section, current_ne_limits, expected_ne, 
                                                subgrade_E=50.0, subgrade_v=0.35, traffic_level='low'):
    """
    Show an interactive dialog for modifying layer thicknesses when a section fails.
    
    Parameters:
    -----------
    parent : tk.Toplevel or tk.Tk
        Parent window
    section : Section
        Current pavement section
    current_ne_limits : dict
        Dictionary of current NE limits for each layer/criterion
    expected_ne : float
        Expected NE requirement for the next step
    subgrade_E, subgrade_v, traffic_level : float/str
        Burmister analysis parameters
    
    Returns:
    --------
    Section or None: Modified section with new thicknesses, or None if cancelled
    """
    from copy import deepcopy
    try:
        import burmister
        from burm_FDM_Ne import calculate_pavement_ne
        from Logica import calculate_section_with_thickness_modifications
    except ImportError:
        # Try relative imports
        from . import burmister
        from .burm_FDM_Ne import calculate_pavement_ne
        from .Logica import calculate_section_with_thickness_modifications
    
    # Create dialog window
    dialog = tk.Toplevel(parent)
    dialog.title("Modificar Espesores de Capas - Sección Falló")
    dialog.geometry("800x700")
    dialog.transient(parent)
    # Use grab_set() to make dialog modal - this is necessary for wait_window() to work
    # It will only block interaction with the parent window, not the entire application
    try:
        dialog.grab_set()
    except:
        pass  # If grab_set fails, continue without it
    
    # Center the window
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
    y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
    dialog.geometry(f"+{x}+{y}")
    
    # Store result
    result = {'section': None, 'cancelled': False}
    
    # Title and info
    info_frame = ttk.Frame(dialog, padding="10")
    info_frame.pack(fill='x', padx=10, pady=5)
    
    ttk.Label(info_frame, text="La sección ha fallado. Modifique los espesores de las capas:", 
              font=('Arial', 12, 'bold')).pack()
    ttk.Label(info_frame, text=f"NE Requerido para el siguiente paso: {expected_ne:,.0f}", 
              font=('Arial', 10)).pack(pady=5)
    ttk.Label(info_frame, text="(Deje vacío para mantener el espesor actual)", 
              font=('Arial', 9), foreground='gray').pack()
    
    # Create scrollable frame for layers
    canvas_frame = ttk.Frame(dialog)
    canvas_frame.pack(fill='both', expand=True, padx=10, pady=5)
    
    canvas = tk.Canvas(canvas_frame, highlightthickness=0)
    scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Store entry widgets and layer info
    thickness_entries = []
    layer_info_labels = []
    
    # Process each layer (excluding subgrade)
    num_layers = len(section) - 1
    for i in range(num_layers):
        layer = section[i]
        current_thickness_cm = layer.thickness_cm
        
        # Get current NE for this layer from ne_limits
        current_ne = float('inf')
        layer_ne_key = None
        for key in current_ne_limits.keys():
            if f'Layer {i+1}' in key:
                if current_ne_limits[key] < current_ne:
                    current_ne = current_ne_limits[key]
                    layer_ne_key = key
        
        # Create layer frame
        layer_frame = ttk.LabelFrame(scrollable_frame, text=f"Capa {i+1}: {layer.name}", padding="10")
        layer_frame.pack(fill='x', padx=5, pady=5)
        
        # Layer visualization (simple bar representation)
        viz_frame = ttk.Frame(layer_frame)
        viz_frame.pack(fill='x', pady=5)
        
        # Create a simple visual representation
        viz_canvas = tk.Canvas(viz_frame, height=40, bg='white')
        viz_canvas.pack(fill='x', padx=10)
        
        # Draw layer bar (proportional to thickness)
        max_thickness = max([l.thickness_cm for l in section[:-1]] + [50])  # At least 50cm for scale
        bar_width = int((current_thickness_cm / max_thickness) * 300)
        viz_canvas.create_rectangle(10, 10, 10 + bar_width, 30, fill='lightblue', outline='black', width=2)
        viz_canvas.create_text(15 + bar_width//2, 20, text=f"{current_thickness_cm} cm", font=('Arial', 8))
        
        # Layer information
        info_text = f"Espesor actual: {current_thickness_cm} cm"
        if current_ne != float('inf'):
            info_text += f" | NE actual: {current_ne:,.0f}"
        else:
            info_text += " | NE actual: ∞"
        info_text += f"\nTipo: {layer.material_type} | E: {layer.E} MPa"
        
        info_label = ttk.Label(layer_frame, text=info_text, font=('Arial', 9))
        info_label.pack(pady=5)
        layer_info_labels.append(info_label)
        
        # Input frame
        input_frame = ttk.Frame(layer_frame)
        input_frame.pack(fill='x', pady=5)
        
        ttk.Label(input_frame, text="Nuevo espesor (cm):").pack(side='left', padx=5)
        thickness_entry = ttk.Entry(input_frame, width=10)
        thickness_entry.pack(side='left', padx=5)
        thickness_entries.append(thickness_entry)
        
        # Add current thickness as placeholder
        thickness_entry.insert(0, str(current_thickness_cm))
    
    # Buttons
    button_frame = ttk.Frame(dialog, padding="10")
    button_frame.pack(fill='x', padx=10, pady=5)
    
    def apply_changes():
        """Apply thickness modifications"""
        try:
            # Get thickness modifications (in cm)
            thickness_mods = []
            for i, entry in enumerate(thickness_entries):
                value = entry.get().strip()
                if value:
                    try:
                        thickness_cm = float(value)
                        if thickness_cm <= 0:
                            messagebox.showerror("Error", f"Espesor debe ser mayor que 0 para la capa {i+1}")
                            return
                        thickness_mods.append(thickness_cm)
                    except ValueError:
                        # Invalid input, keep current
                        messagebox.showerror("Error", f"Valor inválido para la capa {i+1}: {value}")
                        return
                else:
                    # Empty, keep current
                    thickness_mods.append(section[i].thickness_cm)
            
            # Calculate new section with modifications (thickness_mods are in cm)
            modified_section, new_ne, new_ne_limits = calculate_section_with_thickness_modifications(
                section, thickness_mods, subgrade_E, subgrade_v, traffic_level
            )
            
            result['section'] = modified_section
            result['cancelled'] = False
            dialog.destroy()
            
        except Exception as e:
            import traceback
            error_msg = f"Error al aplicar modificaciones:\n{str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Error", error_msg)
            print(f"Error in apply_changes: {e}")
            traceback.print_exc()
    
    def cancel():
        """Cancel and use automatic redesign"""
        result['cancelled'] = True
        dialog.destroy()
    
    ttk.Button(button_frame, text="Aplicar Cambios", command=apply_changes).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Cancelar (usar rediseño automático)", command=cancel).pack(side='left', padx=5)
    
    # Focus and lift dialog to ensure it's visible and on top
    try:
        dialog.focus_force()
        dialog.lift()
        dialog.update()
        print(f"[Interactive Dialog] Dialog created and shown, waiting for user input...")
    except Exception as e:
        print(f"[Interactive Dialog] Error focusing dialog: {e}")
    
    # Wait for dialog to close (this blocks until dialog.destroy() is called)
    dialog.wait_window()
    print(f"[Interactive Dialog] Dialog closed, cancelled: {result['cancelled']}, section: {result['section'] is not None}")
    
    if result['cancelled']:
        return None
    return result['section']

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
        self.accumulated_traffic = None
        self.burmister_result = None
        self.burmister_acumulated = None
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
            
        # Calculate flexibility or load from file using results.py functions
        base_filename = str(self.arr_ruta.get())
        base_csv_path = os.path.join(script_dir, base_filename)
        result = None
        accumulated_sn_df = None
        
        # Try to load existing results using the new function from results.py
        try:
            # Ensure script_dir is in the path
            if script_dir not in sys.path:
                sys.path.append(script_dir)
            
            # Try to import the functions we need from results.py
            results_spec = importlib.util.spec_from_file_location("results", os.path.join(script_dir, "results.py"))
            results_module = importlib.util.module_from_spec(results_spec)
            results_spec.loader.exec_module(results_module)
            
            result, accumulated_sn_df = results_module.read_flexibility_csvs(base_filename, script_dir)
            print(f"Successfully loaded existing simulation results using results.py")
            
            # Persist accumulated SN values for downstream workflows
            self.acumulated = accumulated_sn_df.values
            
            # Try to load optional traffic and Burmister datasets
            traffic_csv_path = base_csv_path + "_traffic.csv"
            burm_csv_path = base_csv_path + "_burmister.csv"
            burm_accum_csv_path = base_csv_path + "_burmister_acumulated.csv"
            
            if os.path.exists(traffic_csv_path):
                self.accumulated_traffic = pd.read_csv(traffic_csv_path).values
            else:
                self.accumulated_traffic = None
            
            if os.path.exists(burm_csv_path):
                self.burmister_result = pd.read_csv(burm_csv_path)
                self.burmister_acumulated = pd.read_csv(burm_accum_csv_path).values if os.path.exists(burm_accum_csv_path) else None
            else:
                self.burmister_result = None
                self.burmister_acumulated = None
            
        except Exception as e:
            # If files don't exist or can't be loaded, run new simulation
            print(f"Could not load existing results ({str(e)}), running new simulation...")
            
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
            
            # Run evaluate_flexibility to generate the results (without burmister)
            try:
                shared_random_esals, shared_accumulated = generate_shared_traffic(self.dict_params)
                
                result, accumulated_sn = evaluate_flexibility(
                    self.dict_params, DF,
                    random_esals=shared_random_esals,
                    acumulated=shared_accumulated
                )
                
                self.acumulated = accumulated_sn
                self.accumulated_traffic = shared_accumulated
                # Initialize burmister results as None - will be computed separately if requested
                self.burmister_result = None
                self.burmister_acumulated = None
                
                # Save results (only flexibility, not burmister)
                ruta_arr = base_csv_path + ".csv"
                traffic_csv_path = base_csv_path + "_traffic.csv"
                
                result.to_csv(ruta_arr, index=False)
                pd.DataFrame(self.acumulated).to_csv(ruta_arr + "_acumulated.csv", index=False)
                pd.DataFrame(self.accumulated_traffic).to_csv(traffic_csv_path, index=False)
                
                # Create accumulated_sn_df from the saved data
                accumulated_sn_df = pd.DataFrame(self.acumulated)
                
                print(f"New simulation completed and saved to {base_filename}")
                
            except Exception as eval_error:
                import traceback
                # Print full traceback to console for debugging
                print("=" * 80)
                print("SIMULATION ERROR - Full Traceback:")
                print("=" * 80)
                traceback.print_exc()
                print("=" * 80)
                # Also include traceback in the error message
                error_msg = f'Failed to run simulation: {str(eval_error)}\n\nFull traceback:\n{traceback.format_exc()}'
                show_copyable_message('Error', error_msg)
                return
        
        # Create new tab for results
        rand_tab = ttk.Frame(notebook)
        notebook.add(rand_tab, text='Flexible Design Results')
        
        # Create a frame for controls
        controls_frame = ttk.Frame(rand_tab)
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Simulation selector
        ttk.Label(controls_frame, text="Simulation:").pack(side=tk.LEFT, padx=5)
        self.sim_selector = ttk.Combobox(controls_frame, width=10, state="readonly")
        available_sims = result['simulation'].unique() if 'simulation' in result.columns else [0]
        self.sim_selector['values'] = list(available_sims)
        self.sim_selector.set(available_sims[0])
        self.sim_selector.pack(side=tk.LEFT, padx=5)
        
        # Display options (only for comparison plots)
        # Note: plot_sn_progression doesn't support envelope/all_sims options
        
        # Plot type selector
        ttk.Label(controls_frame, text="Plot Type:").pack(side=tk.LEFT, padx=(20, 5))
        self.plot_type_var = tk.StringVar(value="single")
        plot_type_combo = ttk.Combobox(controls_frame, textvariable=self.plot_type_var, 
                                     values=["single", "comparison"], state="readonly", width=12)
        plot_type_combo.pack(side=tk.LEFT, padx=5)
        
        # Update button
        def update_plot():
            # Clear the current canvas
            for widget in plot_frame.winfo_children():
                widget.destroy()
            
            try:
                # Ensure the import can find the results module
                if script_dir not in sys.path:
                    sys.path.append(script_dir)
                
                # Import plotting functions from results.py
                results_spec = importlib.util.spec_from_file_location("results", os.path.join(script_dir, "results.py"))
                results_module = importlib.util.module_from_spec(results_spec)
                results_spec.loader.exec_module(results_module)
                
                if self.plot_type_var.get() == "single":
                    # Single simulation plot using plot_sn_progression
                    sim_id = int(self.sim_selector.get())
                    
                    # Call plot_sn_progression with return_fig=True to get the figure
                    fig = results_module.plot_sn_progression(result, accumulated_sn_df, sim_id, return_fig=True)
                    
                else:
                    # Multiple simulations comparison
                    selected_sims = available_sims[:min(4, len(available_sims))]
                    fig, axes = results_module.plot_multiple_simulations_comparison(
                        result, accumulated_sn_df,
                        simulation_ids=selected_sims,
                        max_sims=4
                    )
                
                # Create canvas and toolbar
                canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                canvas.draw()
                
                toolbar = NavigationToolbar2Tk(canvas, plot_frame)
                toolbar.update()
                
                # Pack widgets
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                toolbar.pack(side=tk.BOTTOM, fill=tk.X)
                
            except Exception as e:
                # Fallback to simple plot if advanced plotting fails
                error_label = ttk.Label(plot_frame, text=f"Plotting error: {str(e)}")
                error_label.pack(expand=True)
                show_copyable_message("Plotting Warning", f"Advanced plotting failed: {str(e)}\nUsing fallback display.")
        
        ttk.Button(controls_frame, text="Update Plot", command=update_plot).pack(side=tk.LEFT, padx=10)
        
        # Create frame for the plot
        plot_frame = ttk.Frame(rand_tab)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initial plot
        update_plot()
        
        # Bind combobox change events to update plot
        self.sim_selector.bind('<<ComboboxSelected>>', lambda e: update_plot())
        plot_type_combo.bind('<<ComboboxSelected>>', lambda e: update_plot())
        
        # Add Traditional vs Simulated Analysis button
        def run_traditional_analysis():
            try:
                # Create or update the parameters dictionary using centralized function
                analysis_params = create_params_dict(self, 
                                                   include_traffic=True,
                                                   include_design=True, 
                                                   include_simulation=True,
                                                   include_flexible=False,
                                                   include_functions=True)
                
                # Use the accumulated traffic data from the simulations
                accumulated_source = None
                if hasattr(self, 'accumulated_traffic') and self.accumulated_traffic is not None:
                    accumulated_source = self.accumulated_traffic
                elif hasattr(self, 'acumulated') and self.acumulated is not None:
                    accumulated_source = self.acumulated
                
                if accumulated_source is None:
                    messagebox.showerror("Error", 
                                       "No accumulated traffic data available. Please run a simulation first.\n"
                                       "Go to 'Parámetros de Tráfico' tab and click 'Simular Tráfico'.")
                    return
                
                accumulated_data = np.array(accumulated_source)
                
                # Import and run the analysis
                if script_dir not in sys.path:
                    sys.path.append(script_dir)
                
                results_spec = importlib.util.spec_from_file_location("results", os.path.join(script_dir, "results.py"))
                results_module = importlib.util.module_from_spec(results_spec)
                results_spec.loader.exec_module(results_module)
                
                # Run analysis
                analysis_results = results_module.analyze_traditional_vs_simulated(
                    analysis_params, accumulated_data, return_fig=False
                )
                
                messagebox.showinfo("Analysis Complete", 
                                  f"Traditional vs Simulated Analysis completed!\n\n"
                                  f"Key Results:\n"
                                  f"• Average underestimation: {np.mean(list(analysis_results['underestimation_frequency'].values())):.1f}%\n"
                                  f"• Average overestimation: {np.mean(list(analysis_results['overestimation_frequency'].values())):.1f}%\n"
                                  f"• Average RMSE: {np.mean(list(analysis_results['rmse'].values())):.1f}%\n\n"
                                  f"See console output for detailed statistics.")
                
            except Exception as e:
                messagebox.showerror("Analysis Error", f"Error running traditional analysis:\n{str(e)}")
                import traceback
                traceback.print_exc()
        
        # Add the analysis button to controls
        analysis_btn = ttk.Button(controls_frame, text='Traditional vs Simulated Analysis', 
                                command=run_traditional_analysis)
        analysis_btn.pack(side=tk.RIGHT, padx=10)

        # Add export to Excel button
        def export_to_excel():
            try:
                file_path = filedialog.asksaveasfilename(
                    defaultextension='.xlsx',
                    filetypes=[('Excel files', '*.xlsx')],
                    title='Save Results as Excel File'
                )
                if file_path:
                    with pd.ExcelWriter(file_path) as writer:
                        result.to_excel(writer, sheet_name='Section Results', index=False)
                        accumulated_sn_df.to_excel(writer, sheet_name='Accumulated SN', index=False)
                        
                        # Add summary statistics
                        if 'simulation' in result.columns:
                            summary_stats = []
                            for sim in result['simulation'].unique():
                                sim_data = result[result['simulation'] == sim]
                                stats = {
                                    'Simulation': sim,
                                    'Total_Interventions': len(sim_data),
                                    'Total_Cost': sim_data['total_cost'].sum(),
                                    'Max_SN': sim_data['total_sn'].max(),
                                    'Complete_Rebuilds': len(sim_data[sim_data['startover'] == 0]),
                                    'Partial_Rebuilds': len(sim_data[sim_data['startover'] > 0])
                                }
                                summary_stats.append(stats)
                            
                            summary_df = pd.DataFrame(summary_stats)
                            summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
                    
                    messagebox.showinfo('Success', 'Results exported successfully!')
            except Exception as e:
                show_copyable_message('Error', f'Failed to export results: {str(e)}')
        
        export_btn = ttk.Button(controls_frame, text='Export to Excel', command=export_to_excel)
        export_btn.pack(side=tk.RIGHT, padx=10)
        
        # Add Burmister analysis button (separate from main simulation)
        def run_burmister_analysis():
            """Run Burmister-based flexibility evaluation separately with progress indication"""
            # Check if we have traffic data from the main simulation
            if not hasattr(self, 'accumulated_traffic') or self.accumulated_traffic is None:
                messagebox.showerror("Error", 
                                   "No traffic simulation data available. Please run the main simulation first.\n"
                                   "Click 'Simular Tráfico' to generate traffic data.")
                return
            
            # Reload materials to ensure we have the latest DF
            script_dir_local = os.path.dirname(os.path.abspath(__file__))
            csv_path_local = os.path.join(script_dir_local, str(self.cruta.get()) + ".csv")
            DF_local = cargar_materiales(csv_path_local)
            
            if len(DF_local) == 0:
                messagebox.showerror("Error", "No materials available for Burmister analysis")
                return
            
            # Get subgrade properties (could add UI inputs for these later)
            subgrade_E = 50.0  # Default value
            subgrade_v = 0.35  # Default value
            traffic_level = 'low'  # Default value
            
            # Ensure dict_params has all required parameters
            if not hasattr(self, 'dict_params') or not self.dict_params:
                self.dict_params = create_params_dict(self, 
                                                    include_traffic=True,
                                                    include_design=True, 
                                                    include_simulation=True,
                                                    include_flexible=True,
                                                    include_functions=True)
            else:
                # Update dict_params to ensure all required parameters are present
                updated_params = create_params_dict(self, 
                                                   include_traffic=True,
                                                   include_design=True, 
                                                   include_simulation=True,
                                                   include_flexible=True,
                                                   include_functions=True)
                self.dict_params.update(updated_params)
            
            # Check if interactive mode is enabled
            interactive_mode = hasattr(self, 'interactive_mode') and self.interactive_mode.get()
            print(f"[GUI] Interactive mode enabled: {interactive_mode}")
            
            # Create progress window
            progress_window = tk.Toplevel(self.master)
            progress_window.title("Burmister Analysis Progress")
            progress_window.geometry("500x150")
            progress_window.transient(self.master)
            # Don't use grab_set() to avoid blocking main window
            
            # Center the window
            progress_window.update_idletasks()
            x = (progress_window.winfo_screenwidth() // 2) - (progress_window.winfo_width() // 2)
            y = (progress_window.winfo_screenheight() // 2) - (progress_window.winfo_height() // 2)
            progress_window.geometry(f"+{x}+{y}")
            
            # Status label
            status_label = ttk.Label(progress_window, text="Initializing Burmister analysis...", font=('Arial', 10))
            status_label.pack(pady=10)
            
            # Progress bar (determinate mode)
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, mode='determinate', length=400, variable=progress_var, maximum=100)
            progress_bar.pack(pady=10, padx=20, fill='x')
            
            # Simulation counter label
            sim_label = ttk.Label(progress_window, text="", font=('Arial', 9))
            sim_label.pack(pady=5)
            
            # Cancel button
            cancel_button = ttk.Button(progress_window, text="Cancel", command=lambda: setattr(progress_window, 'cancelled', True))
            cancel_button.pack(pady=5)
            progress_window.cancelled = False
            
            # Store results
            burmister_results = {'result': None, 'accumulated': None, 'error': None}
            
            # Progress callback function (thread-safe)
            def update_progress(current_sim, total_sims):
                """Update progress bar and status label (called from background thread)"""
                if progress_window.winfo_exists() and not progress_window.cancelled:
                    percentage = (current_sim / total_sims) * 100
                    # Schedule GUI update on main thread (use default args to capture values correctly)
                    progress_window.after(0, lambda p=percentage, cs=current_sim, ts=total_sims: progress_var.set(p))
                    progress_window.after(0, lambda: status_label.config(text=f"Running Burmister analysis... This may take several minutes."))
                    progress_window.after(0, lambda cs=current_sim, ts=total_sims: sim_label.config(text=f"Simulation {cs} of {ts}"))
            
            def run_analysis():
                """Run the analysis in a separate thread"""
                try:
                    # Update status
                    status_label.config(text="Initializing Burmister analysis...")
                    sim_label.config(text="")
                    progress_var.set(0)
                    progress_window.update()
                    
                    # Get total number of simulations for display
                    total_sims = self.dict_params.get('size', 1)
                    
                    # Create interactive callback if interactive mode is enabled
                    interactive_callback = None
                    if interactive_mode:
                        import queue
                        import threading
                        
                        def make_interactive_callback(main_window):
                            def callback(section, current_ne_limits, expected_ne, 
                                        subgrade_E_cb, subgrade_v_cb, traffic_level_cb):
                                # This is called from background thread - need to show dialog on main thread
                                result_queue = queue.Queue()
                                dialog_shown = threading.Event()
                                
                                def show_dialog_on_main_thread():
                                    """Show dialog on main thread and put result in queue"""
                                    try:
                                        print(f"[Interactive] Showing dialog for section with {len(section)} layers")
                                        print(f"[Interactive] Expected NE: {expected_ne:,.0f}")
                                        print(f"[Interactive] Current NE limits keys: {list(current_ne_limits.keys())[:3]}...")
                                        
                                        # Ensure main window is visible and focused
                                        try:
                                            main_window.lift()
                                            main_window.focus_force()
                                        except:
                                            pass
                                        
                                        result = show_interactive_layer_modification_dialog(
                                            main_window, section, current_ne_limits, expected_ne,
                                            subgrade_E_cb, subgrade_v_cb, traffic_level_cb
                                        )
                                        print(f"[Interactive] Dialog closed, result: {result is not None}")
                                        if result is not None:
                                            print(f"[Interactive] Modified section has {len(result)} layers")
                                        result_queue.put(('success', result))
                                    except Exception as e:
                                        print(f"[Interactive] Error showing dialog: {e}")
                                        import traceback
                                        traceback.print_exc()
                                        result_queue.put(('error', None))
                                    finally:
                                        dialog_shown.set()
                                
                                # Schedule dialog on main thread
                                print("[Interactive] Scheduling dialog on main thread...")
                                main_window.after(0, show_dialog_on_main_thread)
                                
                                # Give main thread time to show the dialog
                                import time
                                time.sleep(0.2)  # Small delay to let dialog appear
                                
                                # Wait for result from queue (with timeout)
                                timeout = 300  # 5 minutes max
                                start = time.time()
                                
                                while (time.time() - start) < timeout:
                                    try:
                                        status, result = result_queue.get_nowait()
                                        print(f"[Interactive] Got result from queue: {status}")
                                        return result
                                    except queue.Empty:
                                        # Process events to allow dialog to work
                                        try:
                                            main_window.update_idletasks()
                                        except:
                                            pass
                                        time.sleep(0.1)  # Small delay
                                
                                # Timeout - return None to use automatic redesign
                                print("[Interactive] Dialog timeout - using automatic redesign")
                                return None
                            return callback
                        interactive_callback = make_interactive_callback(self.master)
                    
                    # Run Burmister evaluation using the same traffic data with progress callback
                    burmister_result, burmister_acumulated = evaluate_flexibility_burmister(
                        self.dict_params, DF_local,
                        random_esals=None,  # Will use shared traffic
                        acumulated=self.accumulated_traffic,
                        subgrade_E=subgrade_E,
                        subgrade_v=subgrade_v,
                        traffic_level=traffic_level,
                        progress_callback=update_progress,
                        interactive_callback=interactive_callback
                    )
                    
                    if not progress_window.cancelled:
                        burmister_results['result'] = burmister_result
                        burmister_results['accumulated'] = burmister_acumulated
                        
                        # Update status - show 100% completion
                        total_sims = self.dict_params.get('size', 1)
                        progress_window.after(0, lambda: progress_var.set(100))
                        progress_window.after(0, lambda: status_label.config(text="Saving results..."))
                        progress_window.after(0, lambda: sim_label.config(text=f"Completed {total_sims} of {total_sims} simulations"))
                        progress_window.update()
                        
                        # Save Burmister results (use base_csv_path from outer scope)
                        burm_csv_path = base_csv_path + "_burmister.csv"
                        burm_accum_csv_path = base_csv_path + "_burmister_acumulated.csv"
                        
                        if isinstance(burmister_result, pd.DataFrame):
                            burmister_result.to_csv(burm_csv_path, index=False)
                        if burmister_acumulated is not None:
                            pd.DataFrame(burmister_acumulated).to_csv(burm_accum_csv_path, index=False)
                        
                        progress_window.after(0, lambda: status_label.config(text="Analysis completed successfully!"))
                        progress_window.after(0, lambda: sim_label.config(text=""))
                        progress_window.update()
                        
                except Exception as e:
                    if not progress_window.cancelled:
                        burmister_results['error'] = e
                        import traceback
                        burmister_results['traceback'] = traceback.format_exc()
                        # Print full traceback to console for debugging
                        print("=" * 80)
                        print("BURMISTER ANALYSIS ERROR - Full Traceback:")
                        print("=" * 80)
                        traceback.print_exc()
                        print("=" * 80)
                finally:
                    # Close progress window after a short delay
                    progress_window.after(500, lambda: progress_window.destroy())
            
            def check_completion():
                """Check if analysis is complete and handle results"""
                if burmister_results['error'] is not None:
                    # Error occurred
                    error = burmister_results['error']
                    tb = burmister_results.get('traceback', '')
                    error_msg = f'Failed to run Burmister analysis: {str(error)}\n\nFull traceback:\n{tb}'
                    show_copyable_message('Error', error_msg)
                elif burmister_results['result'] is not None:
                    # Success
                    self.burmister_result = burmister_results['result']
                    self.burmister_acumulated = burmister_results['accumulated']
                    
                    burm_csv_path = base_csv_path + "_burmister.csv"
                    burm_accum_csv_path = base_csv_path + "_burmister_acumulated.csv"
                    
                    messagebox.showinfo("Success", 
                                      "Burmister analysis completed successfully!\n"
                                      f"Results saved to:\n{burm_csv_path}\n{burm_accum_csv_path}")
                elif progress_window.winfo_exists():
                    # Still running, check again
                    progress_window.after(100, check_completion)
            
            # Start analysis in separate thread
            analysis_thread = threading.Thread(target=run_analysis, daemon=True)
            analysis_thread.start()
            
            # Start checking for completion
            progress_window.after(100, check_completion)
            
            # Handle window close
            def on_close():
                if analysis_thread.is_alive():
                    progress_window.cancelled = True
                    messagebox.showwarning("Analysis Running", 
                                         "Analysis is still running. It will continue in the background.\n"
                                         "Results will be saved when complete.")
                progress_window.destroy()
            
            progress_window.protocol("WM_DELETE_WINDOW", on_close)
        
        burmister_btn = ttk.Button(controls_frame, text='Run Burmister Analysis', 
                                  command=run_burmister_analysis)
        burmister_btn.pack(side=tk.RIGHT, padx=10)
            
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
        
        # Container with horizontal scrolling for the materials table
        self.table_container = ttk.Frame(tab)
        self.table_container.grid(row=1, column=0, columnspan=12, padx=5, pady=5, sticky="nsew")
        
        # Configure grid weights so the canvas expands
        try:
            tab.columnconfigure(0, weight=1)
            self.table_container.columnconfigure(0, weight=1)
            self.table_container.rowconfigure(0, weight=1)
        except Exception:
            pass
        
        self.mat_canvas = tk.Canvas(self.table_container, highlightthickness=0)
        self.mat_canvas.grid(row=0, column=0, sticky="nsew")
        
        self.mat_h_scroll = ttk.Scrollbar(self.table_container, orient='horizontal', command=self.mat_canvas.xview)
        self.mat_h_scroll.grid(row=1, column=0, sticky="ew")
        self.mat_canvas.configure(xscrollcommand=self.mat_h_scroll.set)
        
        # Inner frame that will hold the table
        self.mat_table_frame = ttk.Frame(self.mat_canvas)
        self.mat_canvas.create_window((0, 0), window=self.mat_table_frame, anchor='nw')
        
        # Update scrollregion whenever the size of the inner frame changes
        def _update_scrollregion(event=None):
            try:
                self.mat_canvas.configure(scrollregion=self.mat_canvas.bbox("all"))
            except Exception:
                pass
        self.mat_table_frame.bind('<Configure>', _update_scrollregion)
        
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
        # Clear existing table contents inside the scrollable frame
        if hasattr(self, 'mat_table_frame') and self.mat_table_frame:
            for widget in self.mat_table_frame.grid_slaves():
                widget.destroy()
        
        # Mostrar el resultado de cargar el material en la interfaz
        DF_mat = cargar_materiales(ruta)
        
        # If the DataFrame is empty (new file), create a template with columns from default.csv
        if len(DF_mat) == 0:
            self.mat_result_label.config(text="Created new material file template")
            # Load headers from default.csv to ensure all columns are shown
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_csv_path = os.path.join(script_dir, "default.csv")
            try:
                default_headers = list(pd.read_csv(default_csv_path, nrows=0).columns)
            except Exception:
                # Fallback to previous minimal set if default.csv is not readable
                default_headers = ['mat_name','SN','min','max','density','cost','unit','surface','subgrade','alkaline']
            DF_mat = pd.DataFrame(columns=default_headers)
        
        # Re-add title labels dynamically from DataFrame columns
        titulos = list(DF_mat.columns)
        for i in range(len(titulos)):
            ttk.Label(self.mat_table_frame, text=titulos[i]).grid(row=0, column=i, padx=5, pady=5, sticky="w")
            
        # Add material data
        for i in range(len(DF_mat)):
            for j in range(len(DF_mat.iloc[i])):
                ttk.Label(self.mat_table_frame, text=str(DF_mat.iloc[i].iloc[j])).grid(row=i+1, column=j, padx=5, pady=1, sticky="w")
        
        # Calculate the row for new entries (after the last material)
        entry_row = len(DF_mat) + 1
        
        # Define options for dropdowns
        unit_options = ['ton', 'cyd', 'sqyd']
        bool_options = ['False', 'True']
        
        # Create or update entry fields for new material across all columns
        self.entrys = []
        for k in range(len(titulos)):
            col_name = titulos[k]
            if col_name == 'unit':
                entry = ttk.Combobox(self.mat_table_frame, values=unit_options, width=10, state='readonly')
                entry.set(unit_options[0])
            elif col_name in ['surface', 'subgrade', 'alkaline']:
                entry = ttk.Combobox(self.mat_table_frame, values=bool_options, width=10, state='readonly')
                entry.set(bool_options[0])
            else:
                entry = ttk.Entry(self.mat_table_frame, width=12)
            entry.grid(row=entry_row, column=k, padx=5, pady=5, sticky="w")
            self.entrys.append(entry)
                
        # Add save button if it doesn't exist
        if not hasattr(self, 'save_button') or not self.save_button:
            self.save_button = ttk.Button(tab, text="Guardar material", command=self.crear_materiales)
            self.save_button.grid(row=0, column=5, columnspan=2, pady=10)
            
        # Add additional configuration fields
        # Place configuration section below the scrollable table
        config_start_row = 2
        
        # Header for configuration section
        ttk.Label(tab, text="Configuration", font=('Arial', 10, 'bold')).grid(row=config_start_row, column=0, columnspan=2, pady=(10,5), sticky="w")
        
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
        
        # Interactive mode checkbox
        self.interactive_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(interval_frame, text="Modo Interactivo (pausar en fallos para modificar espesores)", 
                       variable=self.interactive_mode).grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="w")

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
        # Create a frame for analysis options
        options_frame = ttk.LabelFrame(tab, text="Opciones de Análisis")
        options_frame.pack(fill='x', padx=5, pady=5)

        # Analysis type selector
        ttk.Label(options_frame, text="Tipo de Análisis:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.analysis_type = tk.StringVar(value="traditional_design")
        analysis_combo = ttk.Combobox(options_frame, textvariable=self.analysis_type, 
                                     values=["traditional_design", "traditional_vs_simulated", "npv_analysis"],
                                     state="readonly", width=25)
        analysis_combo.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        analysis_combo.bind('<<ComboboxSelected>>', self.on_analysis_type_change)

        # NPV parameters frame (initially hidden)
        self.npv_frame = ttk.LabelFrame(tab, text="Parámetros NPV")
        self.npv_frame.pack(fill='x', padx=5, pady=5)
        
        # Discount rate
        ttk.Label(self.npv_frame, text="Tasa de Descuento (%):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.discount_rate = tk.StringVar(value="5.0")
        ttk.Entry(self.npv_frame, textvariable=self.discount_rate, width=10).grid(row=0, column=1, padx=5, pady=5, sticky='w')

        # Simulation parameters frame (initially hidden)
        self.sim_params_frame = ttk.LabelFrame(tab, text="Parámetros de Simulación")
        self.sim_params_frame.pack(fill='x', padx=5, pady=5)
        
        # Number of simulations
        ttk.Label(self.sim_params_frame, text="Número de Simulaciones:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.num_simulations = tk.StringVar(value="100")
        ttk.Entry(self.sim_params_frame, textvariable=self.num_simulations, width=10).grid(row=0, column=1, padx=5, pady=5, sticky='w')

        # Create a frame for parameters
        param_frame = ttk.LabelFrame(tab, text="Parámetros de Diseño")
        param_frame.pack(fill='x', padx=5, pady=5)

        # Add a description label
        self.description_label = ttk.Label(param_frame, text="Esta visualización muestra cómo el SN proyectado se compara con el SN de diseño a lo largo del tiempo.", wraplength=400)
        self.description_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        # Create a frame for the plot
        plot_frame = ttk.LabelFrame(tab, text="Visualización")
        plot_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Create matplotlib figure
        self.trad_fig = Figure(figsize=(10, 8))
        self.trad_canvas = FigureCanvasTkAgg(self.trad_fig, master=plot_frame)
        self.trad_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Add toolbar
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill='x')
        toolbar = NavigationToolbar2Tk(self.trad_canvas, toolbar_frame)
        toolbar.update()

        # Add plot button
        ttk.Button(param_frame, text="Generar Análisis", 
                   command=self.run_traditional_analysis).grid(row=1, column=0, 
                                                            columnspan=2, pady=10)

        # Initially hide optional frames
        self.npv_frame.pack_forget()
        self.sim_params_frame.pack_forget()

    def on_analysis_type_change(self, event):
        """Handle analysis type selection changes"""
        analysis_type = self.analysis_type.get()
        
        # Update description based on analysis type
        if analysis_type == "traditional_design":
            self.description_label.config(text="Esta visualización muestra cómo el SN proyectado se compara con el SN de diseño a lo largo del tiempo.")
            self.npv_frame.pack_forget()
            self.sim_params_frame.pack_forget()
        elif analysis_type == "traditional_vs_simulated":
            self.description_label.config(text="Análisis comparativo entre el diseño tradicional y simulaciones de tráfico. Muestra la frecuencia de subestimación vs sobreestimación.")
            self.npv_frame.pack_forget()
            self.sim_params_frame.pack()
        elif analysis_type == "npv_analysis":
            self.description_label.config(text="Análisis del Valor Presente Neto (NPV) de los costos de construcción para diferentes tasas de descuento.")
            self.npv_frame.pack()
            self.sim_params_frame.pack_forget()

    def run_traditional_analysis(self):
        """
        Run the selected traditional analysis type
        """
        try:
            analysis_type = self.analysis_type.get()
            
            if analysis_type == "traditional_design":
                self.plot_traditional_design()
            elif analysis_type == "traditional_vs_simulated":
                self.run_traditional_vs_simulated_analysis()
            elif analysis_type == "npv_analysis":
                self.run_npv_analysis()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al ejecutar el análisis: {str(e)}")
            import traceback
            traceback.print_exc()

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
            fig, ax = plot_traditional_design(params, DF)
            
            # Copy the plot to our figure
            self.trad_fig.add_subplot(1, 1, 1)
            for line in ax.get_lines():
                self.trad_fig.axes[0].plot(line.get_xdata(), line.get_data()[1], 
                                         color=line.get_color(), 
                                         linestyle=line.get_linestyle(),
                                         label=line.get_label())
            self.trad_fig.axes[0].set_xlabel(ax.get_xlabel())
            self.trad_fig.axes[0].set_ylabel(ax.get_ylabel())
            self.trad_fig.axes[0].set_title(ax.get_title())
            self.trad_fig.axes[0].grid(True)
            self.trad_fig.axes[0].legend()

            # Add shaded area for SN exceedance if it exists
            if hasattr(ax, 'collections'):
                for collection in ax.collections:
                    if collection.get_label() == 'SN Exceedance':
                        self.trad_fig.axes[0].fill_between(collection.get_paths()[0].vertices[:, 0],
                                                         collection.get_paths()[0].vertices[:, 1],
                                                         color='red', alpha=0.3,
                                                         label='SN Exceedance')
                        self.trad_fig.axes[0].legend()

            self.trad_fig.tight_layout()
            self.trad_canvas.draw()
            
            # Close the temporary figure
            plt.close(fig)

        except Exception as e:
            messagebox.showerror("Error", f"Error al generar el gráfico: {str(e)}")

    def run_traditional_vs_simulated_analysis(self):
        """
        Run the traditional vs simulated traffic analysis
        """
        try:
            # Get parameters
            params = create_params_dict(self, 
                                       include_traffic=True,
                                       include_design=True, 
                                       include_simulation=True,
                                       include_flexible=False,
                                       include_functions=True)
            
            # Get number of simulations
            num_sims = int(self.num_simulations.get())
            params['size'] = num_sims
            
            # Generate simulated traffic data
            accumulated_traffic_data = self.generate_simulated_traffic_data(params)
            
            # Run analysis
            results = analyze_traditional_vs_simulated(params, accumulated_traffic_data, return_fig=True)
            
            # Display results in GUI
            self.display_analysis_results(results['figure'])
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en análisis tradicional vs simulado: {str(e)}")
            import traceback
            traceback.print_exc()

    def run_npv_analysis(self):
        """
        Run the NPV analysis for traditional design
        """
        try:
            # Get discount rate
            discount_rate = float(self.discount_rate.get()) / 100.0
            
            # For traditional design, we need to create a simple cost structure
            # This is a simplified approach - in practice, you'd need more detailed cost data
            messagebox.showinfo("NPV Analysis", 
                               "Para el análisis NPV del diseño tradicional, se necesitan datos de costos detallados.\n"
                               "Esta funcionalidad requiere implementación adicional de costos por período.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en análisis NPV: {str(e)}")
            import traceback
            traceback.print_exc()

    def generate_simulated_traffic_data(self, params):
        """
        Generate simulated traffic data for analysis
        """
        import numpy as np
        
        # Set up random number generator
        rng = np.random.default_rng(seed=params.get('seedint', 42))
        
        # Generate growth rates
        grow_rates = np.zeros((params['size'], params['n']))
        for i in range(params['size']):
            for j in range(params['n']):
                grow_rates[i,j] = rng.normal(loc=params['mu_function'](j), 
                                           scale=params['sigma_function'](j))
        
        # Calculate initial monthly trips
        initial_monthly_trips = params['TPD'] * 365 / 12 * params['vc'] * params['cd']
        
        # Initialize arrays for monthly traffic and cumulative traffic
        res = np.zeros((params['size'], params['n']))
        cum_res = np.zeros((params['size'], params['n']))
        
        # Apply the growth formula and calculate cumulative traffic
        for sim in range(params['size']):
            res[sim, 0] = initial_monthly_trips
            cum_res[sim, 0] = initial_monthly_trips
            for month in range(1, params['n']):
                # Apply compounded growth
                res[sim, month] = initial_monthly_trips * (1 + grow_rates[sim, month])
                # Calculate cumulative traffic
                cum_res[sim, month] = cum_res[sim, month-1] + res[sim, month]
        
        return cum_res

    def display_analysis_results(self, fig):
        """
        Display analysis results in the GUI
        """
        try:
            # Clear previous plot
            self.trad_fig.clear()
            
            # Copy all subplots from the results figure
            for i, ax in enumerate(fig.axes):
                if i == 0:
                    self.trad_fig.add_subplot(2, 2, i+1)
                else:
                    self.trad_fig.add_subplot(2, 2, i+1)
                
                # Copy lines
                for line in ax.get_lines():
                    self.trad_fig.axes[i].plot(line.get_xdata(), line.get_data()[1], 
                                             color=line.get_color(), 
                                             linestyle=line.get_linestyle(),
                                             linewidth=line.get_linewidth(),
                                             label=line.get_label())
                
                # Copy other elements
                self.trad_fig.axes[i].set_xlabel(ax.get_xlabel())
                self.trad_fig.axes[i].set_ylabel(ax.get_ylabel())
                self.trad_fig.axes[i].set_title(ax.get_title())
                self.trad_fig.axes[i].grid(True, alpha=0.3)
                self.trad_fig.axes[i].legend()
                
                # Copy collections (filled areas, histograms, etc.)
                if hasattr(ax, 'collections'):
                    for collection in ax.collections:
                        if hasattr(collection, 'get_paths') and collection.get_paths():
                            path = collection.get_paths()[0]
                            vertices = path.vertices
                            if len(vertices) > 0:
                                self.trad_fig.axes[i].fill_between(vertices[:, 0], vertices[:, 1],
                                                                 color=collection.get_facecolor(),
                                                                 alpha=collection.get_alpha(),
                                                                 label=collection.get_label())
                
                # Copy patches (bars, etc.)
                if hasattr(ax, 'patches'):
                    for patch in ax.patches:
                        self.trad_fig.axes[i].add_patch(patch)
            
            self.trad_fig.tight_layout()
            self.trad_canvas.draw()
            
            # Close the temporary figure
            plt.close(fig)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar resultados: {str(e)}")
            import traceback
            traceback.print_exc()

# Import functions based on how this module is being used
try:
    # Try relative imports first (when imported as a module)
    from .Logica import (solve_sn, cargar_materiales, solve, resolve, 
                         evaluate_flexibility, evaluate_flexibility_burmister,
                         generate_shared_traffic)
    from .results import (plot_simulated_function, plot_simulated_transit, plot_traditional_design,
                         analyze_traditional_vs_simulated, calculate_construction_npv, 
                         plot_construction_npv_analysis)
except ImportError:
    # Fall back to absolute imports (when run directly)
    from Logica import (solve_sn, cargar_materiales, solve, resolve, 
                        evaluate_flexibility, evaluate_flexibility_burmister,
                        generate_shared_traffic)
    from results import (plot_simulated_function, plot_simulated_transit, plot_traditional_design,
                        analyze_traditional_vs_simulated, calculate_construction_npv, 
                        plot_construction_npv_analysis)

# Load default materials
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "default.csv")
DF = cargar_materiales(csv_path)

# Crear la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()