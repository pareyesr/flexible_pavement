import tkinter as tk
from tkinter import ttk
import os
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from tkinter import messagebox
import matplotlib.pyplot as plt
from tkinter import filedialog
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
        self.data_series = None
        
        self.create_widgets()

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
        notebook.add(sol_tab, text='Capas solución')
        self.create_sol_widgets(sol_tab)
        # Create the tab for re-solution.
        resol_tab = ttk.Frame(notebook)
        notebook.add(resol_tab, text='Modificar capas')
        self.create_resol_widgets(resol_tab)
        # Create the tab for the parameters for the random transit evaluation
        rev_param_tab = ttk.Frame(notebook)
        notebook.add(rev_param_tab, text='Parametros evaluación por transito aleatorio')
        self.create_param_rand_widgets(rev_param_tab,notebook)
        # Create the tab for dynamic graph visualization
        graph_tab = ttk.Frame(notebook)
        notebook.add(graph_tab, text='Visualización Función')
        self.create_graph_widgets(graph_tab)

    def create_param_rand_widgets(self,tab,notebook):
        # Etiquetas y cajas de entrada para recalcular capas
        ttk.Label(tab, text="TPD:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.tpd = ttk.Entry(tab)
        self.tpd.insert(0, "402.39")
        self.tpd.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="archivo npy:").grid(row=0, column=3, padx=10, pady=5, sticky="w")
        self.arr_ruta = ttk.Entry(tab)
        self.arr_ruta.insert(0, "datos_res")
        self.arr_ruta.grid(row=0, column=4, padx=10, pady=5, sticky="w") 
        ttk.Label(tab, text="vc:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.vc = ttk.Entry(tab)
        self.vc.insert(0, "0.5")
        self.vc.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="cd:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.cd = ttk.Entry(tab)
        self.cd.insert(0, "1.0")
        self.cd.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="size:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.size = ttk.Entry(tab)
        self.size.insert(0, "5000")
        self.size.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="n:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.n = ttk.Entry(tab)
        self.n.insert(0, "360")
        self.n.grid(row=4, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="rate:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.rate = ttk.Entry(tab)
        self.rate.insert(0, "0.05")
        self.rate.grid(row=5, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="cost_rb:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.cost_rb = ttk.Entry(tab)
        self.cost_rb.insert(0, "1000")
        self.cost_rb.grid(row=6, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="capas:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.capas = ttk.Entry(tab)
        self.capas.insert(0, "2")
        self.capas.grid(row=7, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="step:").grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.step = ttk.Entry(tab)
        self.step.insert(0, "3")
        self.step.grid(row=8, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="seedint:").grid(row=9, column=0, padx=10, pady=5, sticky="w")
        self.seedint = ttk.Entry(tab)
        self.seedint.insert(0, "63442967")
        self.seedint.grid(row=9, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Annual mean growth rate:").grid(row=10, column=0, padx=10, pady=5, sticky="w")
        self.mu_annual = ttk.Entry(tab)
        self.mu_annual.insert(0, "0.047")
        self.mu_annual.grid(row=10, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Annual standard deviation of growth rate:").grid(row=11, column=0, padx=10, pady=5, sticky="w")
        self.sigma_annual = ttk.Entry(tab)
        self.sigma_annual.insert(0, "0.057")
        self.sigma_annual.grid(row=11, column=1, padx=10, pady=5, sticky="w")
        
        # Add Load button
        ttk.Button(tab, text="Load", command=lambda: self.create_rand_graph_widgets(notebook)).grid(row=0, column=5, columnspan=2, pady=10)
        
    def create_rand_graph_widgets(self,notebook):
        """Create random graph widgets with current parameters"""
        # Get current parameters
        tpd = float(self.tpd.get() or 402.39)
        vc = float(self.vc.get() or 0.5)
        cd = float(self.cd.get() or 1.0)
        size = int(self.size.get() or 5000)
        n = int(self.n.get() or 360)
        rate = float(self.rate.get() or 0.05)
        cost_rb = float(self.cost_rb.get() or 1000)
        capas = int(self.capas.get() or 2)
        step = int(self.step.get() or 3)
        seedint = int(self.seedint.get() or 63442967)
        mu_annual = float(self.mu_annual.get() or 0.047)
        sigma_annual = float(self.sigma_annual.get() or 0.057)
            
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
                
        # Use first solution by default
        sect = solutions[0]
            
        # Calculate flexibility or load from file
        ruta_arr = os.path.join(script_dir, str(self.arr_ruta.get()) + ".npy")
        result = None
        if os.path.exists(ruta_arr):
            try:
                result = np.load(ruta_arr)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {ruta_arr}: {str(e)}")
        
        if result is None:
            result = evaluate_flexibility(tpd, vc, cd, size, n, rate, self.calcular_sn(), 
                                       0.9, 0.45, 2.0, 3000, DF, sect, 
                                       float(self.grade.get() or 0.0),
                                       float(self.emb.get() or 0.0),
                                       float(self.exc.get() or 0.0),
                                       cost_rb, capas, step, seedint,mu_annual,sigma_annual)
            # Save results
            try:
                np.save(ruta_arr, result)
            except Exception as e:
                messagebox.showwarning("Warning", f"Failed to save results to {ruta_arr}: {str(e)}")
                                       
        # Create new tab for results
        rand_tab = ttk.Frame(notebook)
        notebook.add(rand_tab, text='Random Results')
            
        # Plot results
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
            
        # Create histogram with labels
        counts, bins, patches = ax.hist(result, bins=30, density=True, alpha=0.75)
        total=sum(counts)
        # Add value labels on top of each bar
        for i in range(len(patches)):
            # Get x coordinate of the bar center
            x = (bins[i] + bins[i+1])/2
            # Get height of the bar
            height = counts[i]
            # Add text label
            ax.text(x, height, f'{height/total:.2%}', 
                   ha='center', va='bottom', rotation=0,
                   fontsize=8)
            
        ax.set_title('NPV Distribution')
        ax.set_xlabel('Net Present Value ($)')
        ax.set_ylabel('Relative Frequency')
            
        # Format x-axis with thousand separator
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
            
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
                # Create DataFrame with results
                df = pd.DataFrame(result, columns=['NPV'])
                
                # Ask user for save location
                from tkinter import filedialog
                file_path = filedialog.asksaveasfilename(
                    defaultextension='.xlsx',
                    filetypes=[('Excel files', '*.xlsx')],
                    title='Save Results as Excel File'
                )
                
                if file_path:
                    # Save to Excel
                    df.to_excel(file_path, index=False, sheet_name='NPV Distribution')
                    messagebox.showinfo('Success', 'Results exported successfully!')
            except Exception as e:
                messagebox.showerror('Error', f'Failed to export results: {str(e)}')
        
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
            return None
                
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
            return None
                
        if n_sect >= len(all_solutions):
            self.resol_result_label.config(text=f"Solution {n_sect} not available. Maximum is {len(all_solutions)-1}")
            return None
                
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
            
        # Calculate solutions
        combined_data = self.calcular_sol()
        
        if not combined_data:  # If no solutions available
            ttk.Label(tab, text="No solutions available. Please add materials first.").grid(
                row=0, column=0, columnspan=4, padx=10, pady=20)
            return
            
        # Create headers
        headers = ["Material", "Thickness", "Cost", "SN"]
        for i, header in enumerate(headers):
            ttk.Label(tab, text=header, font=('Arial', 10, 'bold')).grid(row=0, column=i, padx=10, pady=5)
            
        # Calculate base row for each solution
        current_row = 1
            
        # Display solutions
        for i, solution in enumerate(combined_data, 1):
            # Solution header with spacing
            ttk.Label(tab, text=f"Solution {i}", font=('Arial', 10, 'bold')).grid(
                row=current_row, column=0, columnspan=4, pady=(20,5))
            current_row += 1
            
            # Iterate directly over the Section object
            for layer in solution:
                # Display layer information
                ttk.Label(tab, text=layer.name).grid(row=current_row, column=0, padx=10, pady=2, sticky='w')
                ttk.Label(tab, text=f"{layer.thickness:.2f}").grid(row=current_row, column=1, padx=10, pady=2)
                ttk.Label(tab, text=f"{layer.cost:.2f}").grid(row=current_row, column=2, padx=10, pady=2)
                ttk.Label(tab, text=f"{layer.sn:.2f}").grid(row=current_row, column=3, padx=10, pady=2)
                current_row += 1
            
            # Add total cost with a separator line above
            separator = ttk.Frame(tab, height=2, relief="groove")
            separator.grid(row=current_row, column=0, columnspan=4, sticky='ew', pady=(5,5))
            current_row += 1
            
            ttk.Label(tab, text=f"Total Cost: {solution.totalCost:.2f}", font=('Arial', 10, 'bold')).grid(
                row=current_row, column=0, columnspan=4, pady=(0,10))
            current_row += 1
            
        # Botón para calcular la solución de capas - place at the top
        ttk.Button(tab, text="Calcular nuevas capas", command=self.calcular_sol).grid(
            row=0, column=4, padx=20, pady=5, sticky='ne')
            
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
        return sn

    def create_graph_widgets(self, parent):
        # Create parameter frame
        param_frame = ttk.LabelFrame(parent, text="Parámetros", padding="10")
        param_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Initialize parameters
        self.global_params = {
            'x_start': tk.DoubleVar(value=0.0),
            'x_end': tk.DoubleVar(value=12.0),
            'noise': tk.DoubleVar(value=0.0)
        }

        self.function_params = {
            'sine': {
                'amplitude': tk.DoubleVar(value=1.0),
                'frequency': tk.DoubleVar(value=0.1),
                'phase': tk.DoubleVar(value=0.0)
            },
            'linear': {
                'slope': tk.DoubleVar(value=1.0),
                'intercept': tk.DoubleVar(value=0.0)
            },
            'logarithmic': {
                'coefficient': tk.DoubleVar(value=1.0),
                'base': tk.DoubleVar(value=10.0),
                'constant': tk.DoubleVar(value=0.0),
                'shift': tk.DoubleVar(value=0.0)  # New parameter for horizontal shift
            }
        }

        # Add traces to all parameters
        for var in self.global_params.values():
            var.trace_add("write", lambda *args: self.update_graph())
        
        for func in self.function_params:
            for var in self.function_params[func].values():
                var.trace_add("write", lambda *args: self.update_graph())

        # Function selection listbox
        ttk.Label(param_frame, text="Funciones").grid(row=0, column=0, padx=5, pady=5)
        self.function_listbox = tk.Listbox(param_frame, selectmode=tk.MULTIPLE, height=3)
        self.function_listbox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        for func in ['sine', 'linear', 'logarithmic']:
            self.function_listbox.insert(tk.END, func)
        self.function_listbox.bind('<<ListboxSelect>>', self.on_function_select)
        self.selected_functions = []

        # X-range controls
        ttk.Label(param_frame, text="X Inicial").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(param_frame, textvariable=self.global_params['x_start']).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(param_frame, text="X Final").grid(row=2, column=0, padx=5, pady=5)
        ttk.Entry(param_frame, textvariable=self.global_params['x_end']).grid(row=2, column=1, padx=5, pady=5)

        # Dynamic parameters container
        self.func_params_container = ttk.Frame(param_frame)
        self.func_params_container.grid(row=3, column=0, columnspan=2, sticky="nsew")

        # Noise controls
        ttk.Label(param_frame, text="Nivel de Ruido").grid(row=4, column=0, padx=5, pady=5)
        ttk.Entry(param_frame, textvariable=self.global_params['noise']).grid(row=4, column=1, padx=5, pady=5)
        ttk.Separator(param_frame, orient='horizontal').grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)
        
        #Cargar datos
        ttk.Button(param_frame, text="Cargar Datos", command=self.load_data_series).grid(row=6, column=0, padx=5, pady=5)
        self.data_label = ttk.Label(param_frame, text="Archivo: Ninguno")
        self.data_label.grid(row=6, column=1, padx=5, pady=5)
        # Create graph frame and canvas (same as before)
        graph_frame = ttk.LabelFrame(parent, text="Gráfico", padding="10")
        graph_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.graph_fig = Figure(figsize=(6, 4), dpi=100)
        self.graph_ax = self.graph_fig.add_subplot(111)
        self.graph_canvas = FigureCanvasTkAgg(self.graph_fig, master=graph_frame)
        self.graph_canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar_frame = ttk.Frame(graph_frame)
        toolbar_frame.pack(fill='x')
        self.graph_toolbar = NavigationToolbar2Tk(self.graph_canvas, toolbar_frame)
        self.graph_toolbar.update()

        # Initial update
        self.update_parameter_ui()
        self.update_graph()

    def on_function_select(self, event):
        selected_indices = self.function_listbox.curselection()
        self.selected_functions = [self.function_listbox.get(i) for i in selected_indices]
        self.update_parameter_ui()
        self.update_graph()
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

    def update_parameter_ui(self):
        # Clear existing widgets
        for widget in self.func_params_container.winfo_children():
            widget.destroy()
        
        current_row = 0
        for func in self.selected_functions:
            # Function header
            ttk.Label(self.func_params_container, 
                    text=f"{func.capitalize()} Parámetros").grid(row=current_row, column=0, columnspan=2, sticky="w")
            current_row += 1
            
            # Parameters
            params = self.function_params[func]
            for param_name, var in params.items():
                ttk.Label(self.func_params_container, text=param_name.capitalize()).grid(
                    row=current_row, column=0, padx=5, pady=2, sticky="e")
                ttk.Entry(self.func_params_container, textvariable=var).grid(
                    row=current_row, column=1, padx=5, pady=2, sticky="w")
                current_row += 1
            
            # Separator
            ttk.Separator(self.func_params_container, orient='horizontal').grid(
                row=current_row, column=0, columnspan=2, sticky="ew", pady=5)
            current_row += 1

    def update_graph(self):
        try:
            x_start = self.global_params['x_start'].get()
            x_end = self.global_params['x_end'].get()
            noise_level = self.global_params['noise'].get()
            x = np.linspace(x_start, x_end, 1000)
            y_total = np.zeros_like(x)

            # Clear previous plot
            self.graph_ax.clear()
            
            # Plot data series first (as bars)
            if self.data_series is not None:
                x_data = self.data_series[:, 0]
                y_data = self.data_series[:, 1]
                self.graph_ax.bar(x_data, y_data, alpha=0.3, 
                                color='gray', width=0.8, 
                                label='Datos Experimentales', 
                                edgecolor='black')

            # Calculate and plot combined function
            for func in self.selected_functions:
                params = self.function_params[func]
                if func == 'sine':
                    y = params['amplitude'].get() * np.sin(
                        params['frequency'].get() * x + 
                        params['phase'].get()
                    )
                elif func == 'linear':
                    y = (params['slope'].get() * x + 
                        params['intercept'].get())
                elif func == 'logarithmic':
                    shift = params['shift'].get()
                    base = params['base'].get()
                    # Ensure input to log is positive
                    valid_x = x + shift
                    valid_x[valid_x <= 0] = 1e-9  # Avoid log(0)
                    y = (params['coefficient'].get() * np.log(valid_x)/np.log(base) + 
                        params['constant'].get())
                y_total += y

            # Add noise and plot function line
            if noise_level > 0:
                y_total += np.random.normal(0, noise_level, x.shape)
            
            self.graph_ax.plot(x, y_total, linewidth=2, color='red', 
                            label='Función Combinada')

            # Configure plot aesthetics
            self.graph_ax.set_title("Comparación de Función y Datos")
            self.graph_ax.set_xlabel("Variable Independiente (X)")
            self.graph_ax.set_ylabel("Variable Dependiente (Y)")
            self.graph_ax.grid(True, linestyle='--', alpha=0.7)
            self.graph_ax.legend(loc='upper right')
            
            # Adjust x-axis limits if data exists
            if self.data_series is not None:
                data_x_min = np.min(self.data_series[:, 0])
                data_x_max = np.max(self.data_series[:, 0])
                current_x_min, current_x_max = self.graph_ax.get_xlim()
                new_x_min = min(current_x_min, data_x_min - 0.5)
                new_x_max = max(current_x_max, data_x_max + 0.5)
                self.graph_ax.set_xlim(new_x_min, new_x_max)

            self.graph_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Error actualizando gráfico:\n{str(e)}")

# Crear la aplicación
if __name__ == "__main__":
    #Es necesario si quieres correr la app desde este modulo
    from Logica import solve_sn
    from Logica import cargar_materiales
    from Logica import solve
    from Logica import resolve
    from Logica import evaluate_flexibility
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "default.csv")
    DF:pd.DataFrame =cargar_materiales(csv_path)
    root = tk.Tk()
    app = App(root)
    root.mainloop()
else:
    from .Logica import solve_sn
    from .Logica import cargar_materiales
    from .Logica import solve
    from .Logica import resolve
    from .Logica import evaluate_flexibility
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "default.csv")
    DF:pd.DataFrame =cargar_materiales(csv_path)
    #Toca importarlo relativo cuando se importa el modulo