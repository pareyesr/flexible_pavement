import tkinter as tk
from tkinter import ttk
import os
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Method AASHTO")
        
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
    def create_param_rand_widgets(self,tab,notebook):
        # Etiquetas y cajas de entrada 
        ttk.Label(tab, text="TPD:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Trafico promedio diario (num de carros)").grid(row=0, column=3, padx=10, pady=5, sticky="w")
        self.tpd = ttk.Entry(tab)
        self.tpd.insert(0, "402.39")
        self.tpd.grid(row=0, column=1, padx=10, pady=5, sticky="w") 
        ttk.Label(tab, text="archivo npy:").grid(row=0, column=3, padx=10, pady=5, sticky="w")
        self.arr_ruta = ttk.Entry(tab)
        self.arr_ruta.insert(0, "datos_res")
        self.arr_ruta.grid(row=0, column=4, padx=10, pady=5, sticky="w") 
        ttk.Label(tab, text="vc:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Distribución por sentido (usalmente 0.5)").grid(row=1, column=3, padx=10, pady=5, sticky="w")
        self.vc = ttk.Entry(tab)
        self.vc.insert(0, "0.5")
        self.vc.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="cd:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Carril de diseño (usualmente 1.0 si es de un solo carril por sentido)").grid(row=2, column=3, padx=10, pady=5, sticky="w")
        self.cd = ttk.Entry(tab)
        self.cd.insert(0, "1.0")
        self.cd.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="i:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="indice de crecimiento").grid(row=3, column=3, padx=10, pady=5, sticky="w")
        self.i = ttk.Entry(tab)
        self.i.insert(0, "0.05")
        self.i.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="n:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Meses de diseño").grid(row=4, column=3, padx=10, pady=5, sticky="w")
        self.n_mes = ttk.Entry(tab)
        self.n_mes.insert(0, "360")
        self.n_mes.grid(row=4, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="tamaño:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Tamaño de la muestra aleatoria").grid(row=5, column=3, padx=10, pady=5, sticky="w")
        self.size = ttk.Entry(tab)
        self.size.insert(0, "5000")
        self.size.grid(row=5, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Rate:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Tasa de evaluación para valor presente neto (MENSUAL)").grid(row=6, column=3, padx=10, pady=5, sticky="w")
        self.rate = ttk.Entry(tab)
        self.rate.insert(0, "0.05")
        self.rate.grid(row=6, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Costo de reconstrucción (fijo):").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Valor en $ para los costos fijos de empezar construcción").grid(row=7, column=3, padx=10, pady=5, sticky="w")
        self.cost_rb = ttk.Entry(tab)
        self.cost_rb.insert(0, "1000")
        self.cost_rb.grid(row=7, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Capas de pavimento a reemplazar por reconstrucción").grid(row=8, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="1 capa es rodadura").grid(row=8, column=3, padx=10, pady=5, sticky="w")
        self.layers = ttk.Entry(tab)
        self.layers.insert(0, "2")
        self.layers.grid(row=8, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Etapas de rediseño en la vida util del pavimento").grid(row=9, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="una etapa cada 10 años para un diseño de 30 años sería 3").grid(row=9, column=3, padx=10, pady=5, sticky="w")
        self.step = ttk.Entry(tab)
        self.step.insert(0, "3")
        self.step.grid(row=9, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Semilla aleatoria:").grid(row=10, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="0 para remover semilla (Sin repitibildad)").grid(row=20, column=3, padx=10, pady=5, sticky="w")
        self.seed = ttk.Entry(tab)
        self.seed.insert(0, "63442967")
        self.seed.grid(row=20, column=1, padx=10, pady=5, sticky="w")
        
        # Create the tab for graphics of random transit test.
        tk.Button(master=tab, text="Load", command=self.create_rand_graph_widgets(notebook)).grid(row=0, column=5, columnspan=2, pady=10)
    def create_rand_graph_widgets(self,notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text='Graficas de resultados de trafico aleatorio')
        #Variables 
        confianza = float(self.confianza_entry.get())
        desviacion = float(self.desviacion_entry.get())
        delta_psi = float(self.delta_psi_entry.get())
        modulo_resiliente = float(self.modulo_resiliente_entry.get())
        #
        #ruta ='.\\src\\'+str(self.cruta.get())+".csv"
        #DF= cargar_materiales(ruta)
        
        n_sect=int(self.section_entry.get())-1
        sect=solve(DF,self.calcular_sn(),float(self.grade.get()),float(self.emb.get()),float(self.exc.get()))[n_sect]
        ruta_arr ='.\\src\\'+str(self.arr_ruta.get())+".npy"
        def load_arr(ruta_arr):
            if os.path.exists(ruta_arr):
                arr = np.load(ruta_arr)
            else:
                arr:np.array = evaluate_flexibility(float(self.tpd.get()),
                                                    float(self.vc.get()),
                                                      float(self.cd.get()),
                                                      int(self.size.get()),
                                                      int(self.n_mes.get()),
                                                      float(self.rate.get()),
                                                      self.calcular_sn(),
                                                      confianza,desviacion,delta_psi,modulo_resiliente,DF,
                                                      sect,float(self.grade.get()),
                                                      float(self.emb.get()),
                                                      float(self.exc.get()),
                                                      float(self.cost_rb.get()),
                                                      int(self.layers.get()),
                                                      int(self.step.get()),
                                                      int(self.seed.get()))
                np.save(ruta_arr,arr)
            return arr
        arr = load_arr(ruta_arr)
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot()
        ax.hist(arr)

        canvas = FigureCanvasTkAgg(fig, master=tab)  # A tk.DrawingArea.
        canvas.draw()
        canvas.mpl_connect("key_press_event", lambda event: print(f"you pressed {event.key}"))
        canvas.mpl_connect("key_press_event", key_press_handler)

        button_quit = tk.Button(master=tab, text="Quit", command=tab.destroy)
        # pack_toolbar=False will make it easier to use a layout manager later on.
        toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
        toolbar.update()

        def update_frequency(new_val):
            # retrieve frequency
            ax.clear()
            ax.hist(arr,bins=int(new_val))
            # required to update canvas and attached toolbar!
            canvas.draw()
        slider_update = tk.Scale(tab, from_=5, to=25, orient=tk.HORIZONTAL,command=update_frequency, label="Rangos")

        # Packing order is important. Widgets are processed sequentially and if there
        # is no space left, because the window is too small, they are not displayed.
        # The canvas is rather flexible in its size, so we pack it last which makes
        # sure the UI controls are displayed as long as possible.
        button_quit.pack(side=tk.BOTTOM)
        slider_update.pack(side=tk.BOTTOM)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    def create_resol_widgets(self,tab):
        # Etiquetas y cajas de entrada para recalcular capas
        ttk.Label(tab, text="Nuevo SN:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.new_sn = ttk.Entry(tab)
        self.new_sn.insert(0, "5.0")
        self.new_sn.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Capas a modificar:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.n_layer = ttk.Entry(tab)
        self.n_layer.insert(0, "1")
        self.n_layer.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text="Sección a modificar:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.section_entry = ttk.Entry(tab)
        self.section_entry.insert(0, "1")
        self.section_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        #Etiqueta resultados
        self.resol_result_label = ttk.Label(tab, text="")
        self.resol_result_label.grid(row=6, column=0, columnspan=2, pady=10)
        combined_data = self.recalcular_sol()
        # Botón para recalcular la solución de capas
        ttk.Button(tab, text="Re-Calcular capas", command=self.recalcular_sol).grid(row=3, column=0, columnspan=2, pady=10)
    def create_sol_widgets(self,tab):
        self.sol_result_label = ttk.Label(tab, text="")
        self.sol_result_label.grid(row=6, column=0, columnspan=2, pady=10)
        combined_data = self.calcular_sol()
        """
        for i in range(len(combined_data)):
            for j in range(len(combined_data[0])):
                ttk.Label(tab, text=combined_data[i][j]).grid(row=i+2, column=j+2, padx=0, pady=5)
        """
        # Botón para calcular la solución de capas
        ttk.Button(tab, text="Calcular nuevas capas", command=self.calcular_sol).grid(row=1, column=0, columnspan=2, pady=10)
    def calcular_sol(self):
        #ruta ='.\\src\\'+str(self.cruta.get())+".csv"
        #DF= cargar_materiales(ruta)
        lst=solve(DF,self.calcular_sn(),float(self.grade.get()),float(self.emb.get()),float(self.exc.get()))[:5]
        combined_data = []
        sn_data=[]
        cost_data=[]
        for section in lst:
            section_data = []
            for layer in section:
                name = layer.name
                thickness = layer.thickness
                section_data.append((name, thickness))
            combined_data.append(section_data)
            cost_data.append(section.totalCost)
            sn_data.append(round(sum([l.sn * l.thickness for l in section]),2))
        #self.sol_result_label.config(text="\n".join(combined_data))
        str_res="SN\tCosto\tCAPAS\n"
        for i in range(len(combined_data)):
            str_res+=str(sn_data[i])+"\t"+str(round(cost_data[i],4))+"\t"+str(combined_data[i])+"\n"
        self.sol_result_label.config(text=str_res)
        return combined_data
    
    def recalcular_sol(self):
        #ruta ='.\\src\\'+str(self.cruta.get())+".csv"
        #DF= cargar_materiales(ruta)
        n_sect=int(self.section_entry.get())-1
        sect=solve(DF,self.calcular_sn(),float(self.grade.get()),float(self.emb.get()),float(self.exc.get()))[n_sect]
        lst=resolve(DF,sect,float(self.new_sn.get()),int(self.n_layer.get()),float(self.grade.get()),float(self.emb.get()),float(self.exc.get()))[:1]
        combined_data = []
        sn_data=[]
        cost_data=[]
        for section in lst:
            section_data = []
            for layer in section:
                name = layer.name
                thickness = layer.thickness
                section_data.append((name, thickness))
            combined_data.append(section_data)
            cost_data.append(section.totalCost)
            sn_data.append(round(sum([l.sn * l.thickness for l in section]),2))
        str_res="SN\tCosto ($/sqy)\tCAPAS\n"
        for i in range(len(combined_data)):
            str_res+=str(sn_data[i])+"\t"+str(round(cost_data[i],4))+"\t"+str(combined_data[i])+"\n"
        self.resol_result_label.config(text=str_res)
        return combined_data
    def create_mat_widgets(self, tab):
        # Etiquetas y cajas de entrada para ingresar materiales
        ttk.Label(tab, text="Cargar datos de materiales de un archivo csv").grid(row=0, column=0, padx=0, pady=5, sticky="w")
        self.cruta = ttk.Entry(tab,width=10)
        self.cruta.insert(0, "default")
        self.cruta.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text=".csv").grid(row=0, column=2, padx=0, pady=5, sticky="w")
        titulos=['mat_name','SN','min','max','density','cost','unit','surface','subgrade','alkaline']
        for i in range(len(titulos)):
            ttk.Label(tab, text=titulos[i]).grid(row=1, column=i, padx=0, pady=5)
        ruta ="."+os.sep+"src"+os.sep+str(self.cruta.get())+".csv"
        return self.cargar_mat(tab,ruta)
    def crear_materiales(self,):
        # Crear un nuevo material
        lst_entrys = self.entrys
        for i in range(len(lst_entrys)):
            if i == 0:
                lst_entrys[i]  = str(lst_entrys[i].get())
            elif i==6:
                lst_entrys[i]  = str(lst_entrys[i].get())
            elif i > 6:
                lst_entrys[i]  = True if bool(lst_entrys[i].get()=="True") else False
            else:
                lst_entrys[i]  = float(lst_entrys[i].get())
        print(lst_entrys)
        """DF.loc[-1]=pd.Series(lst_entrys)
        DF.index = DF.index + 1 
        DF = DF.sort_index()
        DF.to_csv("."+os.sep+"src"+os.sep+str(self.cruta.get())+".csv",index=False)"""
        return cargar_materiales("."+os.sep+"src"+os.sep+str(self.cruta.get())+".csv")

    def cargar_mat(self,tab,ruta):
        # Mostrar el resultado de cargar el material en la interfaz
        DF_mat = cargar_materiales(ruta)
        resultado = "cargado exitosamente"
        # Etiqueta para mostrar el resultado de mat. doble for pa crear filas y columnas?
        for i in range(len(DF_mat)):
            for j in range(len(DF_mat.iloc[i])):
                self.mat_label = ttk.Label(tab, text=str(DF_mat.iloc[i].iloc[j]))
                self.mat_label.grid(row=i+2, column=j, padx=10, pady=1)
        self.entrys=[]
        for k in range(len(DF_mat.iloc[0])):
            self.entrys.append(ttk.Entry(tab,width=10))
            self.entrys[k].grid(row=i+3, column=k, padx=10, pady=5, sticky="w")
        # Button to save the values in the DF_mat
        ttk.Button(tab, text="Guardar material", command=self.crear_materiales).grid(row=0, column=5, columnspan=2, pady=10)
        #etiquetas de entrada
        ttk.Label(tab, text="Excavation Cost ($/cyd)").grid(row=i+4, column=0, padx=0, pady=0, sticky="w")        
        self.exc = ttk.Entry(tab,width=10)
        self.exc.insert(0, "20")
        self.exc.grid(row=i+4, column=1, padx=10, pady=5, sticky="w")  
        ttk.Label(tab, text="Embankment Cost ($/cyd)").grid(row=i+5, column=0, padx=0, pady=0, sticky="w")        
        self.emb = ttk.Entry(tab,width=10)
        self.emb.insert(0, "10")
        self.emb.grid(row=i+5, column=1, padx=10, pady=0, sticky="w")        
        self.mat_result_label = ttk.Label(tab, text="")
        self.mat_result_label.grid(row=i+4, column=j//2, columnspan=2, pady=10)
        self.mat_result_label.config(text="El material de capa fue "+(resultado))
        ttk.Label(tab, text="Grade (in)").grid(row=i+6, column=0, padx=0, pady=0, sticky="w")        
        self.grade = ttk.Entry(tab,width=10)
        self.grade.insert(0, "0.0")
        self.grade.grid(row=i+6, column=1, padx=10, pady=5, sticky="w")          
        return DF_mat
        
    def create_sn_widgets(self, tab):
        # Etiquetas y cajas de entrada para calcular SN
        ttk.Label(tab, text="Nivel de Confianza:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.confianza_entry = ttk.Entry(tab)
        self.confianza_entry.insert(0, "0.95")
        self.confianza_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(tab, text="Desviación Estándar:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.desviacion_entry = ttk.Entry(tab)
        self.desviacion_entry.insert(0, "0.35")
        self.desviacion_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(tab, text="NESE:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.W18_entry = ttk.Entry(tab)
        self.W18_entry.insert(0, "5000000")
        self.W18_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(tab, text="ΔPSI:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.delta_psi_entry = ttk.Entry(tab)
        self.delta_psi_entry.insert(0, "1.9")
        self.delta_psi_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        ttk.Label(tab, text="Módulo Resiliente (PSI):").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.modulo_resiliente_entry = ttk.Entry(tab)
        self.modulo_resiliente_entry.insert(0, "5000")
        self.modulo_resiliente_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        # Botón para calcular el valor de SN
        ttk.Button(tab, text="Calcular SN", command=self.calcular_sn).grid(row=5, column=0, columnspan=2, pady=10)

        # Etiqueta para mostrar el resultado de SN
        self.sn_result_label = ttk.Label(tab, text="")
        self.sn_result_label.grid(row=6, column=0, columnspan=2, pady=10)
        self.calcular_sn
    def calcular_sn(self):
        # Obtener valores de las cajas de entrada para calcular SN
        confianza = float(self.confianza_entry.get())
        desviacion = float(self.desviacion_entry.get())
        delta_psi = float(self.delta_psi_entry.get())
        modulo_resiliente = float(self.modulo_resiliente_entry.get())
        W18 = float(self.W18_entry.get())

        # Crear una instancia con los valores ingresados
        valor_sn = solve_sn(Reliavility=confianza,
                                      Standard_Deviation=desviacion,
                                      esal=W18,
                                      Delta_PSI=delta_psi,
                                      Mr=modulo_resiliente)
        # Mostrar el resultado de SN en la interfaz
        self.sn_result_label.config(text="{:.2f}".format(valor_sn))
        print("El valor calculado de SN es: {:.2f}".format(valor_sn))
        return valor_sn

# Crear la aplicación
if __name__ == "__main__":
    #Es necesario si quieres correr la app desde este modulo
    from Logica import solve_sn
    from Logica import cargar_materiales
    from Logica import solve
    from Logica import resolve
    from Logica import evaluate_flexibility
    DF:pd.DataFrame =cargar_materiales("."+os.sep+"src"+os.sep+"default.csv")
    root = tk.Tk()
    app = App(root)
    root.mainloop()
else:
    from .Logica import solve_sn
    from .Logica import cargar_materiales
    from .Logica import solve
    from .Logica import resolve
    from .Logica import evaluate_flexibility
    DF:pd.DataFrame =cargar_materiales("."+os.sep+"src"+os.sep+"default.csv")
    #Toca importarlo relativo cuando se importa el modulo