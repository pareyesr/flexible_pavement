import tkinter as tk
from tkinter import ttk
import os
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
        df=self.create_mat_widgets(mat_tab)

        # Create the tab for solution.
        sol_tab = ttk.Frame(notebook)
        notebook.add(sol_tab, text='Capas solución')
        self.create_sol_widgets(sol_tab,df)
    def create_sol_widgets(self,tab,DF):
        combined_data = self.calcular_sol(DF)
        for i in range(len(combined_data)):
            for j in range(len(combined_data[0])):
                ttk.Label(tab, text=combined_data[i][j]).grid(row=i+1, column=j+1, padx=0, pady=5)
        # Botón para calcular la solución de capas
        ttk.Button(tab, text="Calcular nuevas capas", command=self.calcular_sol(DF)).grid(row=i+2, column=1, columnspan=2, pady=10)
    def calcular_sol(self,DF):
        print(2)
        lst=solve(DF,self.calcular_sn(),float(self.grade.get()),float(self.emb.get()),float(self.exc.get()))[:3]
        combined_data = []
        for section in lst:
            section_data = []
            for layer in section:
                name = layer.name
                thickness = layer.thickness
                section_data.append((name, thickness))
            combined_data.append((sum([l.sn * l.thickness for l in section]), section_data))
        return combined_data
    def create_mat_widgets(self, tab):
        # Etiquetas y cajas de entrada para ingresar materiales
        ttk.Label(tab, text="Cargar datos de materiales de un archivo csv").grid(row=0, column=0, padx=0, pady=5, sticky="w")
        self.cruta = ttk.Entry(tab)
        self.cruta.insert(0, "default")
        self.cruta.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(tab, text=".csv").grid(row=0, column=2, padx=0, pady=5, sticky="w")
        titulos=['mat_name','SN','min','max','density','cost','unit','surface','subgrade','alkaline']
        for i in range(len(titulos)):
            ttk.Label(tab, text=titulos[i]).grid(row=1, column=i, padx=0, pady=5)
        ruta ='.\\src\\'+str(self.cruta.get())+".csv"
        if os.path.exists(ruta):
            return self.cargar_mat(tab,ruta)
        else:
            #DF=crear_materiales(ruta)
            x=0
        
    def cargar_mat(self,tab,ruta):
        # Mostrar el resultado de cargar el material en la interfaz
        DF_mat = cargar_materiales(ruta)
        resultado = "cargado exitosamente"
        # Etiqueta para mostrar el resultado de mat. doble for pa crear filas y columnas?
        for i in range(len(DF_mat)):
            for j in range(len(DF_mat.iloc[i])):
                self.mat_label = ttk.Label(tab, text=str(DF_mat.iloc[i].iloc[j]))
                self.mat_label.grid(row=i+2, column=j, padx=10, pady=1)
        #etiquetas de entrada
        ttk.Label(tab, text="Excavation Cost ($/cyd)").grid(row=i+3, column=0, padx=0, pady=0, sticky="w")        
        self.exc = ttk.Entry(tab)
        self.exc.insert(0, "20")
        self.exc.grid(row=i+3, column=1, padx=10, pady=5, sticky="w")        
        ttk.Label(tab, text="Embankment Cost ($/cyd)").grid(row=i+4, column=0, padx=0, pady=0, sticky="w")        
        self.emb = ttk.Entry(tab)
        self.emb.insert(0, "10")
        self.emb.grid(row=i+4, column=1, padx=10, pady=0, sticky="w")        
        self.mat_result_label = ttk.Label(tab, text="")
        self.mat_result_label.grid(row=i+3, column=j//2, columnspan=2, pady=10)
        self.mat_result_label.config(text="El material de capa fue "+(resultado))
        ttk.Label(tab, text="Grade (in)").grid(row=i+5, column=0, padx=0, pady=0, sticky="w")        
        self.grade = ttk.Entry(tab)
        self.grade.insert(0, "0.0")
        self.grade.grid(row=i+5, column=1, padx=10, pady=5, sticky="w")        
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
        self.sn_result_label.config(text="El valor calculado de SN es: {:.2f}".format(valor_sn))
        return valor_sn

# Crear la aplicación
if __name__ == "__main__":
    #Es necesario si quieres correr la app desde este modulo
    from Logica import solve_sn
    from Logica import cargar_materiales
    from Logica import solve
    root = tk.Tk()
    app = App(root)
    root.mainloop()
else:
    from .Logica import solve_sn
    from .Logica import cargar_materiales
    from .Logica import solve
    #Toca importarlo relativo cuando se importa el modulo