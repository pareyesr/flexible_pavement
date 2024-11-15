import math
import scipy.stats as st
import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import os
from itertools import combinations 
# MPA to PSI = x * 145.03773773

from scipy.stats import norm

def pred_W18(tpd:int,vc:float,cd:float,i:float,n:int):
    """
    TPD:int = Trafico promedio diario\n
    vc:float =Distribución por sentido (usalmente 0.5)\n
    cd:float =Carril de diseño (usualmente 1.0 si es de un solo carril por sentido)\n
    i:float =indice de crecimiento \n
    n:int =años de diseño\n
    return -> 365*TPD*VC*CD*(1+i)^n/ln(1+i)\n
    """
    return 365*tpd*vc*cd*(1+i)**n/math.log(1+i)


def predict_pavement_esal(r, so, sn, psi, mr):
    """
    Generate a prediction for Equivalent Single Axle Load (ESAL) based on given parameters.

    Parameters:
    r (float): Reliavility (usually 0.5-0.999)
    so (float): standard error (usually 0.4-0.5 for asphalt, 0.35-0.4 for concrete)
    sn (float): structural number
    psi (float): allowable delta in serviceability index (usually 1.0-3.0)
    mr (float): resilient modulus [PSI]

    Returns:
    float: the predicted ESAL value
    """
    right_side = -norm.ppf(r)*so+9.36*np.log10(sn+1)-0.2+(np.log10(psi/(4.2-1.5))/(0.4+1094/(sn+1)**5.19))+2.32*np.log10(mr)-8.07
    esals = 10**right_side
    return esals

def solve_sn(Reliavility, Standard_Deviation, Delta_PSI, Mr, esal):
    def f(sn):
        val = sn[0]
        return predict_pavement_esal(Reliavility, Standard_Deviation, val, Delta_PSI, Mr) - esal
    return fsolve(f, np.array([15]))[0]
"""
Solucionar capa
"""
import random
from copy import deepcopy


class Layer():
    def __init__(self, material_table_row):
        self.name = material_table_row['mat_name']
        self.sn = material_table_row['sn']
        self.cost = material_table_row['cost']
        self.density = material_table_row['density']
        self.unit = material_table_row['unit']
        surface = material_table_row['surface']
        subgrade = material_table_row['subgrade']
        alkaline = material_table_row['alkaline']
        minimum_lift = material_table_row['min']
        self.min_lift = minimum_lift
        self.thickness = minimum_lift
        self.max_lift = material_table_row['max']
        self.cost_per_inch = self.calc_cost_per_inch()
        self.surface_code = 1 if surface == "Yes" else 0
        self.subgrade_code = 1 if subgrade == "Yes" else 0
        self.alkaline_code = 1 if alkaline == "Yes" else 0
        self.cost_per_sn = self.cost_per_inch / self.sn
        return None

    def calc_cost_per_inch(self):
        if self.unit == "ton":
            tonnage = self.density * 27 / 2000
            cost_per_sy = self.cost * tonnage / 36 # in per yd
            return cost_per_sy
        elif self.unit == "cyd":
            cost_per_sy = self.cost / 36
            return cost_per_sy
        elif self.unit == "sqyd":
            return self.cost / self.min_lift
        else:
            return 0.0

class Section(list): # subclass list just for sanity
    def __init__(self, *layers):
        self.totalCost = 0
        super().__init__(*layers)

def make_material_list(material_table:pd.DataFrame)->list[Layer]:
    return [Layer(material_table.iloc[i]) for i in range(len(material_table))]


def make_trial_section(material_list) -> Section:
    # select 1-4 materials at random and save to an array
    num_materials: int = np.random.randint(1, 5)
    section = Section()
    for _ in range(num_materials):
        section.append(deepcopy(random.choice(material_list)))
    section.sort(key = lambda l : l.surface_code)
    section.reverse()
    section.sort(key = lambda l : l.subgrade_code)
    return section

def make_possible_sections(material_list,num_capas) -> list:
    
    lst_compl =[]

    surface_lst = []
    subgrad_lst = []
    alkaline_lst = []
    rest_lst=[]
    mat_size = len(material_list)
    for i in range(mat_size):
        lay = material_list[i]
        if lay.surface_code == 1:
            surface_lst.append(i)
        if lay.subgrade_code == 1:
            subgrad_lst.append(i)
        if lay.alkaline_code == 1:
            alkaline_lst.append(i)
        elif lay.subgrade_code == 0 and lay.surface_code ==0:
            rest_lst.append(i)
    indices=combinations(range(len(material_list)),num_capas)
    return lst_compl

def validate_section(section:Section)->bool:
    # no duplicate courses of materials
    names = [l.name for l in section]
    if len(names) != len(set(names)):
        return False
    # must have a surface course
    if section[0].surface_code == 0:
        return False
    # cannot have multiple subgrade treatments
    if sum([l.subgrade_code for l in section]) > 1:
        return False
    # cannot have adjacent alkaline courses
    alk = np.array([l.alkaline_code for l in section])
    alk_roll = np.roll(alk, 1)
    if np.logical_and(alk, alk_roll).any():
        return False
    # thickness must be a positive number
    if any([l.thickness <= 0 for l in section]):
        return False
    # thickness must be achievable within lift size limits
    for l in section:
        if not l.thickness % l.min_lift < l.max_lift-l.min_lift or l.thickness % l.max_lift == 0:
            return False
    return True

def remove_duplicate_sections(section_list):
    result_list = []
    used_combinations = set()
    for section in section_list:
        names = [l.name for l in section]
        names.sort()
        if tuple(names) in used_combinations:
            continue
        result_list.append(section)
        used_combinations.add(tuple(names))
    return result_list


def section_sn(section):
    return sum([l.sn * l.thickness for l in section])


def section_cost(section, grade, embankment_cost, excavation_cost):
    #Se calcula la diferencia de elevación de la subrazante. Si es negativa se multiplica por el costo de excavación, si es positiva por el costo de explanación
    subgrade_elevation = grade - sum([layer.thickness for layer in section])
    earthwork = ((embankment_cost if subgrade_elevation > 0 else excavation_cost)/36)*subgrade_elevation
    section.totalCost = sum([l.cost_per_inch * l.thickness for l in section]) + earthwork
    return section.totalCost


def modify_thickness(section:Section, goal_sn,n=0):
    """
    section: Sección a modificar
    goal_sn: float, objetivo SN a llegar
    n:int, Capas que se pueden modificar(primera a n), 0 todas
    """
    epsilon = 0.01
    current_sn = section_sn(section)
    if n==0:
        cost_index = [(i,l) for i,l in enumerate(section)]
        cost_index.sort(key=lambda x: x[1].cost_per_sn)
    else:
        print(n)
        lay=section[0:n+1]
        cost_index = [(i,l) for i,l in enumerate(lay)]
        cost_index.sort(key=lambda x: x[1].cost_per_sn)
    increment_size = lambda l: 0.5 if l.min_lift < 2.0 else 1.0
    for _ in range(10):  # this may not benefit from multiple passes
        delta = goal_sn - current_sn+0.1
        if abs(delta) < epsilon:
            break
        for i,_l in cost_index:
            layer:Layer = section[i]
            if layer.min_lift == layer.max_lift:
                continue # pass layers with fixed thickness
            inc = increment_size(layer)
            inc_sn_delta = layer.sn * inc
            adjustment = delta // inc_sn_delta if delta > 0 else np.ceil(delta / inc_sn_delta)
            layer.thickness += inc * adjustment
            if layer.thickness <= layer.min_lift:
                layer.thickness = layer.min_lift
            current_sn = section_sn(section)
            delta = goal_sn - current_sn+0.1
    return section


def solve(material_table, goal_sn, grade=0.0, embankment_cost=0.0, excavation_cost=0.0):
    material_list = make_material_list(material_table)
    sample_population = 5000
    trial_sections = [make_trial_section(material_list) for _ in range(sample_population)]
    unique_sections = remove_duplicate_sections(trial_sections)
    valid_sections = [s for s in unique_sections if validate_section(s)]
    modified_sections = [modify_thickness(s, goal_sn) for s in valid_sections]
    revalidated_sections = [s for s in modified_sections if validate_section(s)]
    revalidated_sections.sort(key=lambda s: section_cost(s, grade, embankment_cost, excavation_cost))
    return revalidated_sections
    
def cargar_materiales(ruta:str)->pd.DataFrame:
    """Carga la lista de materiales a partir de un archivo de texto, devuelve un DATAFRAME"""
    if os.path.exists(ruta):
        tab_ld = pd.read_csv(ruta)
    else:
        #toca crear el archivo
        tab_ld = open(ruta,"a")
    return tab_ld

#TODO SOLVER sobre capas hechas

def resolve(material_table,sect:Section,SN:float,n:int,grade=0.0, embankment_cost=0.0, excavation_cost=0.0):
    """
    section: Sección a recalcular
    SN: SN a superar
    n: Numero de capas dañadas sobre la sección aka capas a remover
    """
    material_list = make_material_list(material_table)
    for i in range(n):
        print(sect[i].name)
    prev_sect = sect[n:]
    trial_sections=[]
    for _ in range(10000):
        # select 1-4 materials at random and save to an array    
        num_materials: int = np.random.randint(1, 5)
        new_caps = Section()
        for _ in range(num_materials):
            new_caps.append(deepcopy(random.choice(material_list)))
        for i in range(len(prev_sect)):
            new_caps.append(prev_sect[i])
        trial_sections.append(new_caps)
    unique_sections = remove_duplicate_sections(trial_sections)
    valid_sections = [s for s in unique_sections if validate_section(s)]
    #Aumentar espesor
    modified_sections = [modify_thickness(s, SN,len(s)-len(prev_sect)) for s in valid_sections]    
    revalidated_sections = [s for s in modified_sections if validate_section(s)]
    revalidated_sections.sort(key=lambda s: section_cost(s, grade, embankment_cost, excavation_cost))
    return revalidated_sections