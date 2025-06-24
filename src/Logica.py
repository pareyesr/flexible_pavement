import math
import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import os
from itertools import combinations 
import multiprocessing
import random
from copy import deepcopy
# MPA to PSI = x * 145.03773773

from scipy.stats import norm

def pred_W18(tpd:int,vc:float,cd:float,i:float,n:int):
    """
    TPD:int = Trafico promedio diario\n
    vc:float =Vehiculos comerciales(B+C) \n
    cd:float = %trafico carril de diseño \n
    i:float =indice de crecimiento anual \n
    n:int =años de diseño\n
    return -> 365*TPD*VC*CD*((1+i)^n-1)/ln(1+i)\n
    """
    A0 = 365 * tpd * vc * cd  # Initial annual traffic
    if i == 0:
        return A0 * n  # Avoid division by zero
    else:
        return A0 * ((1 + i)**n - 1) / i


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

def solve_sn(Reliavility, Standard_Deviation, Delta_PSI, Mr, esal,sn=15):
    """
    Calculate the required Structural Number (SN) for given parameters.
    
    Parameters:
    Reliavility (float): Reliability level (0.5-0.999)
    Standard_Deviation (float): Standard error (0.4-0.5 for asphalt, 0.35-0.4 for concrete)
    Delta_PSI (float): Allowable serviceability loss (1.0-3.0)
    Mr (float): Resilient modulus [PSI]
    esal (float): Design ESALs
    
    Returns:
    float: Calculated SN value, or None if calculation fails
    """
    def f(sn):
        val = sn[0]
        return predict_pavement_esal(Reliavility,Standard_Deviation,val,Delta_PSI,Mr) - esal
    return fsolve(f,np.array([sn]),xtol=0.001)[0]

"""
Solucionar capa
"""


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
        self.surface_code = 1 if bool(surface) else 0
        self.subgrade_code = 1 if bool(subgrade) else 0
        self.alkaline_code = 1 if bool(alkaline) else 0
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
    def get_sn(self):
        return sum([l.sn * l.thickness for l in self])
    def info(self):
        return {
            'total_sn': self.get_sn(),
            'total_cost': self.totalCost,
            'layers': [l.name for l in self],
            'thicknesses': [l.thickness for l in self],
            'sns': [l.sn for l in self]
        }

def make_material_list(material_table:pd.DataFrame)->list[Layer]:
    sorted_df = material_table.sort_values(by='surface', ascending=False)
    return [Layer(sorted_df.iloc[i]) for i in range(len(sorted_df))]

def make_trial_section(material_list) -> Section:
    """Depreciated. Use make_possible_sections instead"""
    # select 1-4 materials at random and save to an array
    num_materials: int = np.random.randint(1, 5)
    section = Section()
    for _ in range(num_materials):
        section.append(deepcopy(random.choice(material_list)))
    section.sort(key = lambda l : l.surface_code)
    section.reverse()
    section.sort(key = lambda l : l.subgrade_code)
    return section

def make_possible_sections(material_list:list,num_capas:int):
    """
    ->Return list[Section] \n
    material_list should be sorted by surface_code on top\n
    the list have all the posible combinations without repetition n! / (k! * (n - k)!) when k <= n.\n
    k being num_capas and n being len(material_list). Aditionally removes restrictions such as having one surface, 1 or less alkaline layer, and 1 or less subgrade treatment.\n
    material_list: list of materials generated by make_material_list()\n
    num_capas: The amount of layers of materials to generate the possible sections
    """
    if num_capas <= 0 or num_capas > len(material_list):
        return []
        
    surface_lst = []
    subgrad_lst = []
    alkaline_lst = []
    mat_size = len(material_list)
    
    # Create lists of indices for materials with special properties
    for i in range(mat_size):
        lay = material_list[i]
        if lay.surface_code == 1:
            surface_lst.append(i)
        if lay.subgrade_code == 1:
            subgrad_lst.append(i)
        if lay.alkaline_code == 1:
            alkaline_lst.append(i)
            
    # Generate all possible combinations
    indices = list(combinations(range(mat_size), num_capas))
    if not indices:
        return []
        
    # Filter combinations that don't start with a surface material
    valid_indices = []
    for combo in indices:
        if combo[0] not in surface_lst:
            continue
            
        surface = 0
        alk = 0
        trat_sub = 0
        valid = True
        
        for h in combo:
            if h in surface_lst:
                surface += 1
            if h in alkaline_lst:
                alk += 1
            if h in subgrad_lst:
                trat_sub += 1
                
            # Check constraints
            if alk >= 2 or trat_sub >= 2 or surface >= 2:
                valid = False
                break
                
        if valid and surface == 1:  # Must have exactly one surface layer
            valid_indices.append(combo)
            
    # Create sections from valid combinations
    lst_compl = []
    for ind_sect in valid_indices:
        section = Section()
        for j in ind_sect:
            section.append(deepcopy(material_list[j]))
        lst_compl.append(section)
        
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


def modify_thickness(section:Section, goal_sn:float,n=0):
    """
    section: Section to modify \n
    goal_sn: Structural number to get\n
    n:int, Number of layers to modify from section (from 1 to n) // 0 all\n
    """
    epsilon = 0.01
    current_sn = section_sn(section)
    # Determine which layers can be modified
    if n == 0:
        modifiable_layers = [(i, layer) for i, layer in enumerate(section)]
    else:
        modifiable_layers = [(i, layer) for i, layer in enumerate(section[:n])]
    # Sort layers by cost efficiency (cost per SN)
    modifiable_layers.sort(key=lambda x: x[1].cost_per_sn)
    
    # Define increment size based on layer minimum thickness
    increment_size = lambda l: 0.5 if l.min_lift < 2.0 else 1.0
    for _ in range(10):  # this may not benefit from multiple passes
        delta = goal_sn - current_sn+0.1
        if abs(delta) < epsilon:
            break
            
        # Modify only the modifiable layers
        for i, _ in modifiable_layers:
            layer = section[i]
            if layer.min_lift == layer.max_lift:
                continue # pass layers with fixed thickness
            inc = increment_size(layer)
            inc_sn_delta = layer.sn * inc
            adjustment = delta // inc_sn_delta if delta > 0 else np.ceil(delta / inc_sn_delta)
            layer.thickness += inc * adjustment
            if layer.thickness <= layer.min_lift:
                layer.thickness = layer.min_lift
            elif layer.thickness > layer.max_lift:
                layer.thickness = layer.max_lift
            current_sn = section_sn(section)
            delta = goal_sn - current_sn+0.1#Added value to be over goal_sn in most cases. otherwise it goes near goal_sn
    return section


def solve(material_table, goal_sn, grade=0.0, embankment_cost=0.0, excavation_cost=0.0,min_capas=1):
    material_list = make_material_list(material_table)
    valid_sections = []
    for i in range(min_capas,6):
        valid_sections += make_possible_sections(material_list,i)
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
        tab_ld.write("mat_name,sn,min,max,density,cost,unit,surface,subgrade,alkaline\n")
        tab_ld.close()
        tab_ld = pd.read_csv(ruta)
    return tab_ld

def resolve(material_table, sect, goal_sn, unmodify_bottom_layers=0,grade=0.0, embankment_cost=0.0, excavation_cost=0.0):
    """
    Modify an existing section by adjusting the top layers to achieve the target SN without modifying the bottom layers.

    Args:
        material_table: DataFrame with material properties
        sect: Existing section to modify
        goal_sn: Target structural number
        unmodify_bottom_layers: Number of base layers that cannot be modified
        grade: Grade adjustment
        embankment_cost: Cost of embankment
        excavation_cost: Cost of excavation
        
    Returns:
        tuple: (modified section, actual SN achieved)
    """
    
    material_list = make_material_list(material_table)
    for i in reversed(range(len(material_list))):
        for j in range(unmodify_bottom_layers):
            if material_list[i].name == sect[-j-1].name:
                material_list.pop(i)
    possible_sections = []
    for i in range(1,len(material_list)):
        possible_sections += make_possible_sections(material_list,i)
    for section in possible_sections:
        section.extend(deepcopy(sect[len(sect)-unmodify_bottom_layers:]))
    modified_sections = []
    for section in possible_sections:
        modified_sections.append(modify_thickness(section, goal_sn,len(section)-unmodify_bottom_layers))
    modified_sections.sort(key=lambda s: section_cost(s, grade, embankment_cost, excavation_cost))
    valid_sections = [s for s in modified_sections if validate_section(s)]
    return valid_sections[0], section_sn(valid_sections[0])

#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
def make_simulated_transit(TPD=402.39,vc=0.5,cd=1.0,size=5000,n=360,seedint=63442967,f_mean=lambda x:0.047,f_std=lambda x:0.057)->tuple[np.array,np.array]:
    """
    Funtion to generate a bunch of simulated transit\n
    n:int = meses de diseño\n
    seedint = semilla de rng, 0 para no usar semilla.\n
    f_mean = function with the mean of the growth rates for the simulated period
    f_std = function with the standard deviation of f_mean
    return tuple of 2D-array like of size*n length of traffic and acumulative traffic at time index (monthly)
    """
    # Monthly growth rates (shape: size x n)
    if seedint != 0:
        rng = np.random.default_rng(seed=seedint)
        #grow_rates = np.random.normal(loc=mu_monthly,scale=sigma_monthly, size=(size,n))
    grow_rates = np.zeros((size, n))
    for i in range(size):
        for j in range(n):
            grow_rates[i,j] = rng.normal(loc=f_mean(j),scale=f_std(j))
    initial_monthly_trips = TPD * 365 / 12 * vc * cd
    res = np.zeros((size, n))
    acum = np.zeros((size, n))
    for sim in range(size):
        res[sim, 0] = initial_monthly_trips
        acum[sim, 0] = initial_monthly_trips
        for month in range(1, n):
            #Growth_rate is being calculated from the first month in mind.
            res[sim, month] = initial_monthly_trips  * (1 + grow_rates[sim, month])
            acum[sim, month] = acum[sim, month-1] + res[sim, month]
    # Round to integers as traffic should not be fractional
    res = np.round(res)
    acum = np.round(acum)
    return res, acum 
#print(make_simulated_transit(100,size=5000,n=360,f_mean=lambda x:x*0.02,f_std=lambda y:y*0.0057))
def calculate_break(arr:np.array,SN_dis,Reliavility,Standard_Deviation,Delta_PSI,Mr)->int:
    """
    arr: array like with sn from acumulative transit\n
    Makes a binary search for the postion when the design fails first\n
    return len(arr)+1 if doesnt fail
    """
    low = 0
    high = len(arr) - 1
    mid = 0
    while low <= high:
        mid = (high + low) // 2
        sn = arr[mid]
        if sn < SN_dis:
            low = mid + 1
        elif sn > SN_dis:
            high = mid - 1
        # means sn IS EQUAL to sn_dis at mid
        else:
            return mid
    # We should reach here meaning mid is the lowest possible without being over SN_dis, so we get the next when it fails 
    #note, it returns len(arr)+1 if dont fail on all the array
    return mid+1

def W18_i_regression(traffic_annual:np.array)->float:
    """
    Calculate the annual growth rate (i) from annual traffic data.
    Args:
        traffic_annual: Array of annual traffic values
    Returns:
        float: Annual growth rate (i) that can be used in pred_W18
    """
    growth_rates = []
    for k in range(1, len(traffic_annual)):
        if traffic_annual[k-1] == 0:
            continue
        i_k = (traffic_annual[k] / traffic_annual[k-1]) - 1
        growth_rates.append(i_k)
    return np.mean(growth_rates)
def npv(r, arr):
    sum_pv = 0.0
    for i in range(len(arr)):
        sum_pv += arr[i] / ((1 + r) ** i)
    return sum_pv
def traditional_design(params:dict,DF,min_capas=1)->np.array:
    """
    Funtion to design a pavement by predicting the W18 and using the traditional approach\n
    params:dict = dictionary with all the parameters, needs to have the following keys:\n
    TPD:int = Trafico promedio diario\n
    vc:float =Distribución por sentido (usalmente 0.5)\n
    cd:float =Carril de diseño (usualmente 1.0 si es de un solo carril por sentido)\n
    n:int =años de diseño\n
    rate:float =tasa de descuento\n    
    DF:pd.DataFrame = DataFrame with the materials\n
    return section of the design,sn_design,m
    """
    # Use same logic as make_simulated_transit
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
    
    m = W18_i_regression(mean_traffic)
    esal = pred_W18(params['TPD'],params['vc'],params['cd'],m,params['n']//12)
    sn_design = solve_sn(params['Reliavility'],params['Standard_Deviation'],params['Delta_PSI'],params['Mr'],esal)

    dis_sect = solve(DF,sn_design,params['grade'],params['emb'],params['excv'],min_capas)[0]
    return dis_sect,sn_design,m
def calculate_sn_projected(acumulated:np.array,params:dict):
    """
    transit: array with acumulated traffic
    params: dictionary with all the parameters
    return: array with projected SN
    """
    sn_projected = np.zeros(len(acumulated))
    sn_projected[-1] = solve_sn(params["Reliavility"], params["Standard_Deviation"], params["Delta_PSI"], params["Mr"],acumulated[-1])
    for i in range(len(acumulated)-2,-1,-1):
        try:
            sn_projected[i] = solve_sn(params["Reliavility"], params["Standard_Deviation"], params["Delta_PSI"], params["Mr"],acumulated[i],sn_projected[i+1])
        except:
            raise Exception("Error in calculate_sn_projected")
    start_value = sn_projected[0]
    for i in range(len(sn_projected)):
        sn_projected[i] -= start_value * (len(sn_projected)-i-1)/(len(sn_projected)-1)
    return sn_projected
def save_design(section:Section,sim:int,period:int,startover:bool=False):
    sect_info = {'simulation':sim,'period':period,'total_sn':section.get_sn(),'total_cost':section.totalCost,'layer_materials':[],'layer_thicknesses':[],'layer_sns':[]}
    # Store layer information
    materials = []
    thicknesses = []
    sns = []
    for layer in section:
        materials.append(layer.name)
        thicknesses.append(layer.thickness)
        sns.append(layer.sn)
    sect_info['layer_materials'] = materials
    sect_info['layer_thicknesses'] = thicknesses
    sect_info['layer_sns'] = sns
    sect_info['startover'] = startover
    return sect_info
def evaluate_flexibility(params:dict,DF)->np.array:
    """
    Function to evaluate the design flexibility\n
    params:dict = dictionary with all the parameters, keys\n
    DF:pd.DataFrame = DataFrame with the materials\n
    return array of size length of npv (one each for all the simulations)
    """
    # Create list to store section information (would be converted to DataFrame later)
    section_info = []
    
    random_esals,acumulated = make_simulated_transit(params['TPD'],params['vc'],params['cd'],params['size'],params['n'],params['seedint'],params['mu_function'],params['sigma_function'])
    n_step = params['step']
    
    # Step 1: Generate traditional design for full period
    first_design, sn_first_design, m = traditional_design(params, DF, params.get('min_capas', 4))
    acumulated_SN = np.zeros((params['size'],params['n']))
    # Step 2: Calculate base layers SN (bottom layers that won't be modified frequently)
    num_top_layers = params.get('capas', 3)  # Number of top layers to modify
    base_sn = 0
    for i in range(num_top_layers, len(first_design)):
        base_sn += first_design[i].sn * first_design[i].thickness
    
    # Step 3: Calculate step design for n_step period
    projected_w18_step = pred_W18(params['TPD'], params['vc'], params['cd'], m, n_step//12)
    projected_sn_step = solve_sn(params['Reliavility'], params['Standard_Deviation'], params['Delta_PSI'], params['Mr'], projected_w18_step)
    
    # Step 4: Calculate reduction in base SN per step (gradual deterioration)
    total_steps = params['n'] // n_step
    reduce_base_step = base_sn / total_steps if total_steps > 0 else 0
    
    
    # Calculate how much SN the top layers need to provide
    # Base layers will deteriorate over time, so we need to account for that
    base_sn_after_step = base_sn - reduce_base_step
    required_top_sn = projected_sn_step - base_sn_after_step
    
    # Design top layers to meet the required SN and ignore the base layers SN
    step_design, actual_sn = resolve(DF, deepcopy(first_design), required_top_sn+base_sn, 
                                   len(first_design)-num_top_layers, params['grade'], params['emb'], params['excv'])
    
    
    # Store initial design information for all simulations
    for sim in range(params['size']):
        section_info.append(save_design(step_design, sim, 0))
    
        # Step 6: Process each simulation
    for sim in range(params['size']):
        acumulated_sim = acumulated[sim, :]
        current_step_design = deepcopy(step_design)  # Start with initial step design
        
        # Use calculate_sn_projected to get all required SNs at once
        sn_projected = calculate_sn_projected(acumulated_sim, params)
        acumulated_SN[sim,:] = sn_projected
        # Calculate current top layers SN capacity
        top_sn = actual_sn-base_sn
        
        previous_break = 0
        current_base_sn = base_sn
        # Loop until we reach the end of the simulation period
        while previous_break < params['n']:
            # Create array of required top SN (subtracting current base SN)
            
            # Use calculate_break to find when top layers fail
            n_break = calculate_break(sn_projected, top_sn+reduce_base_step, 
                                    params['Reliavility'], params['Standard_Deviation'], 
                                    params['Delta_PSI'], params['Mr'])
            
            
            # If no break found within simulation period, we're done
            if n_break >= params['n']:
                break
            
            # Redesign based on current conditions
            annual_traffic = np.zeros((n_break-previous_break)//12)
            for i in range(n_break-previous_break):
                if i//12 < len(annual_traffic):
                    annual_traffic[i//12] += random_esals[sim,previous_break+i]
                else:
                    break
            m = W18_i_regression(annual_traffic)
            esal = pred_W18(params['TPD'],params['vc'],params['cd'],m,n_step//12)
            required_sn = solve_sn(params['Reliavility'], params['Standard_Deviation'], 
                               params['Delta_PSI'], params['Mr'], esal,actual_sn)
            if current_base_sn <= 0:
                # If base is exhausted, design completely new section using solve()
                current_step_design = solve(DF, required_sn, params['grade'], params['emb'], 
                                          params['excv'], params.get('min_capas', 1))[0]
                # Reset base SN since we have a new design
                # Recalculate base SN for new design
                current_base_sn = sum([layer.sn * layer.thickness 
                                     for layer in current_step_design[num_top_layers:]])
                actual_achieved_sn = section_sn(current_step_design)
                base_sn = len(current_step_design)-num_top_layers
                section_info.append(save_design(current_step_design, sim, n_break,startover=True))
            else:
                current_base_sn -= reduce_base_step
                # Use resolve() to modify top layers only
                required_top_sn = required_sn - current_base_sn
                current_step_design, actual_achieved_sn = resolve(DF, deepcopy(current_step_design), 
                                                                required_top_sn+current_base_sn, len(current_step_design)-num_top_layers, 
                                                                params['grade'], params['emb'], params['excv'])
                section_info.append(save_design(current_step_design, sim, n_break,startover=False))
            
            top_sn += actual_achieved_sn-base_sn
            # Update previous break position
            previous_break = n_break
    
    # Create DataFrame with all section information
    results_df = pd.DataFrame(section_info)
    
    return results_df, acumulated_SN