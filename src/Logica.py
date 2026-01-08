import math
import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import os
from itertools import combinations 
import multiprocessing
import random
from copy import deepcopy
import re
# MPA to PSI = x * 145.03773773

from scipy.stats import norm

def pred_W18(tpd:int,vc:float,cd:float,i:float,n:int,cam:float=1.0):
    """
    TPD:int = Trafico promedio diario\n
    vc:float =Vehiculos comerciales(B+C) \n
    cd:float = %trafico carril de diseño \n
    i:float =indice de crecimiento anual \n
    n:int =años de diseño\n
    cam:float = factor de camion\n
    return -> 365*TPD*VC*CD*((1+i)^n-1)/ln(1+i)\n
    """
    A0 = 365 * tpd * vc * cd  # Initial annual traffic
    if i == 0:
        return A0 * n  # Avoid division by zero
    else:
        return cam*A0 * ((1 + i)**n - 1) / i


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
        self.material_type = material_table_row['material_type']
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
        self.capacity_Ne = 0
        
        # Conversion factors
        self.INCH_TO_CM = 2.54
        self.CM_TO_INCH = 1.0 / 2.54
        # Helper function to safely get value from pandas Series or dict, handling NaN
        def safe_get(key, default=np.nan):
            try:
                if hasattr(material_table_row, 'index'):
                    # pandas Series
                    if key in material_table_row.index:
                        val = material_table_row[key]
                        return val if pd.notna(val) else default
                else:
                    # dict
                    if key in material_table_row:
                        val = material_table_row[key]
                        # Check for NaN, None, or empty string
                        if val is None or val == '':
                            return default
                        try:
                            if pd.isna(val):
                                return default
                        except (TypeError, ValueError):
                            pass
                        return val
            except (KeyError, AttributeError):
                pass
            return default
        
        # Map E (Young's Modulus) and Poisson_ratio based on material_type
        if self.material_type == 'Bituminous':
            self.E = safe_get('E_bit_MPa_15C_10Hz')
            self.epsilon_6_strain_10e_6 = safe_get('epsilon_6_strain_10e-6')
            self.b_slope_bit = safe_get('b_slope_bit')
            self.SN_logN_bit = safe_get('SN_logN_bit')
            self.Poisson_ratio = 0.35  # Bituminous Materials
            # Set other type-specific properties to NaN
            self.E_hyd_MPa_360d = np.nan
            self.sigma_6_MPa_10e6 = np.nan
            self.b_slope_hyd = np.nan
            self.SN_logN_hyd = np.nan
            self.Rt_MPa = np.nan
            self.E_untreated_MPa = np.nan
        elif self.material_type == 'Hydraulic':
            self.E = safe_get('E_hyd_MPa_360d')
            self.sigma_6_MPa_10e6 = safe_get('sigma_6_MPa_10e6')
            self.b_slope_hyd = safe_get('b_slope_hyd')
            self.SN_logN_hyd = safe_get('SN_logN_hyd')
            self.Rt_MPa = safe_get('Rt_MPa')
            self.Poisson_ratio = 0.25  # Materials Treated with Hydraulic Binders and Cement Concrete
            # Set other type-specific properties to NaN
            self.E_bit_MPa_15C_10Hz = np.nan
            self.epsilon_6_strain_10e_6 = np.nan
            self.b_slope_bit = np.nan
            self.SN_logN_bit = np.nan
            self.E_untreated_MPa = np.nan
        elif self.material_type == 'Untreated':
            self.E = safe_get('E_untreated_MPa')
            self.Poisson_ratio = 0.35  # Untreated Granular Materials
            # Set other type-specific properties to NaN
            self.E_bit_MPa_15C_10Hz = np.nan
            self.epsilon_6_strain_10e_6 = np.nan
            self.b_slope_bit = np.nan
            self.SN_logN_bit = np.nan
            self.E_hyd_MPa_360d = np.nan
            self.sigma_6_MPa_10e6 = np.nan
            self.b_slope_hyd = np.nan
            self.SN_logN_hyd = np.nan
            self.Rt_MPa = np.nan
        else:
            # Unknown material type, try to get E from any available column
            e_bit = safe_get('E_bit_MPa_15C_10Hz')
            e_hyd = safe_get('E_hyd_MPa_360d')
            e_unt = safe_get('E_untreated_MPa')
            self.E = e_bit if pd.notna(e_bit) else (e_hyd if pd.notna(e_hyd) else e_unt)
            self.Poisson_ratio = 0.35  # Default fallback for unknown material types
            # Set all type-specific properties to NaN
            self.E_bit_MPa_15C_10Hz = np.nan
            self.epsilon_6_strain_10e_6 = np.nan
            self.b_slope_bit = np.nan
            self.SN_logN_bit = np.nan
            self.E_hyd_MPa_360d = np.nan
            self.sigma_6_MPa_10e6 = np.nan
            self.b_slope_hyd = np.nan
            self.SN_logN_hyd = np.nan
            self.Rt_MPa = np.nan
            self.E_untreated_MPa = np.nan
        
        # Override Poisson_ratio if subgrade flag is set (Subgrade Pavement Foundation uses 0.35)
        if bool(subgrade):
            self.Poisson_ratio = 0.35
        self.cost_per_inch = self.calc_cost_per_inch()
        self.surface_code = 1 if bool(surface) else 0
        self.subgrade_code = 1 if bool(subgrade) else 0
        self.alkaline_code = 1 if bool(alkaline) else 0
        self.cost_per_sn = self.cost_per_inch / self.sn
        return None

    @property
    def thickness_cm(self):
        """Get thickness in centimeters (rounded up to nearest integer - ceiling)."""
        return int(math.ceil(self.thickness * self.INCH_TO_CM))
    
    @thickness_cm.setter
    def thickness_cm(self, value_cm):
        """Set thickness in centimeters (converts to inches internally)."""
        self.thickness = value_cm * self.CM_TO_INCH
    
    def set_thickness_cm(self, value_cm):
        """Set thickness in centimeters (converts to inches internally)."""
        self.thickness = value_cm * self.CM_TO_INCH
    
    def set_thickness_inches(self, value_inches):
        """Set thickness in inches (updates internal thickness)."""
        self.thickness = value_inches

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
    def __str__(self):
        if not self:
            return "Section: <empty>"

        lines = [f"Section: {len(self)} layer(s)"]
        for idx, layer in enumerate(self, start=1):
            name = getattr(layer, "name", f"Layer {idx}")
            mat_type = getattr(layer, "material_type", "Unknown")

            thickness_in = getattr(layer, "thickness", None)
            if thickness_in is not None:
                thickness_mm = thickness_in * 25.4
                thickness_str = f"{thickness_mm:.1f} mm ({thickness_in:.2f} in)"
            else:
                thickness_str = "N/A"

            sn = getattr(layer, "sn", None)
            sn_str = f"{sn:.3f}" if sn is not None else "N/A"

            lines.append(
                f"  {idx}. {name} [{mat_type}]  thickness: {thickness_str}, SN: {sn_str}"
            )

        total_sn = self.get_sn()
        lines.append(f"Total Structural Number (SN): {total_sn:.3f}")
        return "\n".join(lines)

    __repr__ = __str__
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
    
    # Remove materials that match the unmodifiable bottom layers
    # Need to be careful with indices when modifying the list during iteration
    indices_to_remove = []
    for i in range(len(material_list)):
        for j in range(unmodify_bottom_layers):
            # Check bounds before accessing sect[-j-1]
            if j < len(sect) and material_list[i].name == sect[-j-1].name:
                indices_to_remove.append(i)
                break  # Only remove once per material
    
    # Remove materials in reverse order to maintain correct indices
    for i in sorted(set(indices_to_remove), reverse=True):
        material_list.pop(i)
    
    # Check if we have enough materials left
    if len(material_list) == 0:
        raise ValueError(
            f"Cannot resolve section: All materials were removed. "
            f"unmodify_bottom_layers={unmodify_bottom_layers}, section_length={len(sect)}"
        )
    
    possible_sections = []
    for i in range(1, len(material_list)):
        possible_sections += make_possible_sections(material_list, i)
    
    # Check if we have any possible sections
    if len(possible_sections) == 0:
        raise ValueError(
            f"Cannot resolve section: No possible sections found. "
            f"material_list length={len(material_list)}, unmodify_bottom_layers={unmodify_bottom_layers}"
        )
    
    # Extend sections with unmodifiable bottom layers
    for section in possible_sections:
        # Ensure we don't go out of bounds when slicing sect
        start_idx = max(0, len(sect) - unmodify_bottom_layers)
        section.extend(deepcopy(sect[start_idx:]))
    
    modified_sections = []
    for section in possible_sections:
        modified_sections.append(modify_thickness(section, goal_sn, len(section) - unmodify_bottom_layers))
    
    modified_sections.sort(key=lambda s: section_cost(s, grade, embankment_cost, excavation_cost))
    valid_sections = [s for s in modified_sections if validate_section(s)]
    
    if len(valid_sections) == 0:
        raise ValueError(
            f"Cannot resolve section: No valid sections found after modification. "
            f"goal_sn={goal_sn}, unmodify_bottom_layers={unmodify_bottom_layers}"
        )
    
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

    solutions = solve(DF,sn_design,params['grade'],params['emb'],params['excv'],min_capas)
    if len(solutions) == 0:
        raise ValueError(
            f"No valid solutions found for traditional design. "
            f"SN required: {sn_design:.2f}, min_capas: {min_capas}, "
            f"materials available: {len(DF)}, "
            f"Try reducing min_capas to {len(DF)} or less."
        )
    dis_sect = solutions[0]
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
def save_design(section:Section,sim:int,period:int,startover,top_sn,top_cost=0):
    sect_info = {'simulation':sim,'period':period,'total_sn':section.get_sn(),'total_cost':section.totalCost,'top_cost':top_cost,'layer_materials':[],'layer_thicknesses':[],'layer_sns':[]}
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
    sect_info['startover'] = round(startover,2)
    sect_info['top_sn'] = round(top_sn,2)
    return sect_info
def evaluate_flexibility(params:dict, DF, random_esals=None, acumulated=None):
    """
    Function to evaluate the design flexibility\n
    params:dict = dictionary with all the parameters, keys\n
    DF:pd.DataFrame = DataFrame with the materials\n
    random_esals: np.array, optional
        Pre-generated monthly traffic array (size × n). If None, generates new traffic.
    acumulated: np.array, optional
        Pre-generated accumulated traffic array (size × n). If None, generates new traffic.
        If random_esals is provided, acumulated should also be provided.
    return array of size length of npv (one each for all the simulations)
    """
    # Create list to store section information (would be converted to DataFrame later)
    section_info = []
    
    # Generate traffic if not provided
    if random_esals is None or acumulated is None:
        random_esals, acumulated = make_simulated_transit(
            params['TPD'], params['vc'], params['cd'], params['size'], 
            params['n'], params['seedint'], params['mu_function'], params['sigma_function']
        )
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
    required_top_sn = projected_sn_step - reduce_base_step
    
    # Design top layers to meet the required SN and ignore the base layers SN
    step_design, actual_sn = resolve(DF, deepcopy(first_design), required_top_sn+base_sn, 
                                   len(first_design)-num_top_layers, params['grade'], params['emb'], params['excv'])
    
    
    # Store initial design information for all simulations
    for sim in range(params['size']):
        section_info.append(save_design(step_design, sim, 0,startover=base_sn,top_sn=actual_sn-base_sn+reduce_base_step))
    
        # Step 6: Process each simulation
    for sim in range(params['size']):
        acumulated_sim = acumulated[sim, :]
        current_step_design = deepcopy(step_design)  # Start with initial step design
        
        # Use calculate_sn_projected to get all required SNs at once
        sn_projected = calculate_sn_projected(acumulated_sim, params)
        acumulated_SN[sim,:] = sn_projected
        # Calculate current top layers SN capacity (as we modify it during the nested while is needed to recalculate it)
        base_sn = 0
        for i in range(num_top_layers, len(current_step_design)):
            base_sn += current_step_design[i].sn * current_step_design[i].thickness
        top_sn = actual_sn-base_sn+reduce_base_step
        previous_top_sn = 0
        previous_break = 0
        current_base_sn = base_sn
        # Loop until we reach the end of the simulation period
        while previous_break < params['n']:
            
            # Use calculate_break to find when top layers fail applied factor to fail before reaching 100%
            n_break = calculate_break(sn_projected, previous_top_sn+(top_sn)*params['factor'], 
                                    params['Reliavility'], params['Standard_Deviation'], 
                                    params['Delta_PSI'], params['Mr'])
            
            
            # If no break found within simulation period, we're done
            if n_break >= params['n']:
                break
            previous_top_sn += top_sn*params['factor']
            # Redesign based on current conditions
            p =np.polyfit(np.arange(0,n_break),sn_projected[0:n_break],1)
            required_sn = np.polyval(p,n_break+n_step) - previous_top_sn
            """annual_traffic = np.zeros((n_break-previous_break)//12)
            for i in range(n_break-previous_break):
                if i//12 < len(annual_traffic):
                    annual_traffic[i//12] += random_esals[sim,previous_break+i]
                else:
                    break
            m = W18_i_regression(annual_traffic)
            esal = pred_W18(params['TPD'],params['vc'],params['cd'],m,n_step//12)
            required_sn = solve_sn(params['Reliavility'], params['Standard_Deviation'], 
                               params['Delta_PSI'], params['Mr'], esal,actual_sn)"""
            
            if current_base_sn <= 0:
                # If base is exhausted, design completely new section using solve()
                current_step_design = solve(DF, required_sn, params['grade'], params['emb'], 
                                          params['excv'], params.get('min_capas', 1))[0]
                # Reset base SN since we have a new design
                # Recalculate base SN for new design
                base_sn = 0
                for i in range(num_top_layers, len(current_step_design)):
                    base_sn += current_step_design[i].sn * current_step_design[i].thickness
                actual_achieved_sn = section_sn(current_step_design)
                top_sn = actual_achieved_sn-base_sn+reduce_base_step
                section_info.append(save_design(current_step_design, sim, n_break,startover=0,top_sn=previous_top_sn+top_sn))
            else:
                current_base_sn -= reduce_base_step
                # Use resolve() to modify top layers only
                required_top_sn = required_sn - reduce_base_step
                current_step_design, actual_achieved_sn = resolve(DF, deepcopy(current_step_design), 
                                                                required_top_sn+base_sn, len(current_step_design)-num_top_layers, 
                                                                params['grade'], params['emb'], params['excv'])
                top_sn = actual_achieved_sn-base_sn+reduce_base_step
                top_cost = 0
                for i in range(num_top_layers):
                    top_cost += current_step_design[i].cost_per_inch * current_step_design[i].thickness
                section_info.append(save_design(current_step_design, sim, n_break,startover=current_base_sn,top_sn=previous_top_sn+top_sn,top_cost=top_cost))
            
            
            # Update previous break position
            previous_break = n_break
    
    # Create DataFrame with all section information
    results_df = pd.DataFrame(section_info)
    
    return results_df, acumulated_SN

def generate_shared_traffic(params:dict):
    """
    Generate traffic simulation that can be shared between evaluate_flexibility 
    and evaluate_flexibility_burmister for fair comparison.
    
    Parameters:
    -----------
    params : dict
        Dictionary with traffic simulation parameters:
        - TPD, vc, cd: Traffic parameters
        - size, n, seedint: Simulation parameters
        - mu_function, sigma_function: Growth functions
    
    Returns:
    --------
    tuple : (random_esals, acumulated)
        - random_esals: 2D array (size × n) of monthly traffic
        - acumulated: 2D array (size × n) of accumulated traffic
    """
    random_esals, acumulated = make_simulated_transit(
        params['TPD'], params['vc'], params['cd'], params['size'], 
        params['n'], params['seedint'], params['mu_function'], params['sigma_function']
    )
    return random_esals, acumulated


def calculate_break_ne(arr:np.array, ne_capacity:float)->int:
    """
    Find when the accumulated traffic (NE requirement) exceeds the section's NE capacity.
    
    Parameters:
    -----------
    arr : np.array
        Array with accumulated traffic (NE requirements)
    ne_capacity : float
        NE capacity of the section (from Burmister analysis)
    
    Returns:
    --------
    int : Index when capacity is exceeded, or len(arr) if it never fails
    """
    for i in range(len(arr)):
        if arr[i] >= ne_capacity:
            return i
    return len(arr)

def calculate_min_ne_from_layers(ne_limits: dict, num_top_layers: int, layer_type: str = 'base') -> float:
    """
    Calculate the minimum NE from either base layers or top layers.
    
    Parameters:
    -----------
    ne_limits : dict
        Dictionary of NE limits from calculate_pavement_ne, with keys like:
        - 'Subgrade Rutting'
        - 'Bituminous Fatigue (Layer 1: ...)'
        - 'Hydraulic Fatigue (Layer 2: ...)'
        - 'Granular Layer Rutting (Layer 1)'
    num_top_layers : int
        Number of top layers (layers 1 to num_top_layers are top layers)
    layer_type : str
        'base' to calculate minimum NE from base layers (includes subgrade)
        'top' to calculate minimum NE from top layers (excludes subgrade)
    
    Returns:
    --------
    float : Minimum NE value from the specified layer type (float('inf') if none found)
    """
    min_ne = float('inf')
    
    for layer_name in ne_limits.keys():
        if layer_name == 'Subgrade Rutting':
            # Always include Subgrade Rutting for base layers, exclude for top layers
            if layer_type == 'base':
                if ne_limits[layer_name] < min_ne:
                    min_ne = ne_limits[layer_name]
            continue
        
        # Extract layer number from keys like "Bituminous Fatigue (Layer 1: ...)" or "Hydraulic Fatigue (Layer 2: ...)"
        # Layer numbers in keys are 1-based, so Layer 1 = index 0, Layer 2 = index 1, etc.
        match = re.search(r'Layer (\d+)', layer_name)
        if match:
            layer_num = int(match.group(1))
            
            if layer_type == 'base':
                # Only include if layer number > num_top_layers (base layers)
                if layer_num > num_top_layers:
                    if ne_limits[layer_name] < min_ne:
                        min_ne = ne_limits[layer_name]
            elif layer_type == 'top':
                # Only include if layer number <= num_top_layers (top layers)
                if layer_num <= num_top_layers:
                    if ne_limits[layer_name] < min_ne:
                        min_ne = ne_limits[layer_name]
    
    return min_ne

def prepare_section_for_burmister(section: Section):
    """
    Prepare a section for Burmister analysis by rounding up thickness to nearest integer cm.
    Creates a deep copy and updates thickness_cm to ceiling value.
    
    Parameters:
    -----------
    section : Section
        Section to prepare
    
    Returns:
    --------
    Section : Deep copy of section with thicknesses rounded up to nearest integer cm
    """
    prepared_section = deepcopy(section)
    for layer in prepared_section:
        # Round up to nearest integer cm (ceiling)
        thickness_cm_ceil = math.ceil(layer.thickness * layer.INCH_TO_CM)
        layer.thickness_cm = thickness_cm_ceil
    return prepared_section

def design_section_for_ne(DF, target_ne:float, initial_section:Section=None, 
                          num_top_layers:int=3, grade:float=0.0, 
                          embankment_cost:float=0.0, excavation_cost:float=0.0,
                          subgrade_E:float=50.0, subgrade_v:float=0.35,
                          traffic_level:str='low', max_iterations:int=10,
                          thickness_increment:float=1.0):
    """
    Design a pavement section to meet a target NE requirement using Burmister analysis.
    
    Parameters:
    -----------
    DF : pd.DataFrame
        Materials DataFrame
    target_ne : float
        Target NE (number of equivalent axles) the section must support
    initial_section : Section, optional
        Initial section to modify (if None, creates new section)
    num_top_layers : int
        Number of top layers to modify (if initial_section provided)
    grade, embankment_cost, excavation_cost : float
        Earthwork parameters
    subgrade_E, subgrade_v : float
        Subgrade properties for Burmister analysis
    traffic_level : str
        Traffic level ('low' or 'high') for NE calculation
    max_iterations : int
        Maximum iterations for iterative design
    thickness_increment : float
        Thickness increment for iterative adjustment (inches)
    min_layers : int, optional
        Minimum number of layers required (default: num_top_layers + 1)
    
    Returns:
    --------
    tuple : (designed_section, achieved_ne, ne_limits_dict)
    """
    try:
        import burmister
        from burm_FDM_Ne import calculate_pavement_ne
    except ImportError:
        raise ImportError("burmister and burm_FDM_Ne modules are required for NE-based design")
    
    # Set minimum layers
    min_layers = num_top_layers + 1
    material_list = make_material_list(DF)
    
    # Make a deep copy of the initial section
    initial_section = deepcopy(initial_section)
    
    # Ensure minimum layers - if section has fewer layers, add layers
    if len(initial_section) < min_layers:
        # Add additional layers from material list
        material_list_sorted = sorted(material_list, key=lambda x: x.cost_per_sn)
        layers_to_add = min_layers - len(initial_section)
        for i in range(layers_to_add):
            # Add cheapest material that's not already in section
            for mat in material_list_sorted:
                if mat.name not in [layer.name for layer in initial_section]:
                    new_layer = deepcopy(mat)
                    new_layer.thickness = new_layer.min_lift
                    initial_section.insert(-1, new_layer)  # Insert before last layer (subgrade)
                    break
    
    # Iteratively adjust thickness to meet NE requirement
    best_section = deepcopy(initial_section)
    #set the top layers to minimum thickness
    for i in range(num_top_layers):
        best_section[i].thickness = best_section[i].min_lift
    for iteration in range(max_iterations):
        # Prepare section for Burmister (round up thickness to nearest integer cm)
        prepared_section = prepare_section_for_burmister(best_section)
        # Run Burmister analysis
        burmister_results = burmister.analyze_section(
            prepared_section,
            subgrade_E=subgrade_E,
            subgrade_v=subgrade_v,
            plot_geometry=False
        )
        
        # Calculate NE capacity (use prepared_section for consistency)
        final_ne, criterion, ne_limits = calculate_pavement_ne(
            prepared_section, burmister_results, traffic_level=traffic_level
        )
        
        top_layers_ne = calculate_min_ne_from_layers(ne_limits, num_top_layers, layer_type='top')
        # Check if we've met the requirement
        if top_layers_ne >= target_ne:
            return prepared_section, final_ne, ne_limits
        
        # If not, increase thickness of modifiable layers
        if top_layers_ne < target_ne:
            # Calculate how much to increase (proportional)
            increase_factor = (target_ne / top_layers_ne) if top_layers_ne > 0 else 2.0
            increase_factor = min(increase_factor, 1.5)  # Limit to 50% increase per iteration
            
            # Increase thickness of top layers
            layers_to_modify = min(num_top_layers, len(best_section))
            for i in range(layers_to_modify):
                new_thickness = best_section[i].thickness * increase_factor
                # Ensure within limits
                new_thickness = max(best_section[i].min_lift, 
                                    min(new_thickness, best_section[i].max_lift))
                best_section[i].thickness = new_thickness
            
            top_layers_ne = calculate_min_ne_from_layers(ne_limits, num_top_layers, layer_type='top')
    # Return final prepared section (rounded to cm)
    final_prepared_section = prepare_section_for_burmister(best_section)
    return final_prepared_section, final_ne, ne_limits

def calculate_section_with_thickness_modifications(section: Section,
                                                   thickness_modifications: list,
                                                   subgrade_E: float = 50.0,
                                                   subgrade_v: float = 0.35,
                                                   traffic_level: str = 'low'):
    """
    Calculate a pavement section with manually specified thickness modifications.
    
    This function applies thickness modifications to a section, runs Burmister analysis,
    and calculates NE capacity. It does not perform any iterative design or goal-seeking.
    
    Parameters:
    -----------
    section : Section
        The pavement section to modify (will be deep copied)
    thickness_modifications : list
        List of thickness values to apply to each layer (in same order as section).
        Length should match number of layers in section (excluding subgrade).
        Values are absolute thicknesses in centimeters (cm).
    subgrade_E : float
        Subgrade Young's modulus in MPa (default: 50.0)
    subgrade_v : float
        Subgrade Poisson ratio (default: 0.35)
    traffic_level : str
        Traffic level for NE calculation: 'low' or 'high' (default: 'low')
    
    Returns:
    --------
    tuple : (designed_section, final_ne, ne_limits)
        - designed_section: Section with modified thicknesses
        - final_ne: Overall NE capacity (minimum from all criteria)
        - ne_limits: Dictionary of NE limits for each failure criterion
    """
    try:
        import burmister
        from burm_FDM_Ne import calculate_pavement_ne
    except ImportError:
        raise ImportError("burmister and burm_FDM_Ne modules are required")
    
    # Make a deep copy of the section
    designed_section = deepcopy(section)
    
    # Apply thickness modifications
    num_layers = len(designed_section) - 1  # Exclude subgrade
    if len(thickness_modifications) != num_layers:
        raise ValueError(f"Number of thickness modifications ({len(thickness_modifications)}) "
                        f"must match number of layers ({num_layers})")
    
    # Conversion factor: 1 cm = 0.393701 inches
    CM_TO_INCH = 0.393701
    
    for i, new_thickness_cm in enumerate(thickness_modifications):
        # Convert from cm to inches (Section stores thickness in inches)
        new_thickness_inches = new_thickness_cm * CM_TO_INCH
        # Ensure thickness is within layer limits (min_lift and max_lift are in inches)
        new_thickness_inches = max(designed_section[i].min_lift,
                                  min(new_thickness_inches, designed_section[i].max_lift))
        designed_section[i].thickness = new_thickness_inches
    
    # Prepare section for Burmister (round up thickness to nearest integer cm)
    prepared_section = prepare_section_for_burmister(designed_section)
    # Run Burmister analysis
    burmister_results = burmister.analyze_section(
        prepared_section,
        subgrade_E=subgrade_E,
        subgrade_v=subgrade_v,
        plot_geometry=False
    )
    
    # Calculate NE capacity
    final_ne, criterion, ne_limits = calculate_pavement_ne(
        prepared_section, burmister_results, traffic_level=traffic_level
    )
    
    return prepared_section, final_ne, ne_limits

def save_design_ne(section:Section, sim:int, period:int, ne_capacity:float, top_cost:float,
                   ne_dict:{}):
    """
    Save design information for NE-based design.
    
    Parameters:
    -----------
    section : Section
        The designed section
    sim : int
        Simulation ID
    period : int
        Period (month) when design was created
    ne_capacity : float
        Total NE capacity of the section
    ne_top : float
        NE capacity provided by top layers
    top_cost : float
        Cost of top layers
    
    Returns:
    --------
    dict : Dictionary with design information
    """
    sect_info = {
        'simulation': sim,
        'period': period,
        'ne_capacity': round(ne_capacity, 2),
        'ne_dict': ne_dict,
        'total_cost': round(section.totalCost, 2),
        'top_cost': round(top_cost, 2),
        'layer_materials': [],
        'layer_thicknesses': [],
        'layer_sns': []
    }
    
    # Store layer information
    materials = []
    thicknesses = []
    sns = []
    for layer in section:
        materials.append(layer.name)
        thicknesses.append(round(layer.thickness, 2))
        sns.append(round(layer.sn, 2))
    
    sect_info['layer_materials'] = materials
    sect_info['layer_thicknesses'] = thicknesses
    sect_info['layer_sns'] = sns
    
    return sect_info

def evaluate_flexibility_burmister(params:dict, DF, subgrade_E:float=50.0, 
                                   subgrade_v:float=0.35, traffic_level:str='low',
                                   random_esals=None, acumulated=None, progress_callback=None,
                                   interactive_callback=None, save_file:str=None):
    """
    Evaluate design flexibility using French Design Method with Burmister analysis.
    Uses NE (number of equivalent axles) instead of SN.
    
    Parameters:
    -----------
    params : dict
        Dictionary with all parameters (same as evaluate_flexibility):
        - TPD, vc, cd: Traffic parameters
        - size, n, seedint: Simulation parameters
        - step: Intervention interval (months)
        - capas: Number of top layers to modify
        - factor: Life factor (0-1)
        - mu_function, sigma_function: Growth functions
        - grade, emb, excv: Earthwork parameters
        - min_capas: Minimum layers in design
    DF : pd.DataFrame
        Materials DataFrame
    subgrade_E : float
        Subgrade Young's modulus in MPa (default: 50.0)
    subgrade_v : float
        Subgrade Poisson ratio (default: 0.35)
    traffic_level : str
        Traffic level for NE calculation: 'low' or 'high' (default: 'low')
    random_esals : np.array, optional
        Pre-generated monthly traffic array (size × n). If None, generates new traffic.
        Use this to share the same traffic simulation with evaluate_flexibility.
    acumulated : np.array, optional
        Pre-generated accumulated traffic array (size × n). If None, generates new traffic.
        If random_esals is provided, acumulated should also be provided.
        Use this to share the same traffic simulation with evaluate_flexibility.
    save_file : str, optional
        Path to save checkpoint file. If provided, saves results after each simulation
        and can resume from checkpoint if interrupted. Should be base filename without extension.
        Will create two files: {save_file}.csv and {save_file}_accumulated.csv
    
    Returns:
    --------
    tuple : (results_df, acumulated_NE)
        - results_df: DataFrame with redesign events and NE capacities
        - acumulated_NE: 2D array (size × n) of projected NE requirements
    """
    try:
        import burmister
        from burm_FDM_Ne import calculate_pavement_ne
    except ImportError:
        raise ImportError("burmister and burm_FDM_Ne modules are required")
    
    # Create list to store section information
    section_info = []
    
    # Checkpoint/resume functionality
    completed_sims = set()
    if save_file is not None:
        # Check if checkpoint files exist
        results_checkpoint = f"{save_file}.csv"
        accum_checkpoint = f"{save_file}_accumulated.csv"
        
        if os.path.exists(results_checkpoint):
            # Load existing results
            try:
                existing_results = pd.read_csv(results_checkpoint)
                if 'simulation' in existing_results.columns:
                    completed_sims = set(existing_results['simulation'].unique())
                    # Convert existing results to section_info format
                    for _, row in existing_results.iterrows():
                        section_info.append(row.to_dict())
                    print(f"Loaded checkpoint: {len(completed_sims)} simulations already completed")
            except Exception as e:
                print(f"Warning: Could not load checkpoint file {results_checkpoint}: {e}")
                completed_sims = set()
        
    # Generate traffic simulations if not provided (allows sharing traffic with evaluate_flexibility)
    # Only generate if acumulated is None (since that's what we actually use)
    if acumulated is None:
        random_esals, acumulated = make_simulated_transit(
            params['TPD'], params['vc'], params['cd'], params['size'], 
            params['n'], params['seedint'], params['mu_function'], params['sigma_function']
        )
    
    n_step = params['step']
    num_top_layers = params.get('capas', 3)
    
    # Calculate projected NE requirements (same as accumulated traffic for French design)
    acumulated_NE = np.zeros((params['size'], params['n']))
    
    # Load accumulated NE from checkpoint if available
    if save_file is not None:
        accum_checkpoint = f"{save_file}_accumulated.csv"
        if os.path.exists(accum_checkpoint):
            try:
                existing_accum = pd.read_csv(accum_checkpoint, header=0).values
                if existing_accum.shape[0] == params['size'] and existing_accum.shape[1] == params['n']:
                    acumulated_NE = existing_accum
                    print(f"Loaded accumulated NE from checkpoint: {accum_checkpoint}")
            except Exception as e:
                print(f"Warning: Could not load accumulated checkpoint: {e}")
    
    # Initialize from traffic if not loaded from checkpoint
    if acumulated_NE.sum() == 0:
        for sim in range(params['size']):
            acumulated_NE[sim, :] = deepcopy(acumulated[sim, :])
    
    # Step 1: Generate traditional design for full period
    first_design, _ , m = traditional_design(params, DF, params.get('min_capas', 4))
    # Prepare section for Burmister (round up thickness to nearest integer cm)
    prepared_first_design = prepare_section_for_burmister(first_design)
    first_ne, breaking_criterion_Ne, dict_limits_Ne = calculate_pavement_ne(prepared_first_design, burmister.analyze_section(prepared_first_design, subgrade_E=subgrade_E, subgrade_v=subgrade_v, plot_geometry=False), traffic_level=traffic_level)
    # Step 2: Calculate base layers Ne (bottom layers that won't be modified frequently
    num_top_layers = params.get('capas', 3)  # Number of top layers to modify
    base_ne = calculate_min_ne_from_layers(dict_limits_Ne, num_top_layers, layer_type='base')
    #initial_design, initial_ne, _ = (first_design, first_ne, dict_limits_Ne)
    print(first_design)
    # Step 3: Calculate step design for n_step period
    projected_w18_step = pred_W18(params['TPD'], params['vc'], params['cd'], m, n_step//12)
    
    # Calculate top NE capacity
    top_ne = calculate_min_ne_from_layers(dict_limits_Ne, num_top_layers, layer_type='top')
    
    # Step 4: (NOT Needed) calculate reduction in base Ne as it degrades all at the same time.
    total_steps = params['n'] // n_step
    # Design top layers to meet the required Ne and ignore the base layers Ne
    initial_design, initial_ne, ne_intial_limits_dict = design_section_for_ne(DF,projected_w18_step, deepcopy(first_design), 
                                   num_top_layers, params['grade'], params['emb'], params['excv'])# Store initial design for all simulations
    
    
    # Step 6: Process each simulation
    total_sims = params['size']
    for sim in range(total_sims):
        # Skip if simulation already completed (checkpoint resume)
        if sim in completed_sims:
            if progress_callback:
                progress_callback(sim + 1, total_sims)
            continue
        
        # Report progress if callback provided
        if progress_callback:
            progress_callback(sim + 1, total_sims)
        
        # Store section info for this simulation
        sim_section_info = []
        
        acumulated_ne_sim = acumulated_NE[sim, :]
        current_design = deepcopy(initial_design)
        
        # Recalculate base NE for current design (dynamically based on current design)
        current_base_ne = deepcopy(base_ne)
        current_top_ne = deepcopy(top_ne)
        
        previous_top_ne = 0.0
        previous_top_sn = 0.0  # Track accumulated top SN for AASHTO redesign
        previous_break = 0
        
        # Loop until end of simulation period
        while previous_break < params['n']:
            # Calculate current total NE capacity}
            
            # Find when capacity is exceeded
            n_break = calculate_break_ne(acumulated_ne_sim, current_top_ne+previous_top_ne)
            
            # If no break found, we're done
            if n_break >= params['n']:
                break
            
            # Update previous top NE
            previous_top_ne += current_top_ne * params['factor']
            
            # Calculate required NE for next intervention period
            # Use linear extrapolation from current trend
            if n_break < params['n']:
                target_ne = acumulated_ne_sim[n_break]
            else:
                target_ne = acumulated_ne_sim[-1]
            
            # Adjust for base layer deterioration
            current_base_ne -= acumulated_ne_sim[n_break]
            
            # Redesign
            if current_base_ne <= 0:
                # Base is exhausted - design completely new section using AASHTO methods
                # Reset previous_top_sn since we're starting completely fresh
                previous_top_sn = 0.0
                
                # Calculate SN requirements from accumulated traffic (similar to evaluate_flexibility)
                acumulated_sim_for_sn = acumulated[sim, :]  # Use original accumulated traffic for SN calculation
                sn_projected = calculate_sn_projected(acumulated_sim_for_sn, params)
                
                # Use linear extrapolation to calculate required SN (full SN needed, no previous_top_sn subtraction)
                p = np.polyfit(np.arange(0, n_break), sn_projected[0:n_break], 1)
                required_sn = np.polyval(p, n_break + n_step)
                
                # Design completely new section using solve() (AASHTO method, similar to evaluate_flexibility)
                new_design = solve(DF, required_sn, params['grade'], params['emb'], 
                                  params['excv'], params.get('min_capas', 1))[0]
                
                # Recalculate base SN for new design
                base_sn = 0
                for i in range(num_top_layers, len(new_design)):
                    base_sn += new_design[i].sn * new_design[i].thickness
                
                # Calculate reduction in base SN per step (gradual deterioration)
                total_steps = params['n'] // n_step
                reduce_base_step = base_sn / total_steps if total_steps > 0 else 0
                
                actual_achieved_sn = section_sn(new_design)
                top_sn = actual_achieved_sn - base_sn + reduce_base_step
                
                # Update previous_top_sn for next iteration
                previous_top_sn += top_sn * params['factor']
                
                # Prepare redesigned section for Burmister analysis
                prepared_new_design = prepare_section_for_burmister(new_design)
                
                # Analyze redesigned section with Burmister to get NE information
                burmister_results = burmister.analyze_section(
                    prepared_new_design, 
                    subgrade_E=subgrade_E,
                    subgrade_v=subgrade_v, 
                    plot_geometry=False
                )
                achieved_ne, _, ne_limits = calculate_pavement_ne(
                    prepared_new_design, burmister_results, traffic_level=traffic_level
                )
                
                current_design = prepared_new_design
                
                # Recalculate base and top NE after redesign
                current_base_ne = calculate_min_ne_from_layers(ne_limits, num_top_layers, layer_type='base')
                current_top_ne = calculate_min_ne_from_layers(ne_limits, num_top_layers, layer_type='top')
                
                # Calculate top cost
                top_cost = 0.0
                for i in range(len(current_design)):
                    top_cost += current_design[i].cost_per_inch * current_design[i].thickness
                
                sim_section_info.append(save_design_ne(
                    current_design, sim, n_break, achieved_ne, top_cost, ne_limits
                ))
            else:
                # Partial redesign (modify top layers only)
                # Check if interactive callback is provided
                if interactive_callback is not None:
                    print(f"[Burmister] Interactive mode enabled - section failed at month {n_break}")
                    print(f"[Burmister] Target NE required: {target_ne:,.0f}")
                    # Get current NE limits for the section
                    try:
                        import burmister
                        from burm_FDM_Ne import calculate_pavement_ne
                        prepared_current = prepare_section_for_burmister(current_design)
                        _, _, current_ne_limits = calculate_pavement_ne(
                            prepared_current, 
                            burmister.analyze_section(prepared_current, subgrade_E=subgrade_E, 
                                                    subgrade_v=subgrade_v, plot_geometry=False),
                            traffic_level=traffic_level
                        )
                        
                        print(f"[Burmister] Current section NE limits calculated, calling interactive callback...")
                        # Call interactive callback
                        modified_section = interactive_callback(
                            current_design, current_ne_limits, target_ne,
                            subgrade_E, subgrade_v, traffic_level
                        )
                        print(f"[Burmister] Interactive callback returned: {modified_section is not None}")
                        
                        if modified_section is not None:
                            # Use user-modified section
                            prepared_modified = prepare_section_for_burmister(modified_section)
                            burmister_results = burmister.analyze_section(
                                prepared_modified, subgrade_E=subgrade_E,
                                subgrade_v=subgrade_v, plot_geometry=False
                            )
                            achieved_ne, _, ne_limits = calculate_pavement_ne(
                                prepared_modified, burmister_results, traffic_level=traffic_level
                            )
                            current_design = prepared_modified
                        else:
                            # User cancelled, use automatic redesign
                            modified_design, achieved_ne, ne_limits = design_section_for_ne(
                                DF, target_ne, initial_section=current_design,
                                num_top_layers=num_top_layers, grade=params['grade'],
                                embankment_cost=params['emb'], excavation_cost=params['excv'],
                                subgrade_E=subgrade_E, subgrade_v=subgrade_v, traffic_level=traffic_level
                            )
                            current_design = modified_design
                    except Exception as e:
                        # Fallback to automatic redesign on error
                        print(f"Error in interactive mode: {e}, using automatic redesign")
                        modified_design, achieved_ne, ne_limits = design_section_for_ne(
                            DF, target_ne, initial_section=current_design,
                            num_top_layers=num_top_layers, grade=params['grade'],
                            embankment_cost=params['emb'], excavation_cost=params['excv'],
                            subgrade_E=subgrade_E, subgrade_v=subgrade_v, traffic_level=traffic_level
                        )
                        current_design = modified_design
                else:
                    # Automatic redesign using AASHTO methods
                    # Calculate SN requirements from accumulated traffic (similar to evaluate_flexibility)
                    acumulated_sim_for_sn = acumulated[sim, :]  # Use original accumulated traffic for SN calculation
                    sn_projected = calculate_sn_projected(acumulated_sim_for_sn, params)
                    
                    # Use linear extrapolation to calculate required SN (same as evaluate_flexibility)
                    p = np.polyfit(np.arange(0, n_break), sn_projected[0:n_break], 1)
                    required_sn = np.polyval(p, n_break + n_step) - previous_top_sn
                    
                    # Calculate current base SN from current design
                    current_base_sn = 0
                    for i in range(num_top_layers, len(current_design)):
                        current_base_sn += current_design[i].sn * current_design[i].thickness
                    
                    # Calculate reduction in base SN per step (gradual deterioration)
                    total_steps = params['n'] // n_step
                    reduce_base_step = current_base_sn / total_steps if total_steps > 0 else 0
                    
                    # Adjust for base layer deterioration
                    current_base_sn -= reduce_base_step
                    
                    # Calculate required top SN (accounting for base deterioration)
                    required_top_sn = required_sn - reduce_base_step
                    
                    # Use resolve() to redesign top layers using AASHTO methods
                    modified_design, actual_achieved_sn = resolve(
                        DF, deepcopy(current_design), 
                        required_top_sn + current_base_sn, 
                        len(current_design) - num_top_layers, 
                        params['grade'], params['emb'], params['excv']
                    )
                    
                    # Calculate top SN achieved (for tracking)
                    top_sn_achieved = actual_achieved_sn - current_base_sn + reduce_base_step
                    
                    # Update previous_top_sn for next iteration
                    previous_top_sn += top_sn_achieved * params['factor']
                    
                    # Prepare redesigned section for Burmister analysis
                    prepared_modified = prepare_section_for_burmister(modified_design)
                    
                    # Analyze redesigned section with Burmister to get NE information
                    burmister_results = burmister.analyze_section(
                        prepared_modified, 
                        subgrade_E=subgrade_E,
                        subgrade_v=subgrade_v, 
                        plot_geometry=False
                    )
                    achieved_ne, _, ne_limits = calculate_pavement_ne(
                        prepared_modified, burmister_results, traffic_level=traffic_level
                    )
                    
                    current_design = prepared_modified
                
                # Recalculate top NE
                current_top_ne =calculate_min_ne_from_layers(ne_limits, num_top_layers, layer_type='top')
                
                # Calculate top cost
                top_cost = 0.0
                layers_to_cost = min(num_top_layers, len(current_design))
                for i in range(layers_to_cost):
                    top_cost += current_design[i].cost_per_inch * current_design[i].thickness
                
                sim_section_info.append(save_design_ne(
                    current_design, sim, n_break,
                    ne_capacity=achieved_ne, ne_dict=ne_limits, top_cost=top_cost
                ))
            
            # Update previous break position
            previous_break = n_break
        
        # Add this simulation's results to main list
        section_info.extend(sim_section_info)
        
        # Save checkpoint after each simulation completes
        if save_file is not None:
            try:
                # Save results - rewrite entire file to ensure consistency
                results_checkpoint = f"{save_file}.csv"
                if len(section_info) > 0:
                    # Create DataFrame from all section_info (includes previously loaded + new)
                    all_results_df = pd.DataFrame(section_info)
                    all_results_df.to_csv(results_checkpoint, mode='w', header=True, index=False)
                
                # Save accumulated NE data (overwrite each time with complete data)
                accum_checkpoint = f"{save_file}_accumulated.csv"
                accum_df = pd.DataFrame(acumulated_NE)
                accum_df.to_csv(accum_checkpoint, index=False, header=True)
                
                completed_count = len(completed_sims) + 1
                print(f"Checkpoint saved: Simulation {sim} completed ({completed_count}/{total_sims})")
            except Exception as e:
                print(f"Warning: Could not save checkpoint after simulation {sim}: {e}")
    
    # Create DataFrame with all section information
    results_df = pd.DataFrame(section_info)
    
    return results_df, acumulated_NE