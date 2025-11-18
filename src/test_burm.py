import csv
import sys
from copy import deepcopy
from pathlib import Path

# --- Import Functions from Your Modules ---
# We assume all .py files and the .csv are in the same directory

try:
    # From Logica.py
    from Logica import Section, Layer
    # From burm_FDM_Ne.py
    from burm_FDM_Ne import load_materials_from_csv, calculate_pavement_ne
    # From burmister.py
    from burmister import analyze_section
except ImportError as e:
    print(f"Error: Could not import necessary modules.")
    print(f"Make sure 'Logica.py', 'burmister.py', and 'burm_FDM_Ne.py' are in the same directory.")
    print(f"Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

APPENDIX7_CSV = "appendix7_materials.csv"
APPENDIX7_FIELDS = [
    "mat_name", "sn", "material_type", "cost", "density", "unit",
    "surface", "subgrade", "alkaline", "min", "max",
    "E_bit_MPa_15C_10Hz", "epsilon_6_strain_10e-6", "b_slope_bit", "SN_logN_bit",
    "E_hyd_MPa_360d", "sigma_6_MPa_10e6", "b_slope_hyd", "SN_logN_hyd",
    "Rt_MPa", "E_untreated_MPa"
]
APPENDIX7_ROWS = [
    {
        "mat_name": "Bituminous Concrete (BB)",
        "sn": 0.44,
        "material_type": "Bituminous",
        "cost": 0.0,
        "density": 0.0,
        "unit": "sqyd",
        "surface": 1,
        "subgrade": 0,
        "alkaline": 0,
        "min": 2.0,
        "max": 12.0,
        "E_bit_MPa_15C_10Hz": 5400.0,
        "epsilon_6_strain_10e-6": 100.0,
        "b_slope_bit": -0.2,
        "SN_logN_bit": 0.25,
        "E_hyd_MPa_360d": "",
        "sigma_6_MPa_10e6": "",
        "b_slope_hyd": "",
        "SN_logN_hyd": "",
        "Rt_MPa": "",
        "E_untreated_MPa": ""
    },
    {
        "mat_name": "Slag Bound Aggregates (GLg)",
        "sn": 0.14,
        "material_type": "Hydraulic",
        "cost": 0.0,
        "density": 0.0,
        "unit": "sqyd",
        "surface": 0,
        "subgrade": 0,
        "alkaline": 0,
        "min": 4.0,
        "max": 32.0,
        "E_bit_MPa_15C_10Hz": "",
        "epsilon_6_strain_10e-6": "",
        "b_slope_bit": "",
        "SN_logN_bit": "",
        "E_hyd_MPa_360d": 15000.0,
        "sigma_6_MPa_10e6": 0.6,
        "b_slope_hyd": -0.08,
        "SN_logN_hyd": 1.0,
        "Rt_MPa": "",
        "E_untreated_MPa": ""
    },
    {
        "mat_name": "Treated Capping Layer (CdF)",
        "sn": 0.18,
        "material_type": "Hydraulic",
        "cost": 0.0,
        "density": 0.0,
        "unit": "sqyd",
        "surface": 0,
        "subgrade": 0,
        "alkaline": 0,
        "min": 4.0,
        "max": 24.0,
        "E_bit_MPa_15C_10Hz": "",
        "epsilon_6_strain_10e-6": "",
        "b_slope_bit": "",
        "SN_logN_bit": "",
        "E_hyd_MPa_360d": 11111.1,
        "sigma_6_MPa_10e6": 0.3,
        "b_slope_hyd": -0.1,
        "SN_logN_hyd": 1.0,
        "Rt_MPa": "",
        "E_untreated_MPa": ""
    }
]


def ensure_appendix7_csv():
    """
    Create the Appendix 7 materials CSV if it does not already exist.
    """
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / APPENDIX7_CSV

    if csv_path.exists():
        return APPENDIX7_CSV

    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=APPENDIX7_FIELDS)
        writer.writeheader()
        for row in APPENDIX7_ROWS:
            writer.writerow(row)

    return APPENDIX7_CSV

def run_appendix_7_test():
    """
    Runs a test mimicking the FDM Appendix 7 example (pages 246-249)
    for a semi-rigid pavement with a treated capping layer.
    
    This function defines materials based on the PDF,
    defines the pavement section, and calculates its design life (NE).
    """
    
    print("--- Starting FDM Appendix 7 Test (PDF Example) ---")
    
    # 1. Load Materials from default.csv
    # We still load this, but we will create our own materials for this test.
    print("1. Loading materials from 'default.csv' (for reference)...")
    try:
        materials_db = load_materials_from_csv('default.csv')
        print(f"   Loaded {len(materials_db)} materials.")
    except FileNotFoundError:
        print("   Warning: 'default.csv' not found. Continuing with manually defined materials.")
    except Exception as e:
        print(f"   Warning: Error loading 'default.csv': {e}. Continuing with manually defined materials.")

    # Ensure Appendix 7 CSV exists and load materials from it
    appendix_csv = ensure_appendix7_csv()
    appendix_materials = load_materials_from_csv(appendix_csv)

    # ---
    # 2. !!! DEFINE MATERIALS FROM APPENDIX 7 PDF !!!
    # ---
    # We are creating these materials from scratch based on the tables
    # in the PDF (pages 247-248).
    
    print("\n2. Defining material properties from PDF (Appendix 7)...")
    
    try:
        # Retrieve layers from CSV and adjust thicknesses (convert mm to inches)
        layer1_BB = deepcopy(appendix_materials["Bituminous Concrete (BB)"])
        layer1_BB.thickness = 140.0 / 25.4  # 14 cm

        layer2_GLg = deepcopy(appendix_materials["Slag Bound Aggregates (GLg)"])
        layer2_GLg.thickness = 400.0 / 25.4  # 40 cm

        layer3_CdF = deepcopy(appendix_materials["Treated Capping Layer (CdF)"])
        layer3_CdF.thickness = 200.0 / 25.4  # 20 cm

        # Update Poisson ratios explicitly when needed
        layer1_BB.Poisson_ratio = 0.35
        layer2_GLg.Poisson_ratio = 0.25
        layer3_CdF.Poisson_ratio = 0.25
        
        # Subgrade Properties (for Burmister analysis)
        # AR1 class subgrade, standard E = 50 MPa
        subgrade_E_MPa = 50.0
        subgrade_v = 0.35
        
        # Traffic Level (for FDM NE calculation)
        # PDF allowable strain (294e-6) matches 'medium' traffic
        traffic_level = 'medium'

        print(f"   Layer 1: {layer1_BB.name} ({layer1_BB.thickness * 25.4:.1f} mm)")
        print(f"   Layer 2: {layer2_GLg.name} ({layer2_GLg.thickness * 25.4:.1f} mm)")
        print(f"   Layer 3: {layer3_CdF.name} ({layer3_CdF.thickness * 25.4:.1f} mm)")
        print(f"   Subgrade E: {subgrade_E_MPa} MPa (AR1 Platform)")
        print(f"   Traffic Level: {traffic_level}")
        
    except Exception as e:
        print(f"   Error defining layers: {e}")
        return

    # --- End of Appendix 7 material definitions ---


    # 3. Create Pavement Section
    print("\n3. Creating pavement section...")
    section = Section()
    section.append(layer1_BB)
    section.append(layer2_GLg)
    section.append(layer3_CdF)
    print(section)

    # 4. Run Burmister Analysis
    print("\n4. Running Burmister elastic analysis...")
    print("   (Using bonded=False to model sliding interface at L2/L3)")
    try:
        burmister_results = analyze_section(
            section, 
            subgrade_E=subgrade_E_MPa, 
            subgrade_v=subgrade_v,
            bonded=False,  # PDF p.249 specifies "sliding" interface
            plot_geometry=False
        )
        print("   Analysis completed successfully!")
        print(f"   Maximum stresses/strains extracted.")
        max_results_df = burmister_results['max_values']
    except Exception as e:
        print(f"   Error in Burmister analysis: {e}")
        return

    # 5. Intermediate Results Verification (Compare with PDF Table A.7.1)
    print("\n5. Intermediate Stresses/Strains (for comparison with PDF Table A.7.1)")
    try:
        # Get stresses (SigmaX is radial tensile stress)
        stress_L2_bottom = max_results_df.loc['Layer 2 Bottom', 'SigmaX']
        stress_L3_bottom = max_results_df.loc['Layer 3 Bottom', 'SigmaX']
        
        # Get strains (already in microstrain from burmister)
        strain_L1_bottom = max_results_df.loc['Layer 1 Bottom', 'EpsX']
        strain_subgrade_top = max_results_df.loc['Subgrade Top', 'EpsZ']

        print(f"   - PDF Value (L2/GLg Bottom): 0.43 MPa")
        print(f"   - Calculated Stress (L2/GLg Bottom): {stress_L2_bottom:.3f} MPa")
        
        print(f"   - PDF Value (L3/CdF Bottom): 0.11 MPa")
        print(f"   - Calculated Stress (L3/CdF Bottom): {stress_L3_bottom:.3f} MPa")
        
        print(f"\n   Other calculated values:")
        print(f"   - Bituminous Strain (L1/BB Bottom): {strain_L1_bottom:.1f} 10e-6")
        print(f"   - Subgrade Strain (Top): {strain_subgrade_top:.1f} 10e-6")

    except KeyError as e:
        print(f"   Error extracting intermediate results: {e}")
    except Exception as e:
        print(f"   Error during intermediate result display: {e}")


    # 6. Calculate NE Design Life (FDM)
    print("\n6. Calculating FDM allowable NE (design life)...")
    try:
        final_ne, criterion, ne_limits = calculate_pavement_ne(
            section, 
            burmister_results, 
            traffic_level=traffic_level
        )
        
        if final_ne is not None:
            print("\n   --- FINAL RESULTS ---")
            print(f"   {'Criterion':<50} {'Allowable NE (axles)':>25}")
            print(f"   {'-'*50} {'-'*25}")
            
            # Sort results by NE value for clarity
            for crit, val in sorted(ne_limits.items(), key=lambda item: item[1]):
                marker = " <-- GOVERNING" if crit == criterion else ""
                print(f"   {crit:<50} {val:>25,.0f}{marker}")
            
            print(f"\n   Governing Criterion: {criterion}")
            print(f"   Final Pavement Design Life (NE): {final_ne:,.0f} axles")
            print("   (Compare to PDF example design traffic NE: 23,340,000)")
            print("   ---------------------")
            
        else:
            print("   Error: Could not determine final NE.")
            
    except Exception as e:
        print(f"   Error in FDM NE calculation: {e}")
        return

if __name__ == "__main__":
    run_appendix_7_test()