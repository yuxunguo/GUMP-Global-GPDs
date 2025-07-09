#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 13:20:00 2025

@author: fpaslan
"""


                                        ###### CROSS SECTION ########
        
import gpddatabase
import pandas as pd
import numpy as np

"""

  DATASET 1
  uuid: TKhscLcB
  collaboration: HallA
  type: DVCS
  observables: CrossSectionUU  
  
"""
# Constants
E_lepton = 3.355  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('TKhscLcB')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_f = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) * 1e-3

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "f": obs_dict[obs_names[0]] * 1e-3,   # Converting pb to nb],
        "delta_f": delta_f,
        "pol": "UU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_excel("1_CS_TKhscLcB.xlsx", index=False, header=False)

print("Final file written to 1_CS_TKhscLcB.xlsx")

#############################################################################################################

"""

  DATASET 2
  uuid: AtY8o7Ej
  collaboration: HallA
  type: DVCS
  observables: CrossSectionUU 
  
"""
# Constants
E_lepton = 5.7572  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('AtY8o7Ej')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows_uu = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))

    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Statistical uncertainty (symmetric)
    stat0_unc = point.get_observables_stat_uncertainties().get_uncertainty(0)
    stat0_val = stat0_unc.get_unc()

    # Systematic uncertainty (asymmetric)
    syst0_unc = point.get_observables_sys_uncertainties().get_uncertainty(0)
    a = syst0_unc.get_unc_upper()
    b = syst0_unc.get_unc_lower()
    syst0_val = max(a, b)

    # Combining stat + syst uncertainty in quadrature, convert from pb → nb
    delta_f_uu = np.sqrt(stat0_val**2 + syst0_val**2) * 1e-3

    # Computing y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating data row
    row_uu = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),
        "f": obs_dict[obs_names[0]] * 1e-3,  # pb → nb
        "delta_f_uu": delta_f_uu,
        "pol": "UU"
    }

    rows_uu.append(row_uu)

# Saving to Excel
df_uu = pd.DataFrame(rows_uu)
df_uu.to_excel("2_CS_AtY8o7Ej_UU.xlsx", index=False, header=False)

print("Final file written to 2_CS_AtY8o7Ej_UU.xlsx")

#############################################################################################################


"""

  DATASET 3
  uuid: msa6dh9v
  collaboration: HallA
  type: DVCS
  observables: CrossSectionUU 
  
"""
# Constants
E_lepton = 5.55  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('msa6dh9v')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_f = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) * 1e-3

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "f": obs_dict[obs_names[0]] * 1e-3,   # Converting pb to nb],
        "delta_f": delta_f,
        "pol": "UU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_excel("3_CS_msa6dh9v.xlsx", index=False, header=False)

print("Final file written to 3_CS_msa6dh9v.xlsx")


#############################################################################################################

"""

  DATASET 4
  uuid: bmTzHHvg
  collaboration: HallA
  type: DVCS
  observables: CrossSectionUU
  
"""
# Constants
E_lepton = 5.55  # GeV
E_hadron = 0.9395654205  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('bmTzHHvg')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))

    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Statistical uncertainty (symmetric)
    stat_unc = point.get_observables_stat_uncertainties().get_uncertainty(0)
    stat_val = stat_unc.get_unc()

    # Systematic uncertainty (asymmetric)
    syst_unc = point.get_observables_sys_uncertainties().get_uncertainty(0)
    a = syst_unc.get_unc_upper()
    b = syst_unc.get_unc_lower()
    syst_val = max(a,b)

    # Combining stat + syst uncertainty in quadrature, convert from pb → nb
    delta_f = np.sqrt(stat_val**2 + syst_val**2) * 1e-3

    # Compute y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating data row
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),
        "f": obs_dict[obs_names[0]] * 1e-3,  # pb → nb
        "delta_f": delta_f,
        "pol": "UU"
    }

    rows.append(row)

# Saving to Excel
df = pd.DataFrame(rows)
df.to_excel("4_CS_bmTzHHvg.xlsx", index=False, header=False)

print("Final file written to 4_CS_bmTzHHvg.xlsx")

#############################################################################################################


"""

  DATASET 5
  uuid: RQncbKtk
  collaboration: CLAS 
  type: DVCS
  observables: CrossSectionUU
  
"""
# Constants
E_lepton = 5.75  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset from the database
ob = db.get_data_object('RQncbKtk')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Listing to store UU 
rows_uu = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Observable values and uncertainties
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    stat_unc = point.get_observables_stat_uncertainties()
    sys_unc = point.get_observables_sys_uncertainties()

    # Observable 0 is CrossSectionUU and 1 is CrossSectionDifferenceLU
    
    # UU
    stat0 = stat_unc.get_uncertainty(0)
    sys0 = sys_unc.get_uncertainty(0)
    delta_f_uu = np.sqrt(stat0.get_unc()**2 + sys0.get_unc()**2) 
    row_uu = {
            "y": y,
            "xB": xB,
            "t": kin_dict["t"],
            "Q": np.sqrt(Q2),
            "phi": np.radians(kin_dict["phi"]),
            "f": obs_dict[obs_names[0]],
            "delta_f": delta_f_uu,
            "pol": "UU"
        }
    rows_uu.append(row_uu)

    
# Exporting UU data
df_uu = pd.DataFrame(rows_uu)
df_uu.to_excel("5_CS_RQncbKtk_UU.xlsx", index=False, header=False)

print("Final file written to 5_CS_RQncbKtk_UU.xlsx")

#############################################################################################################

"""

  DATASET 6
  uuid: mJXCLi4G
  collaboration: HallA
  type: DVCS
  observables: CrossSectionUU
  
  
"""
# Constants
E_lepton = 4.455  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('mJXCLi4G')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_f = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) * 1e-3

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "f": obs_dict[obs_names[0]] * 1e-3,   # Converting pb to nb],
        "delta_f": delta_f,
        "pol": "UU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_excel("6_CS_mJXCLi4G.xlsx", index=False, header=False)


print("Final file written to 6_CS_mJXCLi4G.xlsx")

#############################################################################################################

"""

  DATASET 7
  uuid: Cb6meE7Q
  collaboration: HallA
  type: DVCS
  observables: CrossSectionUU
  
"""
# Constants
E_lepton = 4.45  # GeV
E_hadron = 0.9395654205  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('Cb6meE7Q')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))

    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Statistical uncertainty (symmetric)
    stat_unc = point.get_observables_stat_uncertainties().get_uncertainty(0)
    stat_val = stat_unc.get_unc()

    # Systematic uncertainty (asymmetric)
    syst_unc = point.get_observables_sys_uncertainties().get_uncertainty(0)
    a = syst_unc.get_unc_upper()
    b = syst_unc.get_unc_lower()
    syst_val = max(a, b)

    # Combining stat + syst uncertainty in quadrature, convert from pb → nb
    delta_f = np.sqrt(stat_val**2 + syst_val**2) * 1e-3

    # Computing y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating data row
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),
        "f": obs_dict[obs_names[0]] * 1e-3,  # pb → nb
        "delta_f": delta_f,
        "pol": "UU"
    }

    rows.append(row)

# Saving to Excel
df = pd.DataFrame(rows)
df.to_excel("7_CS_Cb6meE7Q.xlsx", index=False, header=False)

print("Final file written to 7_CS_Cb6meE7Q.xlsx")

#############################################################################################################


"""

  DATASET 8
  uuid: ob8hLTm2
  collaboration: CLASS
  type: DVCS
  observables: CrossSectionUU
  
"""
# Constants
E_lepton = 5.88  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('ob8hLTm2')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_f = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) 

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Convert degrees to radians
        "f": obs_dict[obs_names[0]],   
        "delta_f": delta_f,
        "pol": "UU"  # Add polarization info
    }

    # Append to the list!
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_excel("8_CS_ob8hLTm2.xlsx", index=False, header=False)

print("Final file written to 8_CS_ob8hLTm2.xlsx")

#############################################################################################################


"""

  DATASET 9
  uuid: 75ueQoQw
  collaboration: COMPASS
  type: DVCS
  observables: CrossSectionUUVirtualPhotoProduction
  
"""

# Constants
E_lepton = 160  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('75ueQoQw')
data = ob.get_data()
data_set = data.get_data_set('t_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))

    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Statistical uncertainty (symmetric)
    stat_unc = point.get_observables_stat_uncertainties().get_uncertainty(0)
    stat_val = stat_unc.get_unc()

    # Systematic uncertainty (asymmetric)
    syst_unc = point.get_observables_sys_uncertainties().get_uncertainty(0)
    a = syst_unc.get_unc_upper()
    b = syst_unc.get_unc_lower()
    syst_val = np.sqrt(a**2 + b**2)

    # Combining stat + syst uncertainty in quadrature
    delta_f = np.sqrt(stat_val**2 + syst_val**2) 

    # Computing y
    Q2 = kin_dict["Q2"]
    nu = kin_dict["nu"]
    xB =Q2 / (2* E_hadron* nu)
    y = Q2 / (2 * xB * E_lepton * E_hadron)
   

    # Creating data row
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "f": obs_dict[obs_names[0]],
        "delta_f": delta_f,
        "pol": "UU"
    }

    rows.append(row)

# Saving to Excel
df = pd.DataFrame(rows)
df.to_excel("9_CS_75ueQoQw.xlsx", index=False, header=False)

print("Final file written to 9_CS_75ueQoQw.xlsx")

#############################################################################################################





# List of all the file names
file_names = [
    "1_CS_TKhscLcB.xlsx",
    "2_CS_AtY8o7Ej_UU.xlsx",
    "3_CS_msa6dh9v.xlsx",
    "4_CS_bmTzHHvg.xlsx",
    "5_CS_RQncbKtk_UU.xlsx",
    "6_CS_mJXCLi4G.xlsx",
    "7_CS_Cb6meE7Q.xlsx",
    "8_CS_ob8hLTm2.xlsx",
    "9_CS_75ueQoQw.xlsx"
]

# Combining all dataframes into one
combined_df = pd.concat([pd.read_excel(file, header=None) for file in file_names], ignore_index=True)

# Saving to a new Excel file
combined_df.to_excel("CS_Combined.xlsx", index=False, header=False)

print("Combined Cross sections file saved as  to CS_Combined.xlsx")














                                                     ###### CROSS SECTION DIFFERENCE ########



"""

  DATASET 1
  uuid: EqbtDRkv
  collaboration: HallA
  type: DVCS
  observables: CrossSectionDifferenceLU
  
"""
# Constants
E_lepton = 3.355  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('EqbtDRkv')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_f = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) * 1e-3

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "f": obs_dict[obs_names[0]] * 1e-3,   # Converting pb to nb],
        "delta_f": delta_f,
        "pol": "LU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_excel("1_CSD_EqbtDRkv.xlsx", index=False, header=False)


print("Final file written to 1_CSD_EqbtDRkv.xlsx")

#############################################################################################################

"""

  DATASET 2
  uuid: nfPvTM2c
  collaboration: HallA
  type: DVCS
  observables: CrossSectionDifferenceLU
  
"""
# Constants
E_lepton = 4.455  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('nfPvTM2c')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_f = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) * 1e-3

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "f": obs_dict[obs_names[0]] * 1e-3,   # Converting pb to nb],
        "delta_f": delta_f,
        "pol": "LU"  # Add polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_excel("2_CSD_nfPvTM2c.xlsx", index=False, header=False)


print("Final file written to 2_CSD_nfPvTM2c.xlsx")


#############################################################################################################



"""

  DATASET 3
  uuid: AtY8o7Ej
  collaboration: HallA
  type: DVCS
  observables: CrossSectionDifferenceLU
  
"""

# Constants
E_lepton = 5.7572  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('AtY8o7Ej')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')



# Preparing data
rows_lu = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))

    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Statistical uncertainty (symmetric)
    stat1_unc = point.get_observables_stat_uncertainties().get_uncertainty(1)
    stat1_val = stat1_unc.get_unc()

    # Systematic uncertainty (asymmetric)
    syst1_unc = point.get_observables_sys_uncertainties().get_uncertainty(1)
    c = syst1_unc.get_unc_upper()
    d = syst1_unc.get_unc_lower()
    syst1_val = max(c, d)

    # Combining stat + syst uncertainty in quadrature, convert from pb → nb
    delta_f_lu = np.sqrt(stat1_val**2 + syst1_val**2) * 1e-3

    # Computing y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating data row
    row_lu = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),
        "f": obs_dict[obs_names[1]] * 1e-3,  # pb → nb
        "delta_f_lu": delta_f_lu,
        "pol": "LU"
    }

    rows_lu.append(row_lu)

# Saving to Excel
df_lu = pd.DataFrame(rows_lu)
df_lu.to_excel("3_CSD_AtY8o7Ej_LU.xlsx", index=False, header=False)


print("Final file written to 3_CSD_AtY8o7Ej_LU.xlsx")






"""

  DATASET 4
  uuid: RQncbKtk
  collaboration: CLAS 
  type: DVCS
  observables: CrossSectionDifferenceLU
  
"""
# Constants
E_lepton = 5.75  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset from the database
ob = db.get_data_object('RQncbKtk')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Listing to store UU and LU data separately

rows_lu = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Observable values and uncertainties
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    stat_unc = point.get_observables_stat_uncertainties()
    sys_unc = point.get_observables_sys_uncertainties()

    # Observable 0 is CrossSectionUU and 1 is CrossSectionDifferenceLU
   
    # LU
    stat1 = stat_unc.get_uncertainty(1)
    sys1 = sys_unc.get_uncertainty(1)
    delta_f_lu = np.sqrt(stat1.get_unc()**2 + sys1.get_unc()**2)
    row_lu = {
            "y": y,
            "xB": xB,
            "t": kin_dict["t"],
            "Q": np.sqrt(Q2),
            "phi": np.radians(kin_dict["phi"]),
            "f": obs_dict[obs_names[1]],
            "delta_f": delta_f_lu,
            "pol": "LU"
        }
    rows_lu.append(row_lu)



# Exporting LU data
df_lu = pd.DataFrame(rows_lu)
df_lu.to_excel("4_CSD_RQncbKtk_LU.xlsx", index=False, header=False)

print("Final file written to 4_CSD_RQncbKtk_LU.xlsx")


#############################################################################################################




"""

  DATASET 5
  uuid: BJ84iv8s
  collaboration: HallA
  type: DVCS
  observables: CrossSectionDifferenceLU
  
"""

# Constants
E_lepton = 5.7572  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('BJ84iv8s')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))

    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Statistical uncertainty (symmetric)
    stat_unc = point.get_observables_stat_uncertainties().get_uncertainty(0)
    stat_val = stat_unc.get_unc()

    # Systematic uncertainty (asymmetric)
    syst_unc = point.get_observables_sys_uncertainties().get_uncertainty(0)
    a = syst_unc.get_unc_upper()
    b = syst_unc.get_unc_lower()
    syst_val = max(a,b)

    # Combining stat + syst uncertainty in quadrature, convert from pb → nb
    delta_f = np.sqrt(stat_val**2 + syst_val**2) * 1e-3

    # Computing y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating data row
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),
        "f": obs_dict[obs_names[0]] * 1e-3,  # pb → nb
        "delta_f": delta_f,
        "pol": "LU"
    }

    rows.append(row)

# Saving to Excel
df = pd.DataFrame(rows)
df.to_excel("5_CSD_BJ84iv8s.xlsx", index=False, header=False)

print("Final file written to 5_CSD_BJ84iv8s.xlsx")




#############################################################################################################


"""

  DATASET 6
  uuid: EhPp8CP4
  collaboration: HALLA
  type: DVCS
  observables: CrossSectionDifferenceLU
  
"""


# Constants
E_lepton =  5.55  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('EhPp8CP4')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_f = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) * 1e-3 

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "f": obs_dict[obs_names[0]]* 1e-3,   # Converting pb to nb],   
        "delta_f": delta_f,
        "pol": "LU"  # Add polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_excel("6_CSD_EhPp8CP4.xlsx", index=False, header=False)


print("Final file written to 6_CSD_EhPp8CP4.xlsx")














# List of all the Cross Section Difference LU Files
file_list = [
    "1_CSD_EqbtDRkv.xlsx",
    "2_CSD_nfPvTM2c.xlsx",
    "3_CSD_AtY8o7Ej_LU.xlsx",
    "4_CSD_RQncbKtk_LU.xlsx",
    "5_CSD_BJ84iv8s.xlsx",
    "6_CSD_EhPp8CP4.xlsx"
]

# Combining all the files
combined_df = pd.concat([pd.read_excel(file, header=None) for file in file_list], ignore_index=True)

# Saving to a new Excel file
combined_df.to_excel("CSD_Combined.xlsx", index=False, header=False)

print("Combined Cross section Difference file saved as CSD_Combined.xlsx")






                                                                    ###### ASYMMETRIES ########

i


"""

  DATASET 1
  uuid: A6Pgo5TE
  collaboration: CLAS
  type: DVCS
  observables: ALU Q2 Dependent
  
"""

# Constants
E_lepton = 4.8  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('A6Pgo5TE')
data = ob.get_data()
data_set = data.get_data_set('Q2_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    #syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_A = stat.get_unc() 

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "A": obs_dict[obs_names[0]],
        "delta_A": delta_A,
        "pol": "LU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_excel("1_ALU_A6Pgo5TE_Q2.xlsx", index=False, header=False)


print("Final file written to 1_ALU_A6Pgo5TE_Q2.xlsx")

###########################################################################################################

"""

  DATASET 2
  uuid: A6Pgo5TE
  collaboration: CLAS
  type: DVCS
  observables: ALU t Dependent
  
"""
# Constants
E_lepton = 4.8  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('A6Pgo5TE')
data = ob.get_data()
data_set = data.get_data_set('t_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    #syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_A = stat.get_unc() 

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "A": obs_dict[obs_names[0]],
        "delta_A": delta_A,
        "pol": "LU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_excel("2_ALU_A6Pgo5TE_t.xlsx", index=False, header=False)


print("Final file written to 2_ALU_A6Pgo5TE_t.xlsx")


###########################################################################################################

"""

  DATASET 3
  uuid: NvMm42PD
  collaboration: HERMES
  type: DVCS
  observables: ALU
  
"""
# Constants
E_lepton = 27.6  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('NvMm42PD')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_A = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) 

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "A": obs_dict[obs_names[0]],
        "delta_A": delta_A,
        "pol": "LU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_excel("3_ALU_NvMm42PD.xlsx", index=False, header=False)


print("Final file written to 3_ALU_NvMm42PD.xlsx")

###########################################################################################################


"""

  DATASET 4
  uuid: QfefWWW2
  collaboration: CLAS
  type: DVCS
  observables: ALL
  
"""
# Constants
E_lepton = 10.6  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('QfefWWW2')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    #syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_A = stat.get_unc()

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "A": obs_dict[obs_names[0]],
        "delta_A": delta_A,
        "pol": "LL"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_excel("4_ALL_QfefWWW2.xlsx", index=False, header=False)


print("Final file written to 4_ALL_QfefWWW2.xlsx")


###########################################################################################################


"""

  DATASET 5
  uuid: PusMstKs
  collaboration: CLAS
  type: DVCS
  observables: ALL
  
"""
# Constants
E_lepton = 10.2  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('PusMstKs')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    #syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_A = stat.get_unc()

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "A": obs_dict[obs_names[0]],
        "delta_A": delta_A,
        "pol": "LL"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and exporting
df = pd.DataFrame(rows)
df.to_excel("5_ALL_PusMstKs.xlsx", index=False, header=False)


print("Final file written to 5_ALL_PusMstKs.xlsx")



###########################################################################################################


"""

  DATASET 6
  uuid: AvF5daeP
  collaboration: HERMES
  type: DVCS
  observables: ALU
  
"""
# Constants
E_lepton = 5.77  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('AvF5daeP')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_A = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) 

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "A": obs_dict[obs_names[0]],
        "delta_A": delta_A,
        "pol": "LU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_excel("6_ALU_AvF5daeP.xlsx", index=False, header=False)


print("Final file written to 6_ALU_AvF5daeP.xlsx")


###########################################################################################################


"""

  DATASET 7
  uuid: vGAKAf7P
  collaboration: CLAS 
  type: DVCS
  observables: ALU, AUL, ALL
  
"""
# Constants
E_lepton = 5.932  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset from the database
ob = db.get_data_object('vGAKAf7P')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Listing to store UU and LU data separately
rows_lu = []
rows_ul = []
rows_ll = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Observable values and uncertainties
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    stat_unc = point.get_observables_stat_uncertainties()
    sys_unc = point.get_observables_sys_uncertainties()

    # Observable 0 is CrossSectionUU and 1 is CrossSectionDifferenceLU
    
    # LU
    stat0 = stat_unc.get_uncertainty(0)
    sys0 = sys_unc.get_uncertainty(0)
    delta_ALU = np.sqrt(stat0.get_unc()**2 + sys0.get_unc()**2) 
    row_lu = {
            "y": y,
            "xB": xB,
            "t": kin_dict["t"],
            "Q": np.sqrt(Q2),
            "phi": np.radians(kin_dict["phi"]),
            "ALU": obs_dict[obs_names[0]],
            "delta_ALU": delta_ALU,
            "pol": "LU"
        }
    rows_lu.append(row_lu)

    # UL
    stat1 = stat_unc.get_uncertainty(1)
    sys1 = sys_unc.get_uncertainty(1)
    delta_AUL = np.sqrt(stat1.get_unc()**2 + sys1.get_unc()**2)
    row_ul = {
            "y": y,
            "xB": xB,
            "t": kin_dict["t"],
            "Q": np.sqrt(Q2),
            "phi": np.radians(kin_dict["phi"]),
            "AUL": obs_dict[obs_names[1]],
            "delta_AUL": delta_AUL,
            "pol": "UL"
        }
    rows_ul.append(row_ul)
    
    # LL
    stat2 = stat_unc.get_uncertainty(2)
    sys2 = sys_unc.get_uncertainty(2)
    delta_ALL = np.sqrt(stat2.get_unc()**2 + sys2.get_unc()**2)
    row_ll = {
            "y": y,
            "xB": xB,
            "t": kin_dict["t"],
            "Q": np.sqrt(Q2),
            "phi": np.radians(kin_dict["phi"]),
            "ALL": obs_dict[obs_names[2]],
            "delta_ALL": delta_ALL,
            "pol": "LL"
        }
    rows_ll.append(row_ll)


# Exporting LU data
df_lu = pd.DataFrame(rows_lu)
df_lu.to_excel("7.1_ALU_vGAKAf7P.xlsx", index=False, header=False)

# Exporting UL data
df_ul = pd.DataFrame(rows_ul)
df_ul.to_excel("7.2_AUL_vGAKAf7P.xlsx", index=False, header=False)

# Exporting LL data
df_ll = pd.DataFrame(rows_ll)
df_ll.to_excel("7.3_ALL_vGAKAf7P.xlsx", index=False, header=False)

print("ALU, AUL, ALL files written to 7.1_ALU_vGAKAf7P.xlsx and 7.2_AUL_vGAKAf7P.xlsx and 7.3_ALL_vGAKAf7P.xlsx")







# List of all asymmetry files 
file_list = [
    "1_ALU_A6Pgo5TE_Q2.xlsx",
    "2_ALU_A6Pgo5TE_t.xlsx",
    "3_ALU_NvMm42PD.xlsx",
    "4_ALL_QfefWWW2.xlsx",
    "5_ALL_PusMstKs.xlsx",
    "6_ALU_AvF5daeP.xlsx",
    "7.1_ALU_vGAKAf7P.xlsx",
    "7.2_AUL_vGAKAf7P.xlsx",
    "7.3_ALL_vGAKAf7P.xlsx"
]

# Combining all files
combined_df = pd.concat([pd.read_excel(file, header=None) for file in file_list], ignore_index=True)

# Saving to a new Excel file
combined_df.to_excel("ASYMMETRIES_combined.xlsx", index=False, header=False)

print("Combined asymmetry file saved as ASYMMETRIES_combined.xlsx")





