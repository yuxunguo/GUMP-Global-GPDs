#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 13:20:00 2025

@author: fpaslan
"""

import gpddatabase
import pandas as pd
import numpy as np

# make a reference to the database
db = gpddatabase.ExclusiveDatabase()

# print availible uuids
print(db.get_uuids())



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
df.to_excel("1_DVCS_EqbtDRkv.xlsx", index=False, header=False)


print("Final file written to 1_DVCS_EqbtDRkv.xlsx")




########################################################################################################################################################


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
df.to_excel("2_DVCS_nfPvTM2c.xlsx", index=False, header=False)


print("Final file written to 2_DVCS_nfPvTM2c.xlsx")






########################################################################################################################################################



"""

  DATASET 3
  uuid: TKhscLcB
  collaboration: HallA
  type: DVCS
  observables: CrossSectionDifferenceUU
  
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
df.to_excel("3_DVCS_TKhscLcB.xlsx", index=False, header=False)


print("Final file written to 3_TKhscLcB.xlsx")



########################################################################################################################################################




"""

  DATASET 4
  uuid: AtY8o7Ej
  collaboration: HallA
  type: DVCS
  observables: CrossSectionDifferenceUU, CrossSectionDifferenceLU
              
  
"""

# Constants
E_lepton = 5.7572  # GeV
E_hadron = 0.93827208816  # GeV

# Create database object
db = gpddatabase.ExclusiveDatabase()

# Load the dataset
ob = db.get_data_object('AtY8o7Ej')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Prepare data
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
    syst0_val = np.sqrt(a**2 + b**2)

    # Combine stat + syst uncertainty in quadrature, convert from pb → nb
    delta_f_uu = np.sqrt(stat0_val**2 + syst0_val**2) * 1e-3

    # Compute y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Create data row
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

# Save to Excel
df_uu = pd.DataFrame(rows_uu)
df_uu.to_excel("4_DVCS_AtY8o7Ej_UU.xlsx", index=False, header=False)












# Prepare data
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
    syst1_val = np.sqrt(c**2 + d**2)

    # Combine stat + syst uncertainty in quadrature, convert from pb → nb
    delta_f_lu = np.sqrt(stat1_val**2 + syst1_val**2) * 1e-3

    # Compute y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Create data row
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

# Save to Excel
df_lu = pd.DataFrame(rows_lu)
df_lu.to_excel("4_DVCS_AtY8o7Ej_LU.xlsx", index=False, header=False)


print("Final file written to 4_DVCS_AtY8o7Ej_LU.xlsx")
print("Final file written to 4_DVCS_AtY8o7Ej_UU.xlsx")












########################################################################################################################################################


"""

  DATASET 5
  uuid: msa6dh9v
  collaboration: HallA
  type: DVCS
  observables: CrossSectionDifferenceUU
  
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
df.to_excel("5_DVCS_msa6dh9v.xlsx", index=False, header=False)


print("Final file written to 5_DVCS_msa6dh9v.xlsx")




########################################################################################################################################################



"""
  DATASET 6
  uuid: bmTzHHvg
  collaboration: HallA
  type: DVCS
  observables: CrossSectionDifferenceUU
"""

# Constants
E_lepton = 5.55  # GeV
E_hadron = 0.9395654205  # GeV

# Create database object
db = gpddatabase.ExclusiveDatabase()

# Load the dataset
ob = db.get_data_object('bmTzHHvg')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Prepare data
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

    # Combine stat + syst uncertainty in quadrature, convert from pb → nb
    delta_f = np.sqrt(stat_val**2 + syst_val**2) * 1e-3

    # Compute y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Create data row
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

# Save to Excel
df = pd.DataFrame(rows)
df.to_excel("6_DVCS_bmTzHHvg.xlsx", index=False, header=False)

print("Final file written to 6_DVCS_bmTzHHvg.xlsx")





########################################################################################################################################################
"""

  DATASET 7
  uuid: RQncbKtk
  collaboration: CLAS 
  type: DVCS
  observables: CrossSectionDifferenceUU,CrossSectionDifferenceLU
  
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
rows_uu = []
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

# Exporting UU data
df_uu = pd.DataFrame(rows_uu)
df_uu.to_excel("7_DVCS_RQncbKtk_UU.xlsx", index=False, header=False)

# Exporting LU data
df_lu = pd.DataFrame(rows_lu)
df_lu.to_excel("7_DVCS_RQncbKtk_LU.xlsx", index=False, header=False)

print("UU and LU files written to 7_DVCS_RQncbKtk_UU.xlsx and 7_DVCS_RQncbKtk_LU.xlsx")






########################################################################################################################################################





"""

  DATASET 8
  uuid: mJXCLi4G
  collaboration: HallA
  type: DVCS
  observables: CrossSectionDifferenceUU
  
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
df.to_excel("8_DVCS_mJXCLi4G.xlsx", index=False, header=False)


print("Final file written to 8_DVCS_mJXCLi4G.xlsx")


########################################################################################################################################################




"""

  DATASET 9
  uuid: Cb6meE7Q
  collaboration: HallA
  type: DVCS
  observables: CrossSectionDifferenceUU
  
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
    syst_val = np.sqrt(a**2 + b**2)

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
df.to_excel("9_DVCS_Cb6meE7Q.xlsx", index=False, header=False)

print("Final file written to 9_DVCS_Cb6meE7Q.xlsx")



########################################################################################################################################################


"""

  DATASET 10
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
    syst_val = np.sqrt(a**2 + b**2)

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
df.to_excel("10_DVCS_BJ84iv8s.xlsx", index=False, header=False)

print("Final file written to 10_DVCS_BJ84iv8s.xlsx")

########################################################################################################################################################



"""

  DATASET 11
  uuid: ob8hLTm2
  collaboration: CLASS
  type: DVCS
  observables: CrossSectionDifferenceUU
  
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
df.to_excel("11_DVCS_ob8hLTm2.xlsx", index=False, header=False)


print("Final file written to 11_DVCS_ob8hLTm2.xlsx")


########################################################################################################################################################

"""

  DATASET 12
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
df.to_excel("12_DVCS_75ueQoQw.xlsx", index=False, header=False)

print("Final file written to 12_DVCS_75ueQoQw.xlsx")

########################################################################################################################################################

"""

  DATASET 13
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
        "pol": "LU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_excel("13_DVCS_EhPp8CP4.xlsx", index=False, header=False)


print("Final file written to 13_DVCS_EhPp8CP4.xlsx")
