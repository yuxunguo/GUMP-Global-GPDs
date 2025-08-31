import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MaxAbsScaler

def cluster_DVCSAsym(DVCSAsym_data, t_weight=3, eps=0.11, min_samples=5, verbose=False):
    """
    Cluster DVCSAsym data based on xB, Q2, and t using DBSCAN, calculate cluster statistics
    and deviations, and return the processed DataFrame.

    Parameters:
    -----------
    DVCSAsym_data : pd.DataFrame
        Input DVCS asymmetry data.
    t_weight : float
        Weight applied to 't' for clustering (default=3).
    eps : float
        DBSCAN epsilon parameter (default=0.11).
    min_samples : int
        DBSCAN min_samples parameter (default=5).
    verbose : bool
        If True, prints deviations and cluster info (default=False).

    Returns:
    --------
    DVCSAsym : pd.DataFrame
        The clustered dataframe with cluster means, deviations, and standard deviations.
    """

    DVCSAsym = DVCSAsym_data.copy()
    DVCSAsym['Q2'] = DVCSAsym['Q']**2

    # Scale columns and emphasize t
    X_scaled = MaxAbsScaler().fit_transform(DVCSAsym[['xB','Q2','t']])
    X_scaled[:, 2] *= t_weight

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    DVCSAsym['cluster'] = db.labels_

    # Filter out noise
    DVCSAsym = DVCSAsym[DVCSAsym['cluster'] != -1]

    # Calculate cluster means
    cluster_means = DVCSAsym.groupby('cluster')[['xB','Q2','t']].transform('mean')
    DVCSAsym[['xB_mean','Q2_mean','t_mean']] = cluster_means

    # Calculate deviations
    DVCSAsym['xB_dev'] = DVCSAsym['xB'] - DVCSAsym['xB_mean']
    DVCSAsym['Q2_dev'] = DVCSAsym['Q2'] - DVCSAsym['Q2_mean']
    DVCSAsym['t_dev'] = DVCSAsym['t'] - DVCSAsym['t_mean']

    # Calculate cluster standard deviations
    cluster_std = DVCSAsym.groupby('cluster')[['xB','Q2','t']].transform('std')
    DVCSAsym[['xB_std','Q2_std','t_std']] = cluster_std

    # Optionally print cluster info
    if verbose:
        grouped = DVCSAsym.groupby(['xB_mean', 'Q2_mean','t_mean'])
        for name, group in grouped:
            print(f"\nGroup: {name}")
            print(group[['xB_dev','Q2_dev','t_dev','xB_std','Q2_std','t_std','cluster']])

    # Select only required columns and rename
    DVCSAsym_clean = DVCSAsym[['y','xB_mean','t_mean','Q2_mean','phi','f','delta f','pol','comment']].copy()
    DVCSAsym_clean.rename(columns={'xB_mean':'xB', 't_mean':'t', 'Q2_mean':'Q'}, inplace=True)

    # Convert Q2_mean to sqrt(Q2_mean) as 'Q'
    DVCSAsym_clean['Q'] = np.sqrt(DVCSAsym_clean['Q'])

    return DVCSAsym_clean