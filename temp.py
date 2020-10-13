# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 12:21:07 2020

@author: hites
"""
import operator
import pandas as pd
import numpy as np
import math
from tqdm import tqdm


def femi_numerical(dataset, degrees, max_iter):
    """FEMI algorithm for numeric attributes.

    Note that input dataset must only have numerical attributes.
    
    The algorithm works as below:
    For each cluster k
        For each record i
            let Ri = DI[i]
            let m = missing attributes Ri
            let av = available attributes of Ri
            let u^k_av = \sum_{i = 1 .. |DC|} {Uik * Ri{av}} / \sum_{i = 1 .. |DC|} {Uik}
            let u^k_ms = \sum_{i = 1 .. |DC|} {Uik * Ri{m}} / \sum_{i = 1 .. |DC|} {Uik}

    Args:
        dataset: pd.Dataframe. Dataframe consisting of data points which have
            missing as well as complete information. All attributes are expected
            to be numerical.
        degrees: pd.Dataframe. Membership degree of data points in dataset,
            stored in same order as the order of records in dataset.
            That is dataset.loc[i] has membership degree degrees.loc[i].
        max_iter: int. Maximum iterations for EM algorithm to execute.
    
    Returns:
        dataset: pd.Dataframe. Dataframe consisting of data points with imputed
            values. 
    """
    # Make copies of original dataframe because these are needed whenever
    # DI is appended to DC for usage in next iteration.
    attribute_names = dataset.columns.tolist()
    missing_list = {}
    for Ridx, Ri in dataset[dataset.isnull().any(axis=1)].iterrows():
        # populate with names of attributes with missing values.
        missing_list[Ridx] = [
            name for name in attribute_names if pd.isna(Ri[name])]
    
    for iter in tqdm(range(max_iter), desc='Iteration Loop'):
        for Ridx in tqdm(
                missing_list.keys(), desc='Datapoint Loop', leave=False):
            available_list = [
                attr for attr in attribute_names
                if attr not in missing_list[Ridx]]
            DC = dataset[~dataset.isnull().any(axis=1)]
            UC = degrees[~dataset.isnull().any(axis=1)].values
            # print(f"UC from him {UC.shape}")
            # print(UC)
            # print(f'UC from func {UC.shape}')
            # print(f'DC from func {DC.shape}')
            Dmissing = DC[missing_list[Ridx]].values
            # print(missing_list) 
            # print(df_incomplete)
            Davailable = DC[available_list].values
            
            Ri = dataset.loc[Ridx]
            u_m = np.dot(UC.T, Dmissing) / np.sum(UC, axis=0).reshape(-1, 1)
            u_av = np.dot(UC.T, Davailable) / np.sum(UC, axis=0).reshape(-1, 1)
            Rm = np.zeros((UC.shape[1], len(missing_list[Ridx])))
            for k in range(UC.shape[1]):
                theta_mm = np.dot(
                    (UC[:, k].reshape(-1, 1) * (Dmissing - u_m[k])).T,
                    Dmissing - u_m[k])
                theta_am = np.dot(
                    (UC[:, k].reshape(-1, 1) * (Davailable - u_av[k])).T,
                    Dmissing - u_m[k])
                theta_ma = np.dot(
                    (UC[:, k].reshape(-1, 1) * (Dmissing - u_m[k])).T,
                    Davailable - u_av[k])
                theta_aa = np.dot(
                    (UC[:, k].reshape(-1, 1) * (Davailable - u_av[k])).T,
                    Davailable - u_av[k])
                eps = 0 # This should be calculated properly for first iteration.
                if iter == 1:
                    Q = theta_mm - np.dot(
                        theta_ma, np.dot(np.linalg.pinv(theta_aa), theta_am))
                    H = np.linalg.cholesky(Q)
                    Z = np.random.normal(size=u_m.shape[1])
                    eps = np.dot(H, Z.T).T
                Rm[k] = u_m[k] + np.dot(
                    Ri[available_list].values - u_av[k],
                    np.dot(np.linalg.pinv(theta_aa), theta_am)) + eps
            
            Rmav = np.dot(degrees.loc[Ridx].values, Rm).squeeze()
            # If only one attribute is missing then squeeze() directly returns
            # float value, instead of a 1-D array of length 1.
            if u_m.shape[1] == 1:
                Rmav = [Rmav]

            # Impute values back in dataset.
            for val, attr in zip(Rmav, missing_list[Ridx]):
                dataset.at[Ridx, attr] = val
    
    return dataset

def femi_categorical(dataset, degrees):
    """FEMI algorithm for categorical attributes.

    Note that input dataset must only have categorical attributes.

    Args:
        dataset: pd.Dataframe. Dataframe consisting of data points which have
            missing as well as complete information. All attributes are expected
            to be numerical.
        degrees: pd.Dataframe. Membership degree of data points in dataset,
            stored in same order as the order of records in dataset.
            That is dataset.loc[i] has membership degree degrees.loc[i].
    
    Returns:
        dataset: pd.Dataframe. Dataframe consisting of data points with imputed
            values. 
    """
    for col in dataset.columns.tolist():
        domain_values = [
            val for val in dataset[col].unique().tolist() if not pd.isna(val)]
        confidence_matrix = np.zeros((len(domain_values), len(degrees.columns)))
        for idx, val in enumerate(domain_values):
            confidence_matrix[idx] = degrees[dataset[col] == val].sum(axis=0).values
        UI = degrees[dataset[col].isnull()]
        Ridcs = UI.index.tolist()
        votes = np.dot(UI, confidence_matrix.T)
        categories = np.argmax(votes, axis=1).tolist()
        for Ridx, category in zip(Ridcs, categories):
            dataset.at[Ridx, col] = domain_values[category]
    return dataset



def initializeMembership(n,k):
    U = np.random.rand(n,k)
    U = U/U.sum(axis=1)[:,None]
    return U

def calculateClusterCenter(membership_mat,df):
    cluster_mem_val = list(zip(*membership_mat))
    # print(cluster_mem_val)
    cluster_centers = list()
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [e ** M for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(len(df)):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = list(map(sum, zip(*temp_num)))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
        # print(cluster_centers)
    return cluster_centers

def updateMembershipValue(membership_mat, cluster_centers,df):
    p = float(2/(M-1))
    # print(len(df))
    for i in range(len(df)):
        x = list(df.iloc[i])
        # dis = norm(x, cluster_centers[j])
        distances = [np.linalg.norm(list(map(operator.sub, x , cluster_centers[j]))) for j in range(k) ]
        # print(pd.DataFrame(distances))
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1/den)       
    return membership_mat

def fuzzyCmeans(df,n,k):
    U = initializeMembership(n,k)
    itr =0
    while(itr< 10):
        v = calculateClusterCenter(U,df)
        U = updateMembershipValue(U, v,df)
        itr= itr +1
    return v,U

def calculateNRMS(x_original, x_imputed):
    up = np.linalg.norm( x_original - x_imputed)
    down = np.linalg.norm(x_original)
    nrms = up/down
    return nrms
    

                    
    
if __name__ == '__main__':
    X_ori = pd.read_excel(r'C:\Users\hites\Desktop\New folder\Glass.xlsx', header=None)
    data = pd.read_excel(r'C:\Users\hites\Desktop\New folder\Glass\Glass_C_20.xlsx', header=None)
    normalized_df=(data-data.min())/(data.max()-data.min())
    df_c = data.dropna()
    # print(f'complete dataset from main {df_c.shape}' )
    df_incomplete = data[data.isnull().any(axis=1)]
    df_i = normalized_df[normalized_df.isnull().any(axis=1)]
    df_i = pd.DataFrame(df_i).fillna(0)
    
    n = len(data)           # number of rows
    m = len(data.columns)   # number of columns
    k = 10                 # number of clusters
    M = 2 
     
    [v,u] = fuzzyCmeans(df_c, len(df_c),k)
    v = pd.DataFrame(v)
    u = pd.DataFrame(u)
    
    [vi,ui] = fuzzyCmeans(df_i, len(df_i), k)
   
    ui = pd.DataFrame(ui)
    md = pd.concat([u,ui],ignore_index=True)
    dataset = femi_numerical(data, md, 10)
    print(dataset)
    # dataset.to_excel(r'C:\Users\hites\Desktop\imputed.xlsx', header=False, index = False)   
    
    NRMS = calculateNRMS(X_ori, dataset)
    print(f'NRMS value is {NRMS}')
            



