import pandas as pd
import numpy as np
import os
import pickle


def get_perm(s,n,m,X_in,Y_in,Z):
    if not os.path.exists(f'datasets/lalonde_{m}'):
        os.makedirs(f'datasets/lalonde_{m}')
    np.random.seed(s)
    perm_vec = np.random.permutation(n)[:m]
    T= X_in[perm_vec]
    Y=Y_in[perm_vec]
    X=Z[perm_vec,:]
    with open(f'datasets/lalonde_{m}/job_{s}.pickle', 'wb') as handle:
        pickle.dump({'seed': s, 'T': T, 'Y': Y, 'X': X, 'W': T}, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    df = pd.read_csv("lalonde.csv")
    X = df['treat'].values
    Y = df['re78'].values
    Z_df = df[['age','education','black','hispanic','married','nodegree','re74','re75']]
    Z = Z_df.values
    cat_cols = ['education','black','hispanic','married','nodegree']
    col_stats_list=[]
    col_counts=[]
    col_index_list = [False] * Z.shape[1]

    for cat_col in cat_cols:
        col_index_list[Z_df.columns.get_loc(cat_col)] = True
    for cat_col in cat_cols:
        col_stats = Z_df[cat_col].unique().tolist()
        col_stats_list.append(col_stats)
        col_counts.append(len(col_stats))
    cat_data = {'indicator':col_index_list,'index_lists':col_stats_list}
    n = X.shape[0]
    print(X)
    # X=X.values
    # Y=Y.values
    # Z=Z.values
    for s in range(100):
        get_perm(s=s, n=X.shape[0], m=100, X_in=X, Y_in=Y, Z=Z)

    with open(f'datasets/lalonde_cat_col_data.pickle', 'wb') as handle:
        pickle.dump(cat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

