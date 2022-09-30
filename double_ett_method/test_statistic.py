import numpy as np
import torch
from doubly_robust_method.kernels import *
from doubly_robust_method.kme import *
from sklearn.preprocessing import KBinsDiscretizer
from torch.distributions import Bernoulli
general_ker_obj =Kernel()
import random
import copy
def swap(xs, a, b):
    xs[a], xs[b] = xs[b], xs[a]

def derange(xs):
    for a in range(1, len(xs)):
        b = random.choice(range(0, a))
        swap(xs, a, b)
    return xs

class counterfactual_me_test():
    def __init__(self,X,Y,e,perm_e,T,kme_1,kme_1_indep,permute_e=False,permutations=250,device='cuda:0',debug_mode=False):
        self.permutations = permutations
        self.X=torch.from_numpy(X).float().to(device)
        self.n= Y.shape[0]
        self.Y = torch.from_numpy(Y).float().to(device)
        self.e =e.to(device)
        self.T_1 = torch.from_numpy(T).float().to(device)
        self.T_0 =1.-self.T_1
        mask_1 = (self.T_1==1).squeeze()
        self.Y_1 = self.Y[mask_1,:]
        self.Y_0 = self.Y[~mask_1,:]
        self.debug_mode = debug_mode
        self.X_1 = self.X[mask_1,:]
        self.X_0 = self.X[~mask_1,:]
        self.idx_array = [i for i in range(self.n)]
        self.kme_1=kme_1
        self.kme_1_indep=kme_1_indep
        self.permute_e = permute_e
        self.ls =general_ker_obj.get_median_ls(self.Y,self.Y)

        self.setup_Y_kernel(self.Y,'L',device)
        self.setup_Y_kernel(self.Y_0,'L_0',device)
        self.setup_Y_kernel(self.Y_1,'L_1',device)

        # self.kernel = RBFKernel(self.Y).to(device)
        # self.ls =general_ker_obj.get_median_ls(self.Y, self.Y)
        # self.kernel._set_lengthscale(self.ls)
        # self.L = self.kernel.evaluate()

        self.create_all_weights(self.e)
        self.ref_stat = self.calculate_test_statistic(self.L)
        print('ref errors')

    def setup_Y_kernel(self,Y,name,device):
        kernel = RBFKernel(Y).to(device)
        kernel._set_lengthscale(self.ls)
        L = kernel.evaluate()
        setattr(self,name,L)
        setattr(self,'kernel_'+name,kernel)
    #permutation should occur twice

    def calc_psi(self,X,kme_1):
        self.psi_1 = kme_1.get_psi_part(X, self.psi_1_weight).t()

    def sanity_check_estimates(self,X_0,X_1,kme_1,Y_0,Y_1,L_0,L_1):
        error_1 = kme_1.calculate_error_external(X_1,Y_1,L_1)
        return error_0.item(),error_1.item()

    def calculate_test_statistic(self,L):
        T_1_L_test = L@self.T_1_weight
        T_0_L_test = L@self.T_0_weight
        term_1 = (self.psi_1_weight.t @ L) @ self.psi_1_weight
        term_2=self.kme_1.get_psi_square_term(self,psi_1)
        term_3= -2*(self.get_psi_cross_term(self, self.psi_1,self.Y,self.psi_1_weight))
        return 1/self.n**2 * tot
        # return self.omega.t()@(L@self.omega).item()

    def create_all_weights(self,e):
        self.psi_1_weight=(self.T_1-e)/(e)
        self.T_1_weight=self.T_1/e
        self.T_0_weight = self.T_0/(1.-e)

    def get_permuted2d(self,ker):
        idx = torch.randperm(self.n)
        # idx = np.array(derange(self.idx_array))
        kernel_X = ker[:,idx]
        kernel_X = kernel_X[idx,:]
        return kernel_X,idx

    def permutation_test(self):
        perm_stats=[]
        running_err_0 = 0.
        running_err_1 = 0.

        for i in range(self.permutations):
            perm_L,idx = self.get_permuted2d(self.L)
            if self.permute_e:
                X = self.X[idx]
                self.calc_psi(X,self.kme_0_indep,self.kme_1_indep)
                if self.debug_mode:
                    perm_Y  = self.Y[idx]
                    perm_Y_0 = perm_Y[:self.Y_0.shape[0]]
                    perm_Y_1 = perm_Y[self.Y_0.shape[0]:]
                    perm_X_0 =X[:self.Y_0.shape[0]]
                    perm_X_1 =X[self.Y_0.shape[0]:]
                    perm_L_0 = self.kernel_L_0(perm_Y_0,perm_Y_0)
                    perm_L_1 = self.kernel_L_1(perm_Y_1,perm_Y_1)
                    a,b = self.sanity_check_estimates(perm_X_0,perm_X_1, self.kme_1_indep, perm_Y_0, perm_Y_1, perm_L_0,
                                                      perm_L_1)
                    running_err_0+=a
                    running_err_1+=b
            else:
                e=self.e
            tst = self.calculate_test_statistic(perm_L)
            perm_stats.append(tst.item())
        if self.debug_mode:
            print('average perm errors')
            print(running_err_0/self.permutations,running_err_1/self.permutations)
        return perm_stats,self.ref_stat.item()

def permute(n_bins,og_indices,clusters):
    permutation = copy.deepcopy(og_indices)
    for i in range(n_bins):
        mask = i==clusters
        group = og_indices[mask]
        permuted_group=np.random.permutation(group)
        permutation[mask]=permuted_group
    return permutation

class counterfactual_me_test_correct(counterfactual_me_test):
    def __init__(self,
                 X,Y,e,perm_e,T,kme_1,kme_0,kme_1_indep,kme_0_indep,permute_e=False,permutations=250,device='cuda:0',debug_mode=False):
        super(counterfactual_me_test_correct, self).__init__(X,Y,e,perm_e,T,kme_1,kme_0,kme_1_indep,kme_0_indep,permute_e,permutations,device,debug_mode)
        self.og_indices=np.arange(X.shape[0])
        self.n_bins=X.shape[0]//20
        self.binner = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        numpy_e=e.cpu().numpy().squeeze()[:,np.newaxis]
        self.clusters = self.binner.fit_transform(numpy_e).squeeze()

    def get_permuted2d(self,ker):
        idx = permute(self.n_bins,self.og_indices,self.clusters)
        kernel_X = ker[:,idx]
        kernel_X = kernel_X[idx,:]
        return kernel_X,idx

    def permutation_test(self):
        perm_stats=[]
        running_err_0 = 0.
        running_err_1 = 0.
        for i in range(self.permutations):
            perm_L,idx = self.get_permuted2d(self.L)
            if self.permute_e:
                X = self.X[idx]
                if self.debug_mode:
                    perm_Y  = self.Y[idx]
                    perm_Y_0 = perm_Y[:self.Y_0.shape[0]]
                    perm_Y_1 = perm_Y[self.Y_0.shape[0]:]
                    perm_X_0 =X[:self.Y_0.shape[0]]
                    perm_X_1 =X[self.Y_0.shape[0]:]
                    perm_L_0 = self.kernel_L_0(perm_Y_0,perm_Y_0)
                    perm_L_1 = self.kernel_L_1(perm_Y_1,perm_Y_1)
                    a,b = self.sanity_check_estimates(perm_X_0,perm_X_1, self.kme_1_indep, perm_Y_0, perm_Y_1, perm_L_0,
                                                      perm_L_1)
                    running_err_0+=a
                    running_err_1+=b
            else:
                e=self.e
            tst = self.calculate_test_statistic(perm_L)
            perm_stats.append(tst.item())
        if self.debug_mode:
            print('average perm errors')
            print(running_err_0/self.permutations,running_err_1/self.permutations)
        return perm_stats,self.ref_stat.item()

class counterfactual_me_test_correct_sample(counterfactual_me_test):
    def __init__(self,
                 X,Y,e,perm_e,T,kme_1,kme_1_indep,permute_e=False,permutations=250,device='cuda:0',debug_mode=False):
        super(counterfactual_me_test_correct_sample, self).__init__(X,Y,e,perm_e,T,kme_1,kme_1_indep,permute_e,permutations,device,debug_mode)
        self.bernoulli_T = Bernoulli(probs=self.e)

    def recreate_all_weights(self,e,new_T):
        self.psi_0_weight=(e-new_T)/(1.-e)
        self.psi_1_weight=(new_T-e)/(e)
        self.T_1_weight=new_T/e
        self.T_0_weight = (1-new_T)/(1.-e)

    def sample_T(self):
        return self.bernoulli_T.sample()

    def permutation_test(self):
        perm_stats=[]
        for i in range(self.permutations):
            new_T = self.sample_T()
            self.recreate_all_weights(self.e,new_T)

            tst = self.calculate_test_statistic(self.L)
            perm_stats.append(tst.item())
        return perm_stats,self.ref_stat.item()

if __name__ == '__main__':
    pass