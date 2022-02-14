import torch
from doubly_robust_method.kernels import *
from doubly_robust_method.kme import *
general_ker_obj =Kernel()

class counterfactual_me_test():
    def __init__(self,X,Y,e,T,kme_1,kme_0,kme_1_indep,kme_0_indep,permute_e=False,permutations=250,device='cuda:0'):
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

        self.X_1 = self.X[mask_1,:]
        self.X_0 = self.X[~mask_1,:]

        self.kme_1=kme_1
        self.kme_0=kme_0
        self.kme_1_indep=kme_1_indep
        self.kme_0_indep=kme_0_indep
        self.permute_e = permute_e


        self.setup_Y_kernel(self.Y,'L',device)
        self.setup_Y_kernel(self.Y_0,'L_0',device)
        self.setup_Y_kernel(self.Y_1,'L_1',device)


        # self.kernel = RBFKernel(self.Y).to(device)
        # self.ls =general_ker_obj.get_median_ls(self.Y, self.Y)
        # self.kernel._set_lengthscale(self.ls)
        # self.L = self.kernel.evaluate()


        self.create_all_weights(self.e)
        self.calculate_psi_ref()
        self.ref_stat = self.calculate_test_statistic(self.L)
        print('ref errors')
        print(self.sanity_check_estimates(self.kme_0,self.kme_1,self.Y_0,self.Y_1, self.L_0,self.L_1))

    def setup_Y_kernel(self,Y,name,device):
        kernel = RBFKernel(Y).to(device)
        ls =general_ker_obj.get_median_ls(Y,Y)
        kernel._set_lengthscale(ls)
        L = kernel.evaluate()
        setattr(self,name,L)
        setattr(self,'kernel_'+name,kernel)
    #permutation should occur twice
    def calculate_psi_ref(self):
        self.psi_1 = self.kme_1.get_psi_part(self.X, self.psi_1_weight).t()
        self.psi_0 = self.kme_0.get_psi_part(self.X, self.psi_0_weight).t()


    def sanity_check_estimates(self,kme_0,kme_1,Y_0,Y_1,L_0,L_1):
        mid_ker_0 = kme_0.get_middle_ker(self.X_0)
        cross_0 = kme_0.l(kme_0.Y_tr,Y_0)
        error_0 = kme_0.calculate_error(kme_0.inv, mid_ker_0,L_0,cross_0)
        cross_1 = kme_1.l(kme_1.Y_tr,Y_1)
        mid_ker_1 = kme_1.get_middle_ker(self.X_1)
        error_1 = kme_1.calculate_error(kme_1.inv, mid_ker_1,L_1,cross_1)
        return error_0.item(),error_1.item()

    def calculate_psi(self):
        self.psi_1 = self.kme_1_indep.get_psi_part(self.X, self.psi_1_weight).t()
        self.psi_0 = self.kme_0_indep.get_psi_part(self.X, self.psi_0_weight).t()
    def calculate_test_statistic(self,L):
        T_1_L_test = L@self.T_1_weight
        T_0_L_test = L@self.T_0_weight
        term_1 = T_1_L_test.t()@self.T_1_weight
        term_2 = self.kme_1.get_psi_square_term(self.psi_1)
        term_3 = -2*self.kme_1.get_psi_cross_term(self.psi_1,self.Y,self.T_1_weight)
        term_4 = T_0_L_test.t()@self.T_0_weight
        term_5 = self.kme_0.get_psi_square_term(self.psi_0)
        term_6 = -2*self.kme_0.get_psi_cross_term(self.psi_0,self.Y,self.T_0_weight)
        term_7 = -2*T_1_L_test.t()@T_0_L_test
        term_8 = 2*self.kme_0.get_psi_cross_term(self.psi_0,self.Y,self.T_1_weight)
        term_9 = 2*self.kme_1.get_psi_cross_term(self.psi_1,self.Y,self.T_0_weight)
        term_10 = -2*self.kme_0.get_weird_cross_term(left=self.psi_0,right=self.psi_1,other_Y_tr=self.kme_1.Y_tr)
        tot = term_1+term_2+term_3+term_4+term_5+term_6+term_7+term_8+term_9+term_10
        return 1/self.n**2 * tot
        # return self.omega.t()@(L@self.omega).item()

    def create_all_weights(self,e):
        self.psi_0_weight=(e-self.T_1)/(1.-e)
        self.psi_1_weight=(self.T_1-e)/(e)
        self.T_1_weight=self.T_1/e
        self.T_0_weight = self.T_0/(1.-e)

    def get_permuted2d(self,ker):
        idx = torch.randperm(self.n)
        kernel_X = ker[:,idx]
        kernel_X = kernel_X[idx,:]
        return kernel_X,idx

    def permutation_test(self):
        perm_stats=[]
        for i in range(self.permutations):
            perm_L,idx = self.get_permuted2d(self.L)
            if self.permute_e:
                e=self.e[idx]
                self.create_all_weights(e)
                self.calculate_psi()

                perm_Y  = self.Y[idx]
                perm_Y_0 = perm_Y[:self.Y_0.shape[0]]
                perm_Y_1 = perm_Y[self.Y_0.shape[0]:]
                perm_L_0 = self.kernel_L_0(perm_Y_0,perm_Y_0)
                perm_L_1 = self.kernel_L_1(perm_Y_1,perm_Y_1)
                print('perm errors')
                print(self.sanity_check_estimates(self.kme_0_indep, self.kme_1_indep, perm_Y_0, perm_Y_1, perm_L_0,
                                                  perm_L_1))

            else:
                e=self.e
            tst = self.calculate_test_statistic(perm_L)
            perm_stats.append(tst.item())
        return perm_stats,self.ref_stat.item()

if __name__ == '__main__':
    pass