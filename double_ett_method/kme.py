import numpy as np

from doubly_robust_method.kernels import *
import tqdm
import torch
general_ker_obj = Kernel()
class learnable_kernel(torch.nn.Module):
    def __init__(self,kernel,ls,lamb):
        super(learnable_kernel, self).__init__()
        self.kernel = kernel
        self.lamb = torch.nn.Parameter(torch.tensor([lamb]).float(),requires_grad=True)
        self.ls = torch.nn.Parameter( torch.tensor([ls]).float(),requires_grad=True)
        self.kernel._set_lengthscale(self.ls)
        self.register_buffer('eye',torch.eye(self.kernel.x1.shape[0]))
    def evaluate(self):
        return self.kernel.evaluate()

    def forward(self,x1,x2):
        return self.kernel(x1,x2)

    def inverse(self):
        middle_ker = self.evaluate()
        n = middle_ker.shape[0]
        reg_middle_ker = middle_ker+self.eye * self.lamb
        rank = torch.linalg.matrix_rank(reg_middle_ker)
        if rank == n:
            inv = torch.inverse(reg_middle_ker)
        else:
            inv = torch.linalg.pinv(reg_middle_ker, hermitian=True)
        # print(n)
        # print()
        # L =torch.linalg.cholesky(middle_ker+self.eye * self.lamb)
        return inv,middle_ker

class kme_model():
    def __init__(self,X_tr,Y_tr,T_tr,X_val,Y_val,T_val,treatment_const,device='cuda:0'):
        self.device=device
        # self.T= T_tr.to(device)
        # self.T_0 =1-self.T_1
        self.X_tr = torch.from_numpy(X_tr[T_tr.squeeze()==treatment_const,:]).float().to(device)
        self.Y_tr = torch.from_numpy(Y_tr[T_tr.squeeze()==treatment_const]).float().to(device)

        self.n=X_tr.shape[0]

        self.X_val = torch.from_numpy(X_val[T_val.squeeze()==treatment_const,:]).float().to(device)
        self.Y_val = torch.from_numpy(Y_val[T_val.squeeze()==treatment_const]).float().to(device)


        self.eye = torch.eye(self.n).to(device)
        ls_Y =general_ker_obj.get_median_ls(self.Y_tr, self.Y_tr)
        ls_X =general_ker_obj.get_median_ls(self.X_tr, self.X_tr)
        self.k = RBFKernel(self.X_tr)
        self.l = RBFKernel(self.Y_tr)
        self.l._set_lengthscale(ls_Y.item())
        self.kernel=learnable_kernel(self.k, ls_X.item(), 1e-5).to(device)
        self.L_tr = self.l.evaluate()
        self.L_val = self.l(self.Y_val,self.Y_val)
        self.L_cross = self.l(self.Y_tr,self.Y_val)

    def calc_r2(self,val_error,y):
        return 1.-val_error/y.var()

    def update_loop(self,lamb=0.0):
        inv, middle_ker = self.kernel.inverse()
        # total_error = self.calculate_error( inv,middle_ker,self.L_tr)
        # self.opt.zero_grad()
        # total_error.backward()
        # self.opt.step()
        val_r2 = self.calculate_validation_error(inv)
        # val_r2 = self.calc_r2(val_error, self.Y_val)
        if val_r2.item()<self.best:
            self.best =val_r2.item()
            self.inv = inv
            self.best_lamb = lamb
        # else:
        #     self.count+=1§

    def fit(self,its =10,patience=10):
        #TODO: Neural network regressor
        #Better cross val!
        self.best = np.inf
        self.patience=patience
        self.count=0
        self.kernel.ls.requires_grad=False
        self.kernel.lamb.requires_grad=False
        # self.opt= torch.optim.Adam(self.kernel.parameters(),lr=1e-2)
        list_of_lamb = np.linspace(0,1e-2,20).tolist()
        for lamb in list_of_lamb:
            self.kernel.lamb[0] = lamb
            # for i in range(its):
            self.update_loop(lamb)

            # for j in range(5):
            #     self.kernel.lamb.require_grad=False
            #     self.kernel.ls.require_grad=True
            #     self.update_loop(opt)
            #     if self.count>self.patience:
            #         return
            # for k in range(5):
            #     self.kernel.ls.require_grad = False
            #     self.kernel.lamb.require_grad=True
            #     self.update_loop(opt)
            #     if self.count>self.patience:
            #         return
        return

    def get_embedding(self,realize_Y,ref_X_te):
        k_x = self.kernel(ref_X_te,self.X_tr) #mxn
        L = self.l(realize_Y,self.Y_tr) #j x n
        w=k_x@self.inv #nxn #Should be some sort of inner product no? # mxn @ nxn = mxn
         # m*n , jxn
        return  L @ w.t()  #jxm

    def get_middle_ker(self,X):
        middle_ker = self.kernel(self.X_tr, X)
        return middle_ker

    def calculate_validation_error(self,inv):
        with torch.no_grad():
            middle_ker = self.kernel(self.X_tr,self.X_val)
            tr_error_1 = self.calculate_error(inv,middle_ker,self.L_val,self.L_cross)
            return tr_error_1

    def calculate_error(self,inv,middle_ker,L,L_cross=None):
        if L_cross is None:
            L_cross = self.L_tr
        tmp_calc = inv@middle_ker
        term_1 = ((tmp_calc.t()@self.L_tr) * tmp_calc.t()).sum(dim=1)
        term_2 =torch.diag(L)
        term_3 = -2*( tmp_calc*L_cross).sum(dim=0)
        error = term_1 + term_2 + term_3
        return error.mean()

    #TODO: needs fixing wow
    def get_psi_part(self, x_te, w_te):
        with torch.no_grad():
            middle_ker = self.kernel(x_te,self.X_tr)
            tmp_val = (w_te.t()@middle_ker)@self.inv
        return tmp_val


    def get_psi_square_term(self, tmp_val):
        return (tmp_val.t()@self.L_tr)@tmp_val


    def get_psi_cross_term(self, psi_part,Y_te,T_te):
        b_term = self.l(self.Y_tr,Y_te)@T_te
        return psi_part.t()@b_term

    def get_weird_cross_term(self,left,right,other_Y_tr):
        b_term = self.l(self.Y_tr,other_Y_tr)@right
        return left.t()@b_term

    def calculate_error_external(self,X,Y,L):
        mid_ker = self.get_middle_ker(X)
        cross = self.l(self.Y_tr,Y)
        return self.calculate_error(self.inv,mid_ker,L,cross)






