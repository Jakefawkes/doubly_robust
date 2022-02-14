import torch.nn

from doubly_robust_method.propensity_classifier import *
from doubly_robust_method.kernels import *
from doubly_robust_method.kme import *
# class feature_map_kernel():
#     def __init__(self):

#RFF's can be done, but not sure it's going to be that much better TBH? It's just a scalable version of kernels... in case you wanna be fully non-parametric


class feature_map(torch.nn.Module):
    def __init__(self,
                 d_in_x,
                 cat_size_list,
                 layers_x,
                 dropout=0.9,
                 transformation=torch.tanh,
                 output_dim=10,
                 ):
        super(feature_map, self).__init__()
        self.output_dim=output_dim
        self.init_covariate_net(d_in_x,layers_x,cat_size_list,transformation,dropout,output_dim)

    def identity_transform(self, x):
        return x

    def init_covariate_net(self,d_in_x,layers_x,cat_size_list,transformation,dropout,output_dim):
        module_list = [nn_node(d_in=d_in_x,d_out=layers_x[0],cat_size_list=cat_size_list,transformation=transformation,dropout=dropout)]
        for l_i in range(1,len(layers_x)):
            module_list.append(nn_node(d_in=layers_x[l_i-1],d_out=layers_x[l_i],cat_size_list=[],transformation=transformation,dropout=dropout))
        self.covariate_net = multi_input_Sequential(*module_list)

        if len(layers_x)==1:
            self.final_layer = self.identity_transform
        else:
            self.final_layer = torch.nn.Linear(layers_x[-1],output_dim)

    def forward(self,x_cov,x_cat=[]):
        return self.final_layer(self.covariate_net((x_cov,x_cat)))

class inverse_hack(torch.nn.Module):
    def __init__(self,N,r):
        super(inverse_hack, self).__init__()
        self.C = torch.nn.Parameter(torch.randn(N,r))

    def forward(self,b):
        return self.C.t()@(self.C@b)

    def full_forward(self,b):
        return self.C@(self.C.t()@b)

# def congjuate_gradient_solver():
#     pass

class neural_kme_model():
    def __init__(self,X_tr,Y_tr,T_tr,X_val,Y_val,T_val,
                 treatment_const,
                 neural_net_parameters,
                 approximate_inverse=False,permutation_training=False,device='cuda:0'):

        self.N = X_tr.shape[0]
        self.X_tr, self.Y_tr,self.X_val,self.Y_val=X_tr, Y_tr,X_val,Y_val
        self.mask_tr = T_tr.squeeze()==treatment_const
        self.mask_val = T_val.squeeze()==treatment_const
        self.X_tr = torch.from_numpy(X_tr[self.mask_tr,:]).float().to(device)
        self.Y_tr = torch.from_numpy(Y_tr[self.mask_tr]).float().to(device)
        self.X_val = torch.from_numpy(X_val[self.mask_val,:]).float().to(device)
        self.Y_val = torch.from_numpy(Y_val[self.mask_val]).float().to(device)


        ls_Y =general_ker_obj.get_median_ls(self.Y_tr, self.Y_tr)
        self.l = RBFKernel(self.Y_tr)
        self.l._set_lengthscale(ls_Y.item())
        self.L_tr = self.l.evaluate()
        self.L_val = self.l(self.Y_val,self.Y_val)
        self.L_cross = self.l(self.Y_tr,self.Y_val)


        self.n=X_tr.shape[0]
        self.r = neural_net_parameters['output_dim']
        self.eye = torch.eye(self.r).to(device)
        self.feature_map = feature_map(**neural_net_parameters)
        self.approximate_inverse = approximate_inverse
        self.permutation_training = permutation_training

        if self.permutation_training:
            self.inverse_hack = inverse_hack(self.n,self.r)

    def reset_data(self,X_tr,Y_tr,X_val,Y_val):
        self.X_tr = X_tr[self.mask_tr,:]
        self.Y_tr = Y_tr[self.mask_tr]
        self.X_val =X_val[self.mask_val,:]
        self.Y_val = Y_val[self.mask_val]
        ls_Y =general_ker_obj.get_median_ls(self.Y_tr, self.Y_tr)
        self.l = RBFKernel(self.Y_tr)
        self.l._set_lengthscale(ls_Y.item())
        self.L_tr = self.l.evaluate()
        self.L_val = self.l(self.Y_val,self.Y_val)
        self.L_cross = self.l(self.Y_tr,self.Y_val)

    # def calculate_kernel(self,feature_map_1,feature_map_2,x1,x2):
    #     y_1 = feature_map_1(x1) #Nxd
    #     y_2 = feature_map_2(x2) #Mxd
    #     return y_1@(y_2.t())
    #
    # def kmvm_symmetric(self,feature_map_1,x1,b):
    #     y_1 = feature_map_1(x1) #Nxd
    #     return y_1@(y_1.t()@b)

    def calculate_operator(self):
        self.x_map_tr=self.feature_map(self.X_tr) #Nxr

        if self.approximate_inverse:
            inverse =  self.inverse_hack()+ self.eye * self.lamb
            self.store_part =inverse@self.x_map_tr.t()
        else:
            inverse = self.x_map_tr.t() @ self.x_map_tr + self.eye * self.lamb
            self.store_part = torch.inverse(inverse)@self.x_map_tr.t()

        # return x_map_2@self.store_part
        # tmp_calc = inv @ middle_ker

    def calculate_error(self,X_ref,L,L_cross=None):
        if L_cross is None:
            L_cross = self.L_tr
        if self.X_tr==X_ref:
            x_map_2 = self.x_map_tr
        else:
            x_map_2 = self.feature_map(X_ref)
        tmp_calc= x_map_2@self.store_part
        term_1 = ((tmp_calc.t()@self.L_tr) * tmp_calc.t()).sum(dim=1)
        term_2 =torch.diag(L)
        term_3 = -2*( tmp_calc*L_cross).sum(dim=0)
        error = term_1 + term_2 + term_3
        return error.mean()

    def calculate_validation_error(self,X_val):
        with torch.no_grad():
            tr_error_1 = self.calculate_error(X_val,self.L_val,self.L_cross)
            return tr_error_1

    def get_psi_part(self, x_te, T_te):
        with torch.no_grad():
            x_map_2 = self.feature_map(x_te)
            tmp_val = (T_te.t()@x_map_2)@self.store_part
        return tmp_val

    def get_psi_square_term(self, tmp_val):
        return (tmp_val.t()@self.L_tr)@tmp_val

    def get_psi_cross_term(self, psi_part,Y_te,T_te):
        b_term = self.l(self.Y_tr,Y_te)@T_te
        return psi_part.t()@b_term
    def get_weird_cross_term(self,left,right,other_Y_tr):
        b_term = self.l(self.Y_tr,other_Y_tr)@right
        return left.t()@b_term

    def update_loop(self,lamb=0.0):
        self.calculate_operator()
        total_error = self.calculate_error( self.X_tr,self.L_tr)
        val_error = self.calculate_validation_error(self.X_val)
        if val_error.item()<self.best:
            self.best =val_error.item()
            self.best_lamb = lamb

        self.opt.zero_grad()
        total_error.backward()
        self.opt.step(total_error)

    def fit(self,its =25,patience=10):
        self.best = np.inf
        self.patience = patience
        self.count = 0
        if self.approximate_inverse:
            self.opt= torch.optim.Adam(list(self.feature_map.parameters())+list(self.inverse_hack.parameters()),lr=1e-2)
        else:
            self.opt= torch.optim.Adam(self.feature_map.parameters(),lr=1e-2)
        list_of_lamb = np.linspace(0, 1, 20).tolist()
        for lamb in list_of_lamb:
            self.lamb= lamb
            for i in range(its):
                if self.permutation_training:
                    idx = torch.randperm(self.N)
                    X_tr, Y_tr, X_val, Y_val=self.X_tr[idx,:], self.Y_tr[idx,:], self.X_val[idx,:], self.Y_val[idx,:]
                    self.reset_data(X_tr, Y_tr, X_val, Y_val)
                self.update_loop(lamb)
        self.lamb = self.best_lamb
        self.calculate_operator()
        self.store_part.detach()
        self.store_part.requires_grad = False
        return

    #



