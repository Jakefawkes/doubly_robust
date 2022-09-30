import torch.nn

from doubly_robust_method.propensity_classifier import *
from doubly_robust_method.kernels import *
from doubly_robust_method.kme import *
# class feature_map_kernel():
#     def __init__(self):
import copy
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
        self.C = torch.nn.Parameter(torch.randn(N,r)*0.0001)

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
        self.device = device
        self.N_tr = X_tr.shape[0]
        self.N_val = X_val.shape[0]
        self.X_tr_og, self.Y_tr_og,self.X_val_og,self.Y_val_og=X_tr, Y_tr,X_val,Y_val
        self.mask_tr = T_tr.squeeze()==treatment_const
        self.mask_val = T_val.squeeze()==treatment_const
        self.X_tr = torch.from_numpy(X_tr[self.mask_tr,:]).float().to(device)
        self.Y_tr = torch.from_numpy(Y_tr[self.mask_tr]).float().to(device)
        self.X_val = torch.from_numpy(X_val[self.mask_val,:]).float().to(device)
        self.Y_val = torch.from_numpy(Y_val[self.mask_val]).float().to(device)

        ls_Y =general_ker_obj.get_median_ls(self.Y_tr)
        self.l = RBFKernel(self.Y_tr)
        self.l._set_lengthscale(ls_Y.item())
        self.L_tr = self.l.evaluate()
        self.L_val = self.l(self.Y_val,self.Y_val)
        self.L_cross = self.l(self.Y_tr,self.Y_val)

        self.n=X_tr.shape[0]
        neural_net_parameters['d_in_x'] = X_tr.shape[1]
        self.r = neural_net_parameters['output_dim']
        self.eye = torch.eye(self.r).to(device)
        self.feature_map = feature_map(**neural_net_parameters).to(device)
        self.approximate_inverse = approximate_inverse
        self.permutation_training = permutation_training

        if self.approximate_inverse:
            self.inverse_hack = inverse_hack(self.n,self.r).to(device)

    def reset_data(self,X_tr,Y_tr,X_val,Y_val):
        self.X_tr = torch.from_numpy(X_tr[self.mask_tr,:]).float().to(self.device)
        self.Y_tr = torch.from_numpy(Y_tr[self.mask_tr]).float().to(self.device)
        self.X_val = torch.from_numpy(X_val[self.mask_val,:]).float().to(self.device)
        self.Y_val = torch.from_numpy(Y_val[self.mask_val]).float().to(self.device)
        ls_Y =general_ker_obj.get_median_ls(self.Y_tr)
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
            # inverse = + self.eye * self.lamb
            self.store_part = self.inverse_hack(self.x_map_tr.t())
        else:
            inverse = self.x_map_tr.t() @ self.x_map_tr + self.eye * self.lamb
            self.store_part = torch.inverse(inverse)@self.x_map_tr.t()

        # return x_map_2@self.store_part
        # tmp_calc = inv @ middle_ker

    def calculate_error(self,X_ref,L,L_cross=None):
        if L_cross is None:
            L_cross = self.L_tr
        if torch.equal(self.X_tr,X_ref):
            x_map_2 = self.x_map_tr
        else:
            x_map_2 = self.feature_map(X_ref)
        tmp_calc= x_map_2@self.store_part
        term_1 = ((tmp_calc@self.L_tr) * tmp_calc).sum(dim=1)
        term_2 =torch.diag(L)
        term_3 = -2*( tmp_calc.t()*L_cross).sum(dim=0)
        error = term_1 + term_2 + term_3
        return error.mean()

    def assign_best_model(self):
        self.best_model = copy.deepcopy(self.feature_map)
        if self.approximate_inverse:
            self.best_hack = copy.deepcopy(self.inverse_hack)


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
            self.assign_best_model()
        self.opt.zero_grad()
        total_error.backward()
        self.opt.step()

    def fit(self,its =200,patience=10):
        self.best = np.inf
        self.patience = patience
        self.count = 0
        if self.approximate_inverse:
            self.opt= torch.optim.Adam(list(self.feature_map.parameters())+list(self.inverse_hack.parameters()),lr=1e-2)
        else:
            self.opt= torch.optim.Adam(self.feature_map.parameters(),lr=1e-2)
        # list_of_lamb = np.linspace(0, 1, 20).tolist()
        self.best_model=self.feature_map
        list_of_lamb=[1e-3]
        for lamb in list_of_lamb:
            self.lamb= lamb
            for i in range(its):
                if self.permutation_training:
                    idx_tr = torch.randperm(self.N_tr)
                    idx_val = torch.randperm(self.N_val)
                    X_tr, Y_tr, X_val, Y_val=self.X_tr_og[idx_tr,:], self.Y_tr_og[idx_tr,:], self.X_val_og[idx_val,:], self.Y_val_og[idx_val,:]
                    self.reset_data(X_tr, Y_tr, X_val, Y_val)
                self.update_loop(lamb)
        self.feature_map=self.best_model
        if self.approximate_inverse:
            self.inverse_hack=self.best_hack
        self.lamb = self.best_lamb
        self.calculate_operator()
        self.store_part=self.store_part.detach()
        self.store_part.requires_grad = False
        return

    def calculate_error_external(self,X,Y,L):
        cross = self.l(self.Y_tr,Y)
        return self.calculate_error(X,L,cross)


    def get_embedding(self,X_te,Y_te):
        x_map_2 = self.feature_map(X_te)
        L = self.l(Y_te,self.Y_tr)
        w=x_map_2@self.store_part #Should be some sort of inner product no?
        return w*L

