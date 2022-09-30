import copy

from torch.utils.data.dataset import Dataset
from pycox.preprocessing.feature_transforms import *
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
def categorical_transformer(X,cat_cols,cont_cols):
    c = OrderedCategoricalLong()
    for el in cat_cols:
        X[:,el] = c.fit_transform(X[:,el])
    cat_cols = cat_cols
    if cat_cols:
        unique_cat_cols = X[:,cat_cols].max(axis=0).tolist()
        unique_cat_cols = [el + 1 for el in unique_cat_cols]
    else:
        unique_cat_cols = []
    X_cont=X[cont_cols]
    X_cat=X[cat_cols]
    return X_cont,X_cat,unique_cat_cols

class general_custom_dataset(Dataset):
    def __init__(self,X,y,x_cat=[]):
        super(general_custom_dataset, self).__init__()
        self.split(X=X,y=y,X_cat=x_cat,mode='train')

    def split(self,X,y,mode='train',X_cat=[]):
        setattr(self,f'{mode}_y', torch.from_numpy(y).float())
        setattr(self, f'{mode}_X', torch.from_numpy(X).float())
        self.cat_cols = False
        if not isinstance(X_cat,list):
            self.cat_cols = True
            setattr(self, f'{mode}_cat_X', torch.from_numpy(X_cat.astype('int64').values).long())

    def set(self,mode='train'):
        self.X = getattr(self,f'{mode}_X')
        self.y = getattr(self,f'{mode}_y')
        if self.cat_cols:
            self.cat_X = getattr(self,f'{mode}_cat_X')
        else:
            self.cat_X = []

    def transform_x(self,x):
        return self.x_mapper.transform(x)

    def invert_duration(self,duration):
        return self.duration_mapper.inverse_transform(duration)

    def transform_duration(self,duration):
        return self.duration_mapper.transform(duration)

    def __getitem__(self, index):
        if self.cat_cols:
            return self.X[index,:],self.cat_X[index,:],self.y[index]
        else:
            return self.X[index,:],self.cat_X,self.y[index]

    def __len__(self):
        return self.X.shape[0]

class chunk_iterator():
    def __init__(self,X,y,cat_X,shuffle,batch_size):
        self.X = X
        self.y = y
        self.cat_X = cat_X
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n = self.X.shape[0]
        self.chunks=self.n//batch_size+1
        self.perm = torch.randperm(self.n)
        self.valid_cat = not isinstance(self.cat_X, list)
        if self.shuffle:
            self.X = self.X[self.perm,:]
            self.y = self.y[self.perm,:]
            if self.valid_cat: #F
                self.cat_X = self.cat_X[self.perm,:]
        self._index = 0
        self.it_X = torch.chunk(self.X,self.chunks)
        self.it_y = torch.chunk(self.y,self.chunks)
        if self.valid_cat:  # F
            self.it_cat_X = torch.chunk(self.cat_X,self.chunks)
        else:
            self.it_cat_X = []
        self.true_chunks = len(self.it_X)

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < self.true_chunks:
            if self.valid_cat:
                result = (self.it_X[self._index],self.it_cat_X[self._index],self.it_y[self._index])
            else:
                result = (self.it_X[self._index],[],self.it_y[self._index])
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration

    def __len__(self):
        return len(self.it_X)

class custom_dataloader():
    def __init__(self,dataset,batch_size=32,shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = self.dataset.train_X.shape[0]
        self.len=self.n//batch_size+1
    def __iter__(self):
        return chunk_iterator(X =self.dataset.X,
                              y = self.dataset.y,
                              cat_X = self.dataset.cat_X,
                              shuffle = self.shuffle,
                              batch_size=self.batch_size,
                              )
    def __len__(self):
        self.n = self.dataset.X.shape[0]
        self.len = self.n // self.batch_size + 1
        return self.len



class multi_input_Sequential(torch.nn.Sequential):
    def forward(self, inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class multi_input_Sequential_res_net(torch.nn.Sequential):
    def forward(self, inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                output = module(inputs)
                if inputs.shape[1]==output.shape[1]:
                    inputs = inputs+output
                else:
                    inputs = output
        return inputs

class nn_node(torch.nn.Module): #Add dropout layers, Do embedding layer as well!
    def __init__(self,d_in,d_out,cat_size_list,dropout=0.1,transformation=torch.tanh):
        super(nn_node, self).__init__()

        self.has_cat = len(cat_size_list)>0
        self.latent_col_list = []
        print('cat_size_list',cat_size_list)
        for i,el in enumerate(cat_size_list):
            col_size = el//2+2
            setattr(self,f'embedding_{i}',torch.nn.Embedding(el,col_size))
            self.latent_col_list.append(col_size)
        self.w = torch.nn.Linear(d_in+sum(self.latent_col_list),d_out)
        self.f = transformation
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,X,x_cat=[]):
        if not isinstance(x_cat,list):
            seq = torch.unbind(x_cat,1)
            cat_vals = [X]
            for i,f in enumerate(seq):
                o = getattr(self,f'embedding_{i}')(f)
                cat_vals.append(o)
            X = torch.cat(cat_vals,dim=1)
        return self.dropout(self.f(self.w(X)))

class classifier_binary(torch.nn.Module):
    def __init__(self,
                 d_in_x,
                 cat_size_list,
                 layers_x,
                 dropout=0.9,
                 transformation=torch.tanh,
                 ):
        super(classifier_binary, self).__init__()
        self.init_covariate_net(d_in_x,layers_x,cat_size_list,transformation,dropout)

    def identity_transform(self, x):
        return x

    def init_covariate_net(self,d_in_x,layers_x,cat_size_list,transformation,dropout):
        module_list = [nn_node(d_in=d_in_x,d_out=layers_x[0],cat_size_list=cat_size_list,transformation=transformation,dropout=dropout)]
        for l_i in range(1,len(layers_x)):
            module_list.append(nn_node(d_in=layers_x[l_i-1],d_out=layers_x[l_i],cat_size_list=[],transformation=transformation,dropout=dropout))
        self.covariate_net = multi_input_Sequential(*module_list)

        if len(layers_x)==1:
            self.final_layer = self.identity_transform
        else:
            self.final_layer = torch.nn.Linear(layers_x[-1],1)

    def forward(self,x_cov,x_cat=[]):
        return self.final_layer(self.covariate_net((x_cov,x_cat)))

    def predict(self,x_cov,x_cat=[]):
        return torch.sigmoid(self.final_layer(self.covariate_net((x_cov,x_cat))))


class sklearn_propensity_estimator():
    def __init__(self, X_tr, T_tr, X_val, T_val, nn_params, bs=100, epochs=100, device='cuda:0', X_cat_tr=[],
                 X_cat_val=[]):
        self.X  = X_tr
        self.T  = T_tr.squeeze()
        self.clf = LogisticRegressionCV(cv=10, random_state=0,verbose=False)
        self.X_val = X_val
        self.T_val = T_val.squeeze()
        self.device = device
    def fit(self,tmp=None):
        self.clf.fit(self.X,self.T)
        self.best = self.clf.score(self.X_val,self.T_val)

    def predict(self,X,T=[],x_cat=[]):
        return torch.from_numpy(self.clf.predict_proba(X)[:,-1]).float().to(self.device).unsqueeze(-1)


class propensity_estimator():
    def __init__(self,X_tr,T_tr,X_val,T_val,nn_params,bs=100,epochs=100,device='cuda:0',X_cat_tr=[],X_cat_val=[]):
        self.epochs =epochs
        self.device=device
        self.model = classifier_binary(**nn_params).to(self.device)
        self.n = X_tr.shape[0]
        self.pos_count = np.sum(T_tr)
        self.neg_count = np.sum(T_tr==0)
        self.pos_weight = torch.tensor(self.neg_count/self.pos_count).float()
        self.bs=bs
        self.dataset_tr = general_custom_dataset(X_tr,T_tr,X_cat_tr)
        self.dataset_tr.set('train')
        self.dataset_val = general_custom_dataset(X_val,T_val,X_cat_val)
        self.dataset_val.set('train')
        self.dataloader_tr = custom_dataloader(dataset=self.dataset_tr,batch_size=bs,shuffle=True)
        self.dataloader_val = custom_dataloader(dataset=self.dataset_val,batch_size=bs,shuffle=False)
        self.best_model=None
    def predict(self,X_test,T_tst,X_cat_test):
        dataset = general_custom_dataset(X_test,T_tst,X_cat_test)
        dataset.set('train')
        dataloader = custom_dataloader(dataset=dataset, batch_size=self.bs, shuffle=False)
        preds,_= self.val_loop(dataloader)
        return preds
    def score_auc(self,pred_val,T_val):
        pred = pred_val.cpu().numpy()
        y = T_val.cpu().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        return auc

    def train_loop(self,opt,obj):
        self.model.train()
        for i,(x,x_cat,y) in enumerate(self.dataloader_tr):
            x=x.to(self.device)
            y=y.to(self.device)
            if not isinstance(x_cat,list):
                x_cat=x_cat.to(self.device)
            y_pred = self.model(x,x_cat)
            loss=obj(y_pred,y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def val_loop(self,dl):
        self.model.eval()
        y_s = []
        y_preds =[]
        for i,(x,x_cat,y) in enumerate(dl):
            x=x.to(self.device)
            if not isinstance(x_cat,list):
                x_cat=x_cat.to(self.device)
            with torch.no_grad():
                y_pred = self.model.predict(x,x_cat)
            y_s.append(y)
            y_preds.append(y_pred)
        return torch.cat(y_preds),torch.cat(y_s)


    def fit(self,patience=10):
        self.best = 0.5
        self.patience=patience
        counter=0
        objective = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        opt = torch.optim.Adam(params=self.model.parameters(),lr=1e-2)
        self.best_model = copy.deepcopy(self.model)
        for i in range(self.epochs):
            self.train_loop(opt,objective)
            y_preds,ys = self.val_loop(self.dataloader_val)
            auc = self.score_auc(y_preds,ys)

            if auc> self.best:
                self.best =auc
                self.best_model=copy.deepcopy(self.model)
                counter=0
            else:
                counter+=1
            if counter>self.patience:
                self.model = self.best_model
                return
        self.model=self.best_model






