import torch


class Kernel(torch.nn.Module):
    def __init__(self,):
        super(Kernel, self).__init__()
    @staticmethod
    def sq_dist(x1,x2):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

        # Compute squared distance matrix using quadratic expansion
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))
        # Zero out negative values
        res.clamp_min_(0)

        # res  = torch.cdist(x1,x2,p=2)
        # Zero out negative values
        # res.clamp_min_(0)
        return res

    @staticmethod
    def covar_dist(x1, x2):
        return Kernel.sq_dist(x1,x2).sqrt()
    @staticmethod
    def get_median_ls(X,Y=None):
        with torch.no_grad():
            if Y is None:
                d = Kernel.covar_dist(x1=X, x2=X)
            else:
                d = Kernel.covar_dist(x1=X, x2=Y)
            ret = torch.sqrt(torch.median(d[d >= 0])) # print this value, should be increasing with d
            if ret.item()==0:
                ret = torch.tensor(1.0)
            return ret

class RBFKernel(Kernel):
    def __init__(self,x1=None,x2=None):
        super(RBFKernel, self).__init__()
        self.x1 = x1
        self.x2 = x2
        self.ls = 1.0

    

    def evaluate(self):

        if self.x2 is None:
            return torch.exp(-0.5*self.sq_dist(self.x1,self.x1)/self.ls**2)
        else:
            return torch.exp(-0.5*self.sq_dist(self.x1,self.x2)/self.ls**2)

    def forward(self,x1=None,x2=None):
        if x2 is None:
            return torch.exp(-0.5*self.sq_dist(x1,x1)/self.ls**2)
        else:
            return torch.exp(-0.5*self.sq_dist(x1,x2)/self.ls**2)

    def __matmul__(self, other):
        return self.evaluate()@other
       
    def _set_lengthscale(self,ls):
        self.ls = ls

class LinearKernel(Kernel):
    def __init__(self,x1=None,x2=None):
        super(LinearKernel, self).__init__()
        self.x1 = x1
        self.x2 = x2
        self.nu = 1.0

    def _set_lengthscale(self,nu):
        self.nu = nu

    def evaluate(self):
        if self.x2 is None:
            return self.nu * self.x1@self.x1.t()/self.divide
        else:
            return self.nu * self.x1@self.x2.t()/self.divide

    def forward(self,x1=None,x2=None):
        self.x1 = x1
        self.x2 = x2
        if self.x2 is None:
            self.divide = (torch.linalg.norm(self.x1,ord=float('inf'))**2)
        else:
            self.divide = (torch.linalg.norm(self.x1,ord=float('inf'))*torch.linalg.norm(self.x2,ord=float('inf')))
        return self

    def __matmul__(self, other):
        return self.evaluate()@other

# if __name__ == '__main__':
#     ker_1 = gpytorch.kernels.RBFKernel()
#     ker_1._set_lengthscale(1.0)
#
#     ker_2 = RBFKernel()
#     ker_2._set_lengthscale(1.0)
#     r = torch.randn(10,1)
#     print(ker_1(r).evaluate())
#     print(ker_2(r).evaluate())
