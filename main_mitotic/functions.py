import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import timeit
import torch.nn.functional as F
import torch.nn as nn
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
import INN
from scipy import integrate
from scipy.interpolate import CubicSpline
torch.set_default_dtype(torch.float64)

'''
hyperparameters
'''
pi = math.pi
def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)

class Sigmoid(torch.nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, data):
        # return data+torch.sin(data)**2
        return torch.tanh(data)

class NN(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(NN, self).__init__()
        torch.manual_seed(2)
        # self.snake = Snake()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            Sigmoid(),
            nn.Linear(n_hidden, n_hidden),
            Sigmoid(),
            # nn.Linear(n_hidden, n_hidden),
            # Sigmoid(),
            nn.Linear(n_hidden, n_output),

        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, data):
        return self.net(data)

class ODEFunc(nn.Module):

    def __init__(self,n_input, n_hidden, n_output):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            # nn.Linear(30, 30),
            # nn.Tanh(),
            nn.Linear(n_hidden, n_output),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

    def reverse(self, t, y):
        return -self.net(y)


class ShuffleModule(torch.nn.Module):
    '''
    alternate the order of the feature
    '''

    def __init__(self,dim):
        super(ShuffleModule, self).__init__()
        self.index = torch.randperm(dim)
        self.inverse_index = torch.argsort(self.index)

    def forward(self, x, log_p0=0, log_det_J=0):
        # The log(p_0) and log|det J| will not change under this transformation
        # if self.compute_p:
        #     return self.PixelUnshuffle(x), log_p0, log_det_J
        # else:
        #     return self.PixelUnshuffle(x)
        return x[:,self.index], log_p0, log_det_J
        # else:
        #     return self.PixelUnshuffle(x)

    def inverse(self, y, **args):
        return y[:,self.inverse_index]


class AE(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_latent,w,edim=3):
        super(AE, self).__init__()
        self.enc = NN(n_input, n_hidden, n_latent) # forward homeomorphism from limit-cycle to S^1
        self.dec = NN(n_latent, n_hidden, n_input) # backward homeomorphism from S^1 to limit-cycle
        # self.surface = NN(n_input, n_hidden, n_output) # surface map from the 'interior of limit-cycle' to the disk
        self.surface = INN.Sequential(INN.Nonlinear(edim, 'RealNVP'),ShuffleModule(edim),
                        INN.Nonlinear(edim, 'RealNVP'),ShuffleModule(edim),
                        INN.Nonlinear(edim, 'RealNVP'),ShuffleModule(edim),
                        INN.Nonlinear(edim, 'RealNVP'),ShuffleModule(edim),
                        INN.Nonlinear(edim, 'RealNVP'),ShuffleModule(edim),
                        INN.Nonlinear(edim, 'RealNVP'),ShuffleModule(edim),
                        INN.Nonlinear(edim, 'RealNVP'))

        # self.w = torch.tensor([1.5],requires_grad=True)
        # self.w = torch.nn.Parameter(torch.tensor([1.5]))
        self.w = torch.nn.Parameter(torch.tensor([w]))
        # self.reference_point = torch.tensor([[-0.5,-0.5]],requires_grad=True)
        self.reference_point = torch.zeros([n_input],requires_grad=True)


    def encoder(self,data):
        cir = self.enc(data)
        cir = cir/torch.norm(cir,p=2,dim=1).view(-1,1)
        phi = torch.angle(cir[:, 0] + 1j * cir[:, 1])
        phi_data = torch.autograd.grad(phi.sum(), data, create_graph=True)[0]
        return cir,phi,phi_data

    def decoder(self,phi):
        cir = torch.cat((torch.cos(phi), torch.sin(phi)),dim=1)
        chi = self.dec(cir)
        return chi

    def phase_sensitive_func(self,phi):
        '''
        :param phi: size, (\cdot,1) column input
        :return: Z(\phi), (\cdot,7), 7-D vector
        '''
        cir = torch.cat((torch.cos(phi),torch.sin(phi)),dim=1)
        chi = self.dec(cir)
        chi = chi.requires_grad_(True)
        cir = self.enc(chi)
        cir = cir/torch.norm(cir,p=2,dim=1).view(-1,1)
        phi = torch.angle(cir[:, 0] + 1j * cir[:, 1])
        # phi.requires_grad = True
        phi_data = torch.autograd.grad(phi.sum(), chi, create_graph=True)[0]
        return phi_data.detach()

    def forward(self,data):
        cir = self.enc(data)
        cir = cir/torch.norm(cir,p=2,dim=1).view(-1,1) # Constrain the latent state on the circle by construction
        # phi = torch.angle(torch.sin(cir[:,0])+1j*torch.sin(cir[:,1]))
        phi = torch.angle(cir[:, 0] + 1j * cir[:, 1]) # find the homeomorphism from the quotient space phi and the circle
        # phi_cir = torch.autograd.grad(phi.sum(), cir, create_graph=True)[0]
        # cir_data = torch.autograd.grad(cir.sum(), data, create_graph=True)[0]
        phi_data = torch.autograd.grad(phi.sum(), data, create_graph=True)[0]
        reverse_data = self.dec(cir)
        return cir,phi,phi_data,reverse_data




def cal_M(data):
    L = data.shape[1]
    b = data[:,1:L:7]
    bar_b = b.mean(dim=1)
    numerator = torch.var(bar_b)
    denominator = torch.mean(torch.var(b,dim=0))
    quotinent = numerator/denominator
    return quotinent

def cal_R(data):
    R = torch.mean(torch.exp(data*1j),dim=1)
    time_aver = torch.mean(torch.abs(R))
    return time_aver

def augment(data,edim=7):
    L,dim = data.shape[0],data.shape[1]
    new_data = torch.zeros([L,edim])
    new_data[:,:dim] = data
    return new_data

def angle_3d(data):
    data1 = torch.cat((data[1:],data[:1]),dim=0)
    inner_dot = torch.sum(data1*data,dim=1)
    data_norm = torch.linalg.norm(data,ord=2,dim=1)
    data1_norm = torch.linalg.norm(data1, ord=2, dim=1)
    cos_theta = inner_dot/(data_norm*data1_norm)
    theta = torch.arccos(cos_theta)
    return theta

def generate_interior_data(boundary,reference,k = 3):
    dim = boundary.shape[1]
    data = torch.zeros([(k+1)*len(boundary),dim])
    # reference = reference.view(1,-1)
    for j in range(len(boundary)):
        for m in range(k+1):
            c = m/k
            # data.append(boundary[j:j+1]*c+(1-c)*reference)
            data[j*(k+1)+m]=(boundary[j]*c+(1-c)*reference)
    return data


def disk_uniform(num,r1=0,r2=1.0):
    r = torch.rand([num])*(r2-r1)+r1
    theta = torch.rand([num])*math.pi*2
    data = torch.zeros([num,2])
    data[:,0] = torch.sqrt(r)*torch.cos(theta)
    data[:,1] = torch.sqrt(r)*torch.sin(theta)
    return data

def angle_(phi):
    phi = phi.view(-1,1)
    cir = torch.cos(phi)+1j*torch.sin(phi)
    phi_mod = torch.angle(cir)
    return phi_mod

def Z_(phi,model):
    true_cir = torch.cat((torch.cos(phi).view(-1, 1), torch.sin(phi).view(-1, 1)), dim=1)
    data = model.dec.forward(true_cir).requires_grad_(True)
    enc_cir,enc_phi,phi_data = model.encoder(data)
    return phi_data

def chi_(phi,model):
    true_cir = torch.cat((torch.cos(phi).view(-1, 1), torch.sin(phi).view(-1, 1)), dim=1)
    data = model.dec.forward(true_cir)
    return data

def H(x,y):
    tau = 0.1
    out = torch.zeros_like(x)
    out[:,0]=1/(1+torch.exp(-y[:,0]/tau))-1/(1+torch.exp(-x[:,0]/tau))
    return out

def H1(x,y): # linear coupling
    out = torch.zeros_like(x)
    out[:,0]=y[:,0]-x[:,0]
    return out

def H2(x,y): # chemical synaptic coupling
    g = 0.1
    k = 50
    alpha,beta=158/(360),(158+100)/(360)
    out = torch.zeros_like(x)
    phi = torch.arctan(y[:,1]/y[:,0])
    out[:,0] = g/(1+torch.exp(k*(alpha-phi))+torch.exp(k*(phi-beta)))
    return out


def step1(phi1,phi2,model,coupling_func):
    N1 = len(phi1)
    phi = torch.cat((phi1.view(-1,1),phi2.view(-1,1)),dim=0)
    phi_data = Z_(phi,model)
    chi = chi_(phi,model)
    gamma_12 = torch.sum(phi_data[:N1]*coupling_func(chi[:N1],chi[N1:]),dim=1)
    return gamma_12,angle_(phi1-phi2)

def permute(phi,i):
    phi = phi.view(-1,1)
    return torch.cat((phi[i:],phi[:i]),dim=0)


def Selkov_(input):
    gamma = 2.0
    alpha = 1.1
    scale_x = 2.0
    scale_y = 2.0
    output=torch.zeros_like(input)
    for i in range(len(input)):
        x, y = input[i,:]
        x, y = (x + 2.5) / scale_x, (y + 2.5) / scale_y
        dx = scale_x*(1.0 - x * y**gamma)
        dy = scale_y*(alpha*y*(x*y**(gamma-1)-1.0))
        output[i, :] = torch.tensor([dx, dy])
    return output



# train_data()
def pW_cal(a, b, p=2, metric='euclidean'):
    import ot
    """ Args:
            a, b: samples sets drawn from α,β respectively
            p: the coefficient in the OT cost (i.e., the p in p-Wasserstein)
            metric: the metric to compute cost matrix, 'euclidean' or 'cosine'
    """
    # cost matrix
    M = ot.dist(a, b, metric=metric)

    M = pow(M, p)

    # uniform distribution assumption
    alpha = ot.unif(len(a))
    beta = ot.unif(len(b))

    # p-Wasserstein Distance
    pW = ot.emd2(alpha, beta, M, numItermax=500000)
    pW = pow(pW, 1 / p)

    return pW

def ptp(x):
    return torch.max(x)-torch.min(x)

def get_batch(Data,num):
    length = len(Data)
    s = torch.from_numpy(np.random.choice(np.arange(length, dtype=np.int64),num, replace=False))
    batch_y = Data[s,:].requires_grad_(True)  # (M, D)
    return batch_y

colors = [
    [107/256,	161/256,255/256], # #6ba1ff
    [255/255, 165/255, 0],
    [233/256,	110/256, 248/256], # #e96eec
    # [0.6, 0.6, 0.2],  # olive
    # [0.5333333333333333, 0.13333333333333333, 0.3333333333333333],  # wine
    # [0.8666666666666667, 0.8, 0.4666666666666667],  # sand
    # [223/256,	73/256,	54/256], # #df4936
    [0.6, 0.4, 0.8], # amethyst
    [0.0, 0.0, 1.0], # ao
    [0.55, 0.71, 0.0], # applegreen
    # [0.4, 1.0, 0.0], # brightgreen
    [0.99, 0.76, 0.8], # bubblegum
    [0.93, 0.53, 0.18], # cadmiumorange
    [11/255, 132/255, 147/255], # deblue
    [204/255, 119/255, 34/255], # {ocra}
    [31/255,145/255,158/255],
    [117/255,157/255,219/255], #darkblue
    [233/255,108/255,102/255], #orange red
    [45/255,182/255,163/255] # cyan
]
colors = np.array(colors)