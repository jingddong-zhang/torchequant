import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import timeit
import torch.nn.functional as F
import torch.nn as nn
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint
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
            nn.Linear(n_hidden, n_hidden),
            Sigmoid(),
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

    def dphi_reverse(self,phi):
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
        return chi,phi_data.detach()


    # def hessian(self,data):
    #     cir = self.enc(data)
    #     cir = cir / torch.norm(cir, p=2, dim=1).view(-1, 1)  # Constrain the latent state on the circle by construction
    #     # phi = torch.angle(torch.sin(cir[:,0])+1j*torch.sin(cir[:,1]))
    #     phi = torch.angle(
    #         cir[:, 0] + 1j * cir[:, 1])  # find the homeomorphism from the quotient space phi and the circle
    #     # phi_cir = torch.autograd.grad(phi.sum(), cir, create_graph=True)[0]
    #     # cir_data = torch.autograd.grad(cir.sum(), data, create_graph=True)[0]
    #     phi_data = torch.autograd.grad(phi.sum(), data, create_graph=True)[0]
    #     reverse_data = self.dec(cir)
    #     return cir, phi, phi_data, reverse_data

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



class ECC(torch.nn.Module):
    def __init__(self, AE,edim=7):
        super(ECC, self).__init__()
        self.AE = AE
        self.reference_point = torch.tensor([[-0.5,-0.5]],requires_grad=True)
        self.equant = self.AE.surface.inverse(torch.zeros([1,edim]))

    def couple(self,data_i,data_j):
        '''
        :param data_i: x_i
        :param data_j: x_j
        :return: coupling function H(x_i,x_j)
        '''
        H = torch.zeros_like(data_j)
        H[:,-1] = data_j[:,-1]-data_i[:,-1]
        return H

    def couple_1(self,data_i,data_j):
        N = 100
        H = torch.zeros_like(data_j)
        H[:, -1] = data_j[:, -1] + data_i[:, -1] / (N - 1)
        return H

    def w_func(self,data):
        '''
        :param data: limit cycle data, (T,dim)
        :return: vector field related to the \Delta\beta, (0,0,0,a-A,b-B,c-C,0)
        '''
        a, b, c, A, B, C, S = data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],data[:,6]
        output = torch.zeros_like(data)
        output[:,3] = a - A
        output[:,4] = b - B
        output[:,5] = c - C
        return output

    def permute(self,data,index):
        '''
        :return: the ordered data initiated from the index
        '''
        Y = torch.cat((data[index:], data[:index]), axis=0)
        return Y

    def trans_theta(self,theta,positive=True):
        '''
        :param theta: input angle
        :param positive: True: [0,2pi], False:[-2pi,0]
        :return: the equivalent angle in [0,2pi] or [-2pi,0]
        '''
        theta = torch.angle(torch.cos(theta) + 1j * torch.sin(theta))  # mod the theta s.t. its in [-pi,pi]
        if positive:
            if theta >= 0:
                theta = theta
            else:
                theta = theta+2*math.pi
        else:
            if theta <= 0:
                theta = theta
            else:
                theta = theta - 2 * math.pi
        return theta

    def gamma(self, data):
        latent, phi, dphi, reverse = self.AE.forward(data)
        Gamma = []  # element: [\theta,\gamma(\theta)]
        interval = torch.linspace(-math.pi,math.pi,len(phi))
        for j in range(len(data)):
            theta = phi[j]-phi[0] # Z(phi_0)\cdot H(\phi_0,\phi_j), as a function of phi_j-phi_i
            gamma_theta = self.torch_trapz(interval,torch.sum(dphi*self.couple(reverse,self.permute(reverse,j)),dim=1))/(2*math.pi) # dphi = permute(dphi,0), initiated from the same start
            if -math.pi<=theta<=math.pi:
                Gamma.append([theta,gamma_theta])
            theta = -theta # Z(phi_j)\cdot H(\phi_0,\phi_j)
            gamma_theta = self.torch_trapz(interval,torch.sum(self.permute(dphi,j)*self.couple(self.permute(reverse,j),reverse),dim=1))/(2*math.pi) # dphi = permute(dphi,0), initiated from the same start
            if -math.pi <= theta <= math.pi:
                Gamma.append([theta,gamma_theta])
        Gamma = torch.tensor(Gamma)
        # print(f'gamma:{Gamma[0]}')
        # plt.plot(np.arange(len(dphi)),(dphi*reverse).detach()[:,-1])
        # # plt.plot(np.arange(len(dphi)), (reverse).detach()[:, -1])
        # plt.title(f'{torch.mean((dphi*reverse).detach()[:,-1])}')
        # plt.show()
        return Gamma

    def delta_w(self,data,delta_beta):
        latent, phi, dphi, reverse = self.AE.forward(data)
        interval = torch.linspace(-math.pi, math.pi, len(phi))
        delta_w = delta_beta*self.torch_trapz(interval,torch.sum(dphi * self.w_func(reverse), dim=1)) / (
                                      2 * math.pi)  # Z(\phi)\cdot\delta\beta (0,0,0,a-A,b-B,c-C,0)

        return delta_w

    def delta_w_ext(self, data):
        latent, phi, dphi, reverse = self.AE.forward(data)
        g = self.w_func(reverse).detach()
        r_phixx = torch.autograd.grad(torch.sum(dphi * g, dim=1).sum(), data, create_graph=True)[0]
        kernel = torch.sum(0.5*r_phixx*g,dim=1)
        interval = torch.linspace(-math.pi, math.pi, len(phi))
        delta_w = self.torch_trapz(interval, kernel) / (2 * math.pi)  # 1/2*Tr[G\topY(\phi)G]
        return delta_w

    def delta_w_phase(self,phi,delta_beta):
        '''
        :param phi: size (\cdot,1)
        :param delta_beta: size, (num,)
        :return: the delta_w depending on phi, the coupling function
        '''
        dphi = self.AE.phase_sensitive_func(phi) # size, (num,7)
        chi = self.AE.decoder(phi) # size (num,7)
        coupling = torch.mean(chi[:,-1])*dphi[:,-1] # size, (num,)
        delta_w = delta_beta*torch.sum(dphi * self.w_func(chi), dim=1)  # Z(\phi)\cdot\delta\beta (0,0,0,a-A,b-B,c-C,0)
        return delta_w,coupling

    def torch_trapz(self,x,y):
        x = x.detach().numpy()
        y = y.detach().numpy()
        output = integrate.trapz(y, x)
        return torch.from_numpy(np.array(output))

    def fourier_regression(self,data,degree=20):
        data = data.detach().numpy()
        x = data[:,0]
        y = data[:,1]
        a0 = integrate.trapz(y, x) / (2 * np.pi)
        # pred_y = np.zeros_like(y)
        # pred_y += a0
        coef = []
        coef.append(a0)
        for i in range(degree):
            n = i + 1
            an = integrate.trapz(y * np.cos(n * x), x) / (np.pi)
            bn = integrate.trapz(y * np.sin(n * x), x) / (np.pi)
            coef += [an, bn]
            # pred_y += an * np.cos(n * x) + bn * np.sin(n * x)
        coef = torch.tensor(coef)
        def func(theta,coef=coef,degree=degree):
            y = torch.zeros_like(theta)
            y += coef[0]
            for i in range(degree):
                n = i+1
                an,bn = coef[1+i*2:1+i*2+2]
                y += an*torch.cos(n*theta)+bn*torch.sin(n*theta)
            return y
        return coef,func

    def cubic_regression(self,data):
        '''
        :param data: the theta and its corresponding gamma value, the theta should be sorted increasingly
        :return: the cublicline function
        '''
        data = data.detach().numpy()
        cs = CubicSpline(data[:,0], data[:,1])
        def torch_cs(input,cs=cs):
            input = input.detach().numpy()
            output = cs(input)
            return torch.from_numpy(output)
        return torch_cs

    def phi2x(self,phi):
        '''
        :param phi: time trajectory of the phase dynamics
        :return: time trajectory on the limit cycle
        '''
        L,num = phi.shape
        target = torch.zeros([L,num*7])
        for i in range(num):
            latent = torch.cat((torch.cos(phi[:,i:i+1]),torch.sin(phi[:,i:i+1])),dim=1)
            target[:,i*7:i*7+7] = self.AE.dec(latent)
        return target


class ECC_dynamics(nn.Module):

    def __init__(self,Q,delta_beta,delta_w,w,Gamma,model):
        super(ECC_dynamics, self).__init__()
        self.alpha = 216.0
        self.n = 2.0
        self.k = 20.0
        self.delta_beta = delta_beta
        self.beta = 1.0+delta_beta # variability in the cell population
        self.eta = 2.0
        self.ks0 = 1
        self.ks1 = 0.01
        self.Q = Q # cell density
        self.delta_w = delta_w # difference of the natural frequency
        self.Gamma = Gamma # reduced PRC
        self.w = w # learned frequency
        self.model = model

    def forward(self, t, state):
        dstate = torch.zeros_like(state)
        L = state.shape[1] # numbers of variables
        a, b, c, A, B, C, S = state[:,0:L:7],state[:,1:L:7],state[:,2:L:7],state[:,3:L:7],state[:,4:L:7],state[:,5:L:7],state[:,6:L:7]
        dstate[:,0:L:7] = -a+self.alpha/(1+C**self.n)
        dstate[:,1:L:7] = -b+self.alpha/(1+A**self.n)
        dstate[:, 2:L:7] = -c+self.alpha/(1+B**self.n)+self.k*S/(1+S)
        dstate[:, 3:L:7] = self.beta*(a-A)
        dstate[:, 4:L:7] = self.beta*(b-B)
        dstate[:, 5:L:7] = self.beta*(c-C)
        dstate[:, 6:L:7] = -self.ks0*S+self.ks1*A-self.eta*(S-self.Q*S.mean(dim=1).view(-1,1))
        return dstate

    def phase_forward(self, t, phi):
        L = phi.shape[1] # numbers of variables
        delta_matrix = torch.zeros([L,L])
        for i in range(1,L):
            for j in range(i+1,L):
                delta_matrix[i,j] = phi[:,j]-phi[:,i]
        delta_matrix += -delta_matrix.T
        coupling = self.Gamma(delta_matrix).mean(dim=1).view(-1,L)
        dphi = self.w*torch.ones_like(phi)+self.delta_w+self.eta*self.Q*coupling
        # dphi = self.w*torch.ones_like(phi)+self.delta_w+self.eta*self.Q*torch.sin(delta_matrix).mean(dim=1).view(-1,L)
        # dphi = torch.ones_like(phi)+self.delta_w+self.eta*self.Q*coupling*5
        return dphi

    def phase_forward1(self, t, phi):
        delta_w,coupling = self.model.delta_w_phase(phi.view(-1,1),self.delta_beta)
        dphi = self.w*torch.ones_like(phi)+self.delta_w+self.eta*self.Q*coupling
        return dphi.detach()


class ECC_dynamics_ext(nn.Module):

    def __init__(self, Q, delta_w, w, Gamma, model, scale=None,autoencoder=None):
        super(ECC_dynamics_ext, self).__init__()
        self.alpha = 216.0
        self.n = 2.0
        self.k = 20.0
        # self.delta_beta = delta_beta
        self.beta = 1.0  # variability in the cell population
        self.eta = 2.0
        self.ks0 = 1
        self.ks1 = 0.01
        self.Q = Q  # cell density
        self.delta_w = delta_w  # difference of the natural frequency
        self.Gamma = Gamma  # reduced PRC
        self.w = w  # learned frequency
        self.model = model
        self.sigma = scale
        self.k = 15.0
        self.AE = autoencoder



    def forward(self, t, state):
        dstate = torch.zeros_like(state)
        L = state.shape[1]  # numbers of variables
        a, b, c, A, B, C, S = state[:, 0:L:7], state[:, 1:L:7], state[:, 2:L:7], state[:, 3:L:7], state[:,
                                                                                                  4:L:7], state[:,
                                                                                                          5:L:7], state[
                                                                                                                  :,
                                                                                                                  6:L:7]
        dstate[:, 0:L:7] = -a + self.alpha / (1 + C ** self.n)
        dstate[:, 1:L:7] = -b + self.alpha / (1 + A ** self.n)
        dstate[:, 2:L:7] = -c + self.alpha / (1 + B ** self.n) + self.k * S / (1 + S)
        dstate[:, 3:L:7] = self.beta * (a - A)
        dstate[:, 4:L:7] = self.beta * (b - B)
        dstate[:, 5:L:7] = self.beta * (c - C)
        dstate[:, 6:L:7] = -self.ks0 * S + self.ks1 * A - self.eta * (S - self.Q * S.mean(dim=1).view(-1, 1))
        return dstate

    def forward_OU(self, t, state):
        dstate = torch.zeros_like(state)
        L = state.shape[1]  # numbers of variables
        a, b, c, A, B, C, S = state[:, 0:L:7], state[:, 1:L:7], state[:, 2:L:7], state[:, 3:L:7], state[:,
                                                                                                  4:L:7], state[:,
                                                                                                          5:L:7], state[
                                                                                                                  :,
                                                                                                                  6:L:7]
        dstate[:, 0:L:7] = -a + self.alpha / (1 + C ** self.n)
        dstate[:, 1:L:7] = -b + self.alpha / (1 + A ** self.n)
        dstate[:, 2:L:7] = -c + self.alpha / (1 + B ** self.n) + self.k * S / (1 + S)
        LL = 3*int(L/7)
        w = torch.from_numpy(np.random.normal(0,1,LL))
        dstate[:, 3:L:7] = self.beta * (a - A) + self.sigma*math.sqrt((1-math.exp(-2*self.k*t))/(2*self.k))*w[0:LL:3]
        dstate[:, 4:L:7] = self.beta * (b - B) + self.sigma*math.sqrt((1-math.exp(-2*self.k*t))/(2*self.k))*w[1:LL:3]
        dstate[:, 5:L:7] = self.beta * (c - C) + self.sigma*math.sqrt((1-math.exp(-2*self.k*t))/(2*self.k))*w[2:LL:3]
        dstate[:, 6:L:7] = -self.ks0 * S + self.ks1 * A - self.eta * (S - self.Q * S.mean(dim=1).view(-1, 1))
        return dstate

    def forward_perturbed(self, noise, state):
        dstate = torch.zeros_like(state)
        L = state.shape[1]  # numbers of variables
        a, b, c, A, B, C, S = state[:, 0:L:7], state[:, 1:L:7], state[:, 2:L:7], state[:, 3:L:7], state[:,
                                                                                                  4:L:7], state[:,
                                                                                                          5:L:7], state[
                                                                                                                  :,
                                                                                                                  6:L:7]
        LL = noise.shape[1]
        dstate[:, 3:L:7] = (a - A)*noise[:,0:LL:3]
        dstate[:, 4:L:7] = (b - B)*noise[:,1:LL:3]
        dstate[:, 5:L:7] = (c - C)*noise[:,2:LL:3]

        return dstate

    def phase_forward(self, t, phi):
        L = phi.shape[1]  # numbers of variables
        delta_matrix = torch.zeros([L, L])
        for i in range(1, L):
            for j in range(i + 1, L):
                delta_matrix[i, j] = phi[:, j] - phi[:, i]
        delta_matrix += -delta_matrix.T
        coupling = self.Gamma(delta_matrix).mean(dim=1).view(-1, L)
        dphi = self.w * torch.ones_like(phi) + self.delta_w + self.eta * self.Q * coupling
        # dphi = self.w*torch.ones_like(phi)+self.delta_w+self.eta*self.Q*torch.sin(delta_matrix).mean(dim=1).view(-1,L)
        # dphi = torch.ones_like(phi)+self.delta_w+self.eta*self.Q*coupling*5
        return dphi

    def phase_forward_OU(self, t, phi):
        L = phi.shape[1]  # numbers of variables
        delta_matrix = torch.zeros([L, L])
        for i in range(1, L):
            for j in range(i + 1, L):
                delta_matrix[i, j] = phi[:, j] - phi[:, i]
        delta_matrix += -delta_matrix.T
        coupling = self.Gamma(delta_matrix).mean(dim=1).view(-1, L)
        LL = L*3
        # reverse, dphi = self.AE.dphi_reverse(phi.view(-1, 1))
        # w =  self.sigma * math.sqrt((1 - math.exp(-2 * self.k * t)) / (2 * self.k)) *torch.from_numpy(np.random.normal(0, 1, LL))
        w = torch.from_numpy(np.random.normal(0, 1, LL))
        dphi = self.w * torch.ones_like(phi) + self.delta_w + self.eta * self.Q * coupling + w[0:LL:3]*self.sigma#+ self.phase_perturbed(w.view(-1,3),reverse.detach(),dphi.detach())
        return dphi

    def phase_perturbed(self, noise, state, Z_phi):
        dstate = torch.zeros_like(state)
        L = state.shape[1]  # numbers of variables
        a, b, c, A, B, C, S = state[:, 0:L:7], state[:, 1:L:7], state[:, 2:L:7], state[:, 3:L:7], state[:,
                                                                                                  4:L:7], state[:,
                                                                                                          5:L:7], state[
                                                                                                                  :,
                                                                                                                  6:L:7]
        LL = noise.shape[1]
        dstate[:, 3:L:7] = (a - A)*noise[:,0:LL:3]
        dstate[:, 4:L:7] = (b - B)*noise[:,1:LL:3]
        dstate[:, 5:L:7] = (c - C)*noise[:,2:LL:3]
        dphi = torch.sum(dstate*Z_phi,dim=1)
        return dphi.view(1,-1)

    def phase_forward1(self, t, phi):
        delta_w, coupling = self.model.delta_w_phase(phi.view(-1, 1), self.delta_beta)
        dphi = self.w * torch.ones_like(phi) + self.delta_w + self.eta * self.Q * coupling
        return dphi.detach()



def OU():
    x0 = 0.0
    n = 5000
    X = np.zeros(n)
    X[0] = x0
    k = 15
    dt = 0.01
    sigma = 0.1
    mu = 0.0
    w = np.random.normal(0,1,n)
    Y = np.zeros(n)
    Y[0] = x0
    for i in range(n-1):
        x = X[i]
        new_x = x + dt*(k*(mu-x)) + sigma*np.sqrt(dt)*w[i]
        X[i+1] = new_x
        t = i*dt
        Y[i+1] = sigma*np.sqrt((1-np.exp(-2*k*t))/(2*k))*w[i]
    plt.subplot(121)
    plt.plot(np.linspace(0,n*dt,n),X)
    plt.ylim(-2,2)
    plt.subplot(122)
    plt.plot(np.linspace(0,n*dt,n),Y)
    plt.ylim(-2,2)
    plt.show()

# OU()

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

def ECC_(x):
    alpha = 216.0
    n = 2.0
    k = 20.0
    beta = 1.0
    eta = 2.0
    ks0,ks1 = 1,0.01
    y=torch.zeros_like(x)
    for i in range(len(x)):
        a,b,c,A,B,C,S=x[i,:]
        da,db,dc=-a+alpha/(1+C**n),-b+alpha/(1+A**n),-c+alpha/(1+B**n)+k*S/(1+S)
        dA,dB,dC=beta*(a-A),beta*(b-B),beta*(c-C)
        dS=-ks0*S+ks1*A-eta*(S-0)
        y[i,:]=torch.tensor([da,db,dc,dA,dB,dC,dS])
    return y

def ECC_varyQ(x,Q):
    alpha = 216.0
    n = 2.0
    k = 20.0
    beta = 1.0
    eta = 2.0
    ks0,ks1 = 1,0.01
    y=torch.zeros_like(x)
    for i in range(len(x)):
        a,b,c,A,B,C,S=x[i,:]
        da,db,dc=-a+alpha/(1+C**n),-b+alpha/(1+A**n),-c+alpha/(1+B**n)+k*S/(1+S)
        dA,dB,dC=beta*(a-A),beta*(b-B),beta*(c-C)
        dS=-ks0*S+ks1*A-eta*(1.0-Q)*S
        y[i,:]=torch.tensor([da,db,dc,dA,dB,dC,dS])
    return y

def FHN_(x):
    epi = 0.05
    a = 0.7
    b = 0.2
    y=torch.zeros_like(x)
    for i in range(len(x)):
        s1,s2 = x[i,:]
        y[i,:]=torch.tensor([(s1-s1**3/3-s2)/epi,s1+a-b*s2])
    return y

def kepler_(x):
    m = 1.0
    g = 9.8
    y=torch.zeros_like(x)
    for i in range(len(x)):
        s1,s2,a,b = x[i,:]
        dist = (s1 ** 2 + s2 ** 2) ** 1.5
        y[i,:]=torch.tensor([a*m,b*m,-g*m**2*s1/dist,-g*m**2*s2/dist])
    return y

class FHN(nn.Module):

    def __init__(self):
        super(FHN, self).__init__()
        self.epi = 0.05
        self.a = 0.7
        self.b = 0.2
        self.I = 0.

    def forward(self, t, x):
        dx = torch.zeros_like(x)
        # L = 3*self.A.shape[0] # 拼接向量x的长度
        L = 1
        x,y = x[:,0:2*L:2],x[:,1:2*L:2]
        '''
        FitzHugh-Nagumo向量场
        '''
        dx[:,0:2*L:2] = (x-x**3/3-y+self.I)/self.epi
        dx[:,1:2*L:2] = x+self.a-self.b*y

        return dx

class FHN_couple1(nn.Module):

    def __init__(self,A,sigma):
        super(FHN_couple1, self).__init__()
        self.epi = 0.05
        self.a = 0.7
        self.b = 0.2
        self.I = 0.
        self.A = A - torch.diag(torch.sum(A,dim=1)) # laplace 矩阵
        self.sigma = sigma # strength

    def forward(self, t, x):
        dx = torch.zeros_like(x)
        L = 2*self.A.shape[0] # 拼接向量x的长度
        x,y = x[:,0:2*L:2],x[:,1:2*L:2]
        '''
        FitzHugh-Nagumo向量场
        '''
        dx[:,0:2*L:2] = (x-x**3/3-y+self.I)/self.epi
        dx[:,1:2*L:2] = x+self.a-self.b*y
        '''
        接来来计算耦合部分
        '''
        h_x = torch.zeros_like(x)
        h_x[:,0:2*L:2] = self.sigma*x
        dx += torch.mm(torch.kron(self.A,torch.eye(2)),h_x.T).T
        return dx


def train_data():
    dim = 1
    true_y0 = torch.randn([1, dim * 2])  # 初值
    # true_y0 = torch.tensor([[0.1,0.2,2.]])
    t = torch.linspace(0., 10., 6000)  # 时间点
    func = FHN()
    with torch.no_grad():
        true_y = odeint(func, true_y0, t, method='dopri5')[-3070:,0,:]

    np.save('./data/train_FHN.npy',true_y)
    plt.subplot(121)
    plt.plot(np.arange(len(true_y)),true_y[:,0])
    plt.subplot(122)
    Y = np.load('./data/train_FN_long.npy')
    plt.scatter(true_y[:,0],true_y[:,1])
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.show()

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