import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange
from functions import *

def pnas(input):
    alpha = 216.0
    n = 2.0
    k = 20.0
    beta = 1.0
    eta = 2.0
    ks0,ks1 = 1,0.01
    a,b,c,A,B,C,S = input
    da,db,dc = -a+alpha/(1+C**n),-b+alpha/(1+A**n),-c+alpha/(1+B**n)+k*S/(1+S)
    dA,dB,dC = beta*(a-A),beta*(b-B),beta*(c-C)
    dS = -ks0*S+ks1*A-eta*(S-0)
    return np.array([da,db,dc,dA,dB,dC,dS])


class ECC(nn.Module):

    def __init__(self,Q,delta_beta):
        super(ECC, self).__init__()
        self.alpha = 216.0
        self.n = 2.0
        self.k = 20.0
        self.beta = 1.0+delta_beta # variability in the cell population
        self.eta = 2.0
        self.ks0 = 1
        self.ks1 = 0.01
        self.Q = Q # cell density
        self.scale = 1.0

    def forward(self, t, state):
        dstate = torch.zeros_like(state)
        L = state.shape[1] # numbers of variables
        a, b, c, A, B, C, S = state[:,0:L:7],state[:,1:L:7],state[:,2:L:7],state[:,3:L:7],state[:,4:L:7],state[:,5:L:7],state[:,6:L:7]
        dstate[:,0:L:7] = -a+self.alpha/(1+C**self.n)
        dstate[:,1:L:7] = -b+self.alpha/(1+A**self.n)
        dstate[:, 2:L:7] = -c+self.alpha/(1+B**self.n)+self.k*S/(self.scale+S)
        dstate[:, 3:L:7] = self.beta*(a-A)
        dstate[:, 4:L:7] = self.beta*(b-B)
        dstate[:, 5:L:7] = self.beta*(c-C)
        dstate[:, 6:L:7] = self.scale*(-self.ks0*S+self.ks1*A-self.eta*(S-self.Q*S.mean()))
        return dstate

    def uncouple_forward(self, t, state):
        dstate = torch.zeros_like(state)
        L = state.shape[1] # numbers of variables
        a, b, c, A, B, C, S = state[:,0:L:7],state[:,1:L:7],state[:,2:L:7],state[:,3:L:7],state[:,4:L:7],state[:,5:L:7],state[:,6:L:7]
        dstate[:,0:L:7] = -a+self.alpha/(1+C**self.n)
        dstate[:,1:L:7] = -b+self.alpha/(1+A**self.n)
        dstate[:, 2:L:7] = -c+self.alpha/(1+B**self.n)+self.k*S/(self.scale+S)
        dstate[:, 3:L:7] = self.beta*(a-A)
        dstate[:, 4:L:7] = self.beta*(b-B)
        dstate[:, 5:L:7] = self.beta*(c-C)
        dstate[:, 6:L:7] = self.scale*(-self.ks0*S+self.ks1*A-self.eta*(1-self.Q)*S)
        return dstate

def generate():
    dim = 7
    num = 10
    setup_seed(1)
    x0 = torch.rand([1, dim * num])
    delta_beta = torch.from_numpy(np.random.normal(0, 0.0, [num]))
    t = torch.linspace(0, 200, 10000)
    info_list = [] # save the information of Q and corresponding period
    for k in tqdm(range(21),desc='Generate the limit cycles'):
        Q = float(format(k * 0.05, '.2f'))
        func = ECC(Q, delta_beta)
        with torch.no_grad():
            orig_X = odeint(func.uncouple_forward, x0, t)[:, 0, :]
        result_args = np.argpartition(orig_X[-2000:, 0], -10)[-10:]
        index_list = []
        for i in range(10):
            for j in range(i, 10):
                res = np.abs(result_args[i] - result_args[j])
                if res > 600 and res < 800:
                    index_list.append(res)
        print(int(np.mean(index_list)),Q)
        L = int(np.mean(index_list))
        info_list.append([Q,L])
        torch.save(orig_X[-L:,0:7],'./data/orbit/train_ECC_Q_{}.pt'.format(Q))
    np.save('./data/orbit/orbit_information',np.array(index_list))

# generate()

# data = torch.load('./data/orbit/train_ECC_Q_0.0.pt')
# print(data.shape)

def plot():
    dt = 0.02
    dim = 7
    num = 10
    setup_seed(1)

    # data = torch.load('./data/train_ECC.pt')[80:]
    # print(data.shape)
    # x0 = data[0:1, :].repeat(1, num)

    x0 = torch.rand([1, dim * num])
    delta_beta = torch.from_numpy(np.random.normal(0, 0.05, [num]))
    Q = 0.9
    func = ECC(Q, delta_beta)
    t = torch.linspace(0, 200, 10000)
    with torch.no_grad():
        X = odeint(func, x0, t)[:, 0, :]
        orig_X = odeint(func.uncouple_forward, x0, t)[:, 0, :]
    S = X[:, 6:num * dim:dim]
    m_S = S.mean(dim=1)

    result_args = np.argpartition(orig_X[-2000:, 0], -10)[-10:]
    index_list = []
    for i in range(10):
        for j in range(i, 10):
            res = np.abs(result_args[i] - result_args[j])
            if res > 600 and res < 800:
                index_list.append(res)
    print(result_args, '\n', int(np.mean(index_list)))
    L = 720
    # L = int(np.mean(index_list))
    # torch.save(X[-L:],'./data/train_ECC_ks1_0.1.pt')

    fig = plt.figure()
    # X = np.load('./data/train_pnas.npy')
    # ax = fig.gca(projection='3d')
    # ax.plot(X[-L:,3],X[-L:,4],X[-L:,6])
    # plt.plot(X[-L:,0],X[-L:,1])
    plt.subplot(221)
    for i in range(num):
        # plt.plot(np.arange(len(X)),X[:,1+i*7]-X[:,1+3+i*7])
        plt.plot(np.arange(len(X)), X[:, 6 + i * 7])
        # plt.plot(X[-L:,1+i*7],X[-L:,2+i*7])
    plt.subplot(222)
    plt.plot(np.arange(len(X)), orig_X[:, 0])
    plt.subplot(223)
    for i in range(num):
        plt.plot(X[-L:,0+i*7],X[-L:,1+i*7])
    plt.title('M={:.2f}'.format(cal_M(X[-1000:])))
    plt.subplot(224)
    for i in range(1):
        plt.plot(orig_X[-L:, 0 + i * 7], orig_X[-L:, 1 + i * 7])
        # plt.plot(data[:,1],data[:,2])

    plt.show()

plot()
# for k in range(10):
#     Q = float(format(k * 0.1, '.2f'))
#     data = torch.load('./data/orbit/train_ECC_Q_{}.pt'.format(Q))
#     plt.plot(data[:,0],data[:,1],label='Q={}'.format(Q))
#     plt.legend()
# plt.show()