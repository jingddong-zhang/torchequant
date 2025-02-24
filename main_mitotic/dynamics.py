import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange
from functions import *

class Mitotic(nn.Module):

    def __init__(self):
        super(Mitotic, self).__init__()
        self.v_i = 0.023
        self.v_d = 0.1
        self.K_d = 0.02
        self.k_d = 3.3*1e-3
        self.V_M1 = 0.5
        self.V_2 = 0.167
        self.V_M3 = 0.2
        self.V_4 = 0.1
        self.K_C = 0.3
        self.K_1 = 0.1
        self.K_2 = 0.1
        self.K_3 = 0.1
        self.K_4 = 0.1
        self.scale = 5.0

    def forward(self, t, state):
        dstate = torch.zeros_like(state)
        L = state.shape[1] # numbers of variables
        dim = 3
        C = (state[:,0:L:dim]/self.scale+0.125)
        M = (state[:,1:L:dim]/self.scale+0.4)
        X = (state[:,2:L:dim]/self.scale+0.3)
        dstate[:,0:L:dim] = self.v_i-self.v_d*X*C/(self.K_d+C)-self.k_d*C
        dstate[:,1:L:dim] = C/(self.K_C+C)*self.V_M1*(1.0-M)/(self.K_1+1.0-M)-self.V_2*M/(self.K_2+M)
        dstate[:,2:L:dim] = self.V_M3*M*(1.0-X)/(self.K_3+1.0-X)-self.V_4*X/(self.K_4+X)
        return dstate*self.scale


def plot():
    dim = 3
    num = 1
    # setup_seed(0)



    x0 = torch.tensor([[0.2,0.3,0.01]])
    func = Mitotic()
    t = torch.linspace(0, 300, 10000)
    with torch.no_grad():
        X = odeint(func, x0, t)[:, 0, :]

    result_args = np.argpartition(X[-2000:, 0], -10)[-10:]
    index_list = []
    for i in range(10):
        for j in range(i, 10):
            res = np.abs(result_args[i] - result_args[j])
            if res > 300 and res < 1500:
                index_list.append(res)
    print(result_args, '\n', int(np.mean(index_list)))
    L = int(np.mean(index_list))
    # L = 720
    vector = func(0.0,X[-L:,:])
    torch.save(X[-L:],'./data/train_mitotic.pt')
    torch.save(vector, './data/train_mitotic_func.pt')
    print(vector.shape,X[-L:].shape)
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    for i in range(dim):
        # plt.plot(np.arange(len(X)), X[:, i],label=f'{i}')
        plt.plot(X[-L:,0],X[-L:,1])
    plt.legend()


    plt.show()

plot()
