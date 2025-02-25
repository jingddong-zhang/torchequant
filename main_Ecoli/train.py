import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import timeit
import torch.nn.functional as F
import torch.nn as nn
torch.set_default_dtype(torch.float64)
#
# def snake(x):
#     return x+torch.sin(x)**2
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
            # Snake(),
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

    def time_derivative(self,data):
        F=self.forward(data)
        dF=torch.autograd.grad(F.sum(),data,create_graph=True)[0]
        w = self.w
        return dF,w

class AE(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(AE, self).__init__()
        self.enc = NN(n_input, n_hidden, n_output)
        self.dec = NN(n_output, n_hidden, n_input)
        self.w = torch.tensor([1.5],requires_grad=True)

    def decoder(self, data):
        phi = self.enc(data)
        reverse_data = self.dec(phi)
        return reverse_data

    def encoder(self,data):
        phi = self.enc(data)
        dphi = torch.autograd.grad(phi.sum(),data,create_graph=True)[0]
        return phi,dphi

def f_(x):
    epi=0.05
    a=0.7
    b=0.2
    y=torch.zeros_like(x)
    for i in range(len(x)):
        s1,s2 = x[i,:]
        y[i,:]=torch.tensor([(s1-s1**3/3-s2)/epi,s1+a-b*s2])
    return y

def get_batch(Data):
    length = len(Data)
    s = torch.from_numpy(np.random.choice(np.arange(length, dtype=np.int64),300, replace=False))
    batch_y = Data[s,:].requires_grad_(True)  # (M, D)
    return batch_y


# X=np.load('./data/train_FN.npy')
X=np.load('./data/train_FN_long.npy')[0:3030]

def plot0(X):
    X = X[0:-1:10]
    N = len(X)
    st_c,ed_c = np.array([0,1,0]),np.array([0,0,1])
    for i in range(N):
        c = st_c + i/(N-1)*(ed_c-st_c)
        plt.scatter(X[i,0],X[i,1],lw=0.1,color=c)
    plt.show()
# plot0(X)
# plt.plot(X[:,0],X[:,1])
# plt.show()
print(X.shape)
data = torch.from_numpy(X)


'''
For learning 
'''
D_in = 2  # input dimension
H1 = 20 * D_in  # hidden dimension
D_out = 1 # output dimension

out_iters = 0
while out_iters < 1:
    # break
    start = timeit.default_timer()
    model = AE(D_in, H1, D_out)
    i = 0
    max_iters = 5000
    learning_rate = 0.01
    optimizer = torch.optim.Adam([i for i in model.enc.parameters()]+[model.w], lr=learning_rate)
    data = data.requires_grad_(True)
    enc_data,dec_data = data[0:-1:10],data
    f=f_(enc_data)
    while i < max_iters:
        # break
        phi,dphi = model.encoder(enc_data)
        ptp = torch.max(phi) - torch.min(phi)
        inner = torch.sum(dphi*f,dim=1)
        loss1 = torch.mean((inner-model.w)**2)+torch.relu(1.-model.w) # surrogate loss
        loss = torch.mean((inner-model.w)**2)+(math.pi * 2 - ptp) ** 2
        print(i, "loss=", loss.item(),model.w,ptp.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss<=1e-4:
            break
        i += 1
    # torch.save(model.enc.state_dict(), './data/FHN_enc.pkl')
    i = 0
    optimizer = torch.optim.Adam([i for i in model.dec.parameters()],lr=0.03)
    while i < max_iters:
        # break
        reverse = model.decoder(dec_data)
        loss = torch.mean((reverse-dec_data)**2)
        print(i, "loss=", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss<=1e-4:
            break
        i += 1
    # torch.save(model.dec.state_dict(), './data/FHN_dec.pkl')

    stop = timeit.default_timer()
    '''
    check auto-encoder
    '''
    data = torch.from_numpy(np.load('./data/train_FN_long.npy')).requires_grad_(True)[0:3070]
    # phi, dphi, reverse = model.forward(data)
    phi,dphi = model.encoder(data)
    reverse = model.decoder(data)
    reverse = reverse.detach().numpy()
    data = data.detach().numpy()
    plt.subplot(121)
    plt.scatter(np.arange(len(phi)),phi.detach().numpy())

    plt.subplot(122)
    plt.plot(reverse[:,0],reverse[:,1])
    # plt.plot(data[:, 0], data[:, 1],color='r')
    plt.show()
    print('\n')
    print("Total time: ", stop - start)

    out_iters += 1


'''
generate train data for regression
'''



def gamma(j,i=0):
    model=Net(D_in,H1,D_out)
    model.load_state_dict(torch.load('./data/theta_FHN.pkl'))
    X = np.load('./data/train_FN.npy')[20:]
    input = torch.from_numpy(X).requires_grad_(True)
    theta = model.forward(input).detach().numpy()
    dtheta,w = model.time_derivative(input)
    dtheta = dtheta.detach().numpy()
    scale = math.pi*2/(np.max(theta)-np.min(theta))
    def H(x,y):
        tau = 0.1
        out = np.zeros_like(x)
        out[:,0]=1/(1+np.exp(-y[:,0]/tau))-1/(1+np.exp(-x[:,0]/tau))
        return out
    def permute(X,i):
        Y = np.concatenate((X[i:],X[:i]),axis=0)
        return Y
    value = np.mean(np.sum(permute(dtheta,i)*H(permute(X,i),permute(X,j)),axis=1))
    return (theta[j]-theta[i])*scale,value


def generate_FHN():
    n=1490
    theta_=np.zeros(n*2)
    gamma_=np.zeros(n*2)
    for j in range(n):
        # theta_[j],gamma_[j]=gamma(j) # 正向
        theta_[j+n],gamma_[j+n] = gamma(1490+j,1490)
        theta_[j],gamma_[j] = gamma(1490-(n-j),1490)
        # theta_[-j-1],gamma_[-j-1]=gamma(-j-1,-1) # 倒向
        if j%100==0:
            print(j)
    np.save('./data/data_regress_FHN',np.concatenate((theta_.reshape(1,-1).T,gamma_.reshape(1,-1).T),axis=1))
    plt.plot(np.arange(len(gamma_)),gamma_)
    plt.xticks([0,1490,1490*2],[r'$-\pi$',0,r'$\pi$'])
    print(theta_[0],theta_[-1])
    plt.show()