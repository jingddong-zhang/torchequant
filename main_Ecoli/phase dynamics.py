import numpy as np

from functions import *


model = AE(2, 20*2, 2)
model.load_state_dict(torch.load('./data/FHN.pkl'))
X = torch.from_numpy(np.load('./data/train_FN_long.npy')).requires_grad_(True)
cir,Phi,phi_data = model.encoder(X)
# Phi = torch.linspace(-pi,pi,2000) # generate phi

delta_ = []
gamma_ = []
N = len(Phi)
for i in range(N):
    phi1 = permute(Phi,0)
    phi2 = permute(Phi, i)
    gamma,delta_phi = step1(phi1,phi2,model,H1)
    delta = delta_phi[0]
    gamma = torch.mean(gamma)
    gamma_.append(gamma.detach().numpy())
    delta_.append(delta.detach().numpy())
    print(delta.item(),gamma.item(),i)

np.save('./data/phase_FHN_H1',{'phase':delta_,'gamma':gamma_})
plt.scatter(delta_,gamma_)
plt.axhline(0,ls='--')
plt.axvline(0,ls='--')
plt.show()