import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch

from functions import *

'''
hyperparameter
'''
D_in = 3 # input dimension
H1 = 32  # hidden dimension
D_latent = 2 # latent dimension
edim = 3 # dimension of equant

'''
models 
'''

auto_encoder = AE(D_in, H1, D_latent,w=0.33,edim=edim)
auto_encoder.load_state_dict(torch.load('./data/model_mitotic_equant.pkl'))


'''
loading data
'''
data = torch.load('./data/train_mitotic.pt')
data.requires_grad_(True)

f = torch.load('./data/train_mitotic_func.pt')
latent,phi,dphi,reverse = auto_encoder.forward(data)
ptp = torch.max(phi) - torch.min(phi)
inner = torch.sum(dphi*f,dim=1)
loss = torch.mean((reverse - data) ** 2) + torch.mean((inner - auto_encoder.w) ** 2)
                #+ (math.pi * 2 - ptp) ** 2 + torch.relu(1.0 - model.w ** 2)
print( "loss=", loss.item(),auto_encoder.w.item(),ptp.item())



'''
visualizing
'''


def transform(data):
    new_data = torch.zeros_like(data)
    new_data[:,0:1] = data[:, 0:1]/5.0+0.125
    new_data[:,1:2] = data[:, 1:2]/5.0+0.4
    new_data[:,2:3] = data[:, 2:3]/5.0+0.3
    return new_data

def calculate(data):
    '''
    :param data: (num_bd,k,dim) k is the sampled numbers on each line of sight
    :return: distance of the data to the line of sight
    '''
    distance = torch.zeros([data.shape[0],data.shape[1]-2]) # -2: remove the endpoints
    for i in range(len(data)):
        for j in range(data.shape[1]-2):
            vec_1 = data[i,j+1] - data[i,0]
            vec_2 = data[i,-1] - data[i,0]
            vec_2 = vec_2/torch.linalg.norm(vec_2,ord=2)
            distance[i,j] = torch.sqrt(torch.linalg.norm(vec_1,ord=2)**2-torch.sum(vec_1*vec_2)**2)
    return distance




def plot_equant(data):
    num_samples = 6 # number of samples on each line of sight
    aug_data = data # without augmentation, shape (T,dim)
    edim = aug_data.shape[1]
    latent, phi, dphi, reverse = auto_encoder.forward(data)
    equant = auto_encoder.surface.inverse(torch.zeros([1,edim]))[0]
    latent_bd, _, _ = auto_encoder.surface(aug_data)  # odeint(surface, data, t)[-1]
    pred_y_bd = auto_encoder.surface.inverse(latent_bd)  # odeint(surface.reverse, latent_ode, t)[-1]
    line_reference_data = aug_data - equant
    data_angle = angle_3d(line_reference_data)
    loss_angle = (data_angle - data_angle.mean()).abs()
    print(f'loss angle={loss_angle}')

    inner = torch.sum(dphi * f, dim=1)
    loss_phase = (inner - auto_encoder.w).abs()
    print(f'loss angle={loss_angle}')

    print(f'equant = {equant}')
    aug_data = aug_data.detach()
    latent_bd = latent_bd.detach()
    disk = generate_interior_data(latent_bd[0:-1:10,:2], torch.tensor([0.0, 0.0]), num_samples)
    aug_disk = augment(disk,edim) # the same space with aug_data
    aug_surface = auto_encoder.surface.inverse(aug_disk) # inverse of the disk D
    dist = calculate(aug_surface.view(-1,num_samples+1,edim))
    print(torch.mean(dist))
    aug_surface = aug_surface.detach().numpy()
    orig_surface = generate_interior_data(aug_data[0:-1:10], equant, num_samples)
    latent_surface, _, _ = auto_encoder.surface(orig_surface)
    dist = calculate(latent_surface.view(-1,num_samples+1,edim))
    print(torch.mean(dist),torch.mean((latent_surface - aug_disk) ** 2))
    latent_surface = latent_surface.detach().numpy()
    orig_surface = orig_surface.detach()

    latent_bd = latent_bd.detach().numpy()
    pred_y_bd = pred_y_bd.detach().numpy()
    reverse = reverse.detach().numpy()
    latent = latent.detach().numpy()

    np.save(f'./data/Mitotic_equant_{num_samples}',{'equant':equant.detach().numpy(),'M':(orig_surface),'C':(aug_data),'L_M':aug_surface,
                             'D':disk,'S^1':latent,'L_S^1':latent_bd,'w':auto_encoder.w.detach().numpy(),'f':(10000-1)/(len(data)*300),
                                                  'phase':phi.detach().numpy(),'L_D':latent_surface,'loss_angle':loss_angle.detach().numpy(),
                                                     'loss_phase':loss_phase.detach().numpy(),'data_angle':data_angle.detach().numpy()})
    import matplotlib
    rc_fonts = {
        "text.usetex": True,
        'text.latex.preview': True,  # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}""", r"""\usepackage{amsmath}""", r"""\usepackage{amsfonts}"""],
        'font.sans-serif': 'Times New Roman'
    }
    matplotlib.rcParams.update(rc_fonts)
    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'
    fontsize = 20
    ticksize = 18
    fig = plt.figure(figsize=(9,3))
    plt.subplots_adjust(left=0.05, bottom=0.17, right=0.95, top=0.87, hspace=0.25, wspace=0.3)

    ax = fig.add_subplot(131,projection='3d')
    ax.grid(None)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.axis('off')
    equant = equant.detach().numpy()
    plt.plot(aug_data[:, 0], aug_data[:, 1],aug_data[:,2],c=colors[-3])
    ax.scatter(equant[0], equant[1],equant[2], s=80, marker='o', c=colors[-3], label='Equant')
    ax.scatter(orig_surface[:, 0], orig_surface[:, 1], orig_surface[:, 2], c=colors[-3], alpha=0.5)
    ax.set_xlabel(r'$a$', fontsize=ticksize,labelpad=-15)
    ax.set_ylabel(r'$b$', fontsize=ticksize,labelpad=-15)
    ax.set_zlabel(r'$c$', fontsize=ticksize,labelpad=-15)
    # plt.legend(fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.title(r'$\text{Original}~\mathbb{M}$', fontsize=fontsize)

    ax = fig.add_subplot(132,projection='3d')
    ax.grid(None)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    aug_data = aug_data.detach().numpy()
    # ax.scatter(equant[0],equant[1],equant[2],s=30,marker='o',c='red')
    # ax.scatter(aug_data[:,0], aug_data[:,1], aug_data[:,2], s=10, marker='o',c='orange')
    plt.plot(aug_data[:, 0], aug_data[:, 1], aug_data[:, 2],c=colors[-3],label=r'$\bm C$')
    ax.scatter(equant[0], equant[1],equant[2], s=80, marker='o', c=colors[-3], label='Equant')
    ax.scatter(aug_surface[:,0],aug_surface[:,1],aug_surface[:,2],c=colors[-3],alpha=0.5)
    ax.set_xlabel(r'$a$', fontsize=ticksize,labelpad=-15)
    ax.set_ylabel(r'$b$', fontsize=ticksize,labelpad=-15)
    ax.set_zlabel(r'$c$', fontsize=ticksize,labelpad=-15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.title(r'$\text{Reconstruct}~\mathbb{M}$', fontsize=fontsize)
    plt.legend(fontsize=ticksize,ncol=2,handlelength=1.0,columnspacing=0.5,handletextpad=0.5,frameon=False)

    plt.subplot(133)
    plt.plot(latent[:, 0], latent[:, 1],label=r'$S^1$',c='k')
    plt.plot(latent_bd[:, 0], latent_bd[:, 1], label=r'$g_{\theta}(C)$',c=colors[-1])
    plt.scatter(0, 0, s=80, marker='o', c=colors[-1], label='Origin')
    # plt.scatter(disk[:,0],disk[:,1],c=colors[-1],alpha=0.5)
    plt.scatter(latent_surface[:, 0], latent_surface[:, 1], c=colors[-1], alpha=0.5)
    plt.xticks([-1,1],fontsize=ticksize)
    plt.yticks([-1, 1], fontsize=ticksize)
    plt.legend(loc=3,bbox_to_anchor=[-1.1,-0.35],fontsize=ticksize,ncol=3,handlelength=1.0,columnspacing=0.5,handletextpad=0.5,frameon=False)
    plt.title(r'$\text{Latent disk}~\mathbb{D}$', fontsize=fontsize)

    plt.show()
# plot_equant(data)


def Mitotic(state):
    v_i = 0.023
    v_d = 0.1
    K_d = 0.02
    k_d = 3.3*1e-3
    V_M1 = 0.5
    V_2 = 0.167
    V_M3 = 0.2
    V_4 = 0.1
    K_C = 0.3
    K_1 = 0.1
    K_2 = 0.1
    K_3 = 0.1
    K_4 = 0.1
    scale = 5.0
    dstate = torch.zeros_like(state)
    L = state.shape[1] # numbers of variables
    dim = 3
    C = (state[:,0:L:dim]/scale+0.125)
    M = (state[:,1:L:dim]/scale+0.4)
    X = (state[:,2:L:dim]/scale+0.3)
    dstate[:,0:L:dim] = v_i-v_d*X*C/(K_d+C)-k_d*C
    dstate[:,1:L:dim] = C/(K_C+C)*V_M1*(1.0-M)/(K_1+1.0-M)-V_2*M/(K_2+M)
    dstate[:,2:L:dim] = V_M3*M*(1.0-X)/(K_3+1.0-X)-V_4*X/(K_4+X)
    return dstate*scale

def uniformity():
    aug_data = data  # without augmentation, shape (T,dim)
    edim = aug_data.shape[1]
    latent, phi, dphi, reverse = auto_encoder.forward(data)
    equant = auto_encoder.surface.inverse(torch.zeros([1, edim]))[0]
    latent_bd, _, _ = auto_encoder.surface(aug_data)  # odeint(surface, data, t)[-1]
    pred_y_bd = auto_encoder.surface.inverse(latent_bd)  # odeint(surface.reverse, latent_ode, t)[-1]
    latent_bd = latent_bd.detach()

    fontsize = 20
    ticksize = 18
    fig = plt.figure(figsize=(6, 6))

    plt.subplot(121)
    r = np.linspace(0.8, 1.2, 3)
    phi = np.linspace(-np.pi, np.pi, 20)
    X, Y = np.kron(r.reshape(1, -1), np.cos(phi.reshape(1, -1)))[0], \
           np.kron(r.reshape(1, -1), np.sin(phi.reshape(1, -1)))[0]
    input = torch.from_numpy(np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1))
    output = auto_encoder.dec(input)
    _,_,dphi,_ = auto_encoder.forward(output)
    F_  = Mitotic(output)
    speed = torch.sum(dphi*F_,dim=1).detach()
    print(speed.shape)
    U = -speed * Y #X - w * Y - X * (X ** 2 + Y ** 2)
    V = speed * X #w * X + Y - Y * (X ** 2 + Y ** 2)
    dinput = np.concatenate((U.reshape(-1, 1), V.reshape(-1, 1)), axis=1)
    plt.plot(latent_bd[:, 0], latent_bd[:, 1], label=r'$g_{\theta}(C)$', c=colors[-3])
    # plt.streamplot(X, Y, U, V, color=('gray'), linewidth=0.5,
    #                density=0.9, arrowstyle='-|>', arrowsize=0.9)
    plt.quiver(X, Y, U, V, color='k',angles='xy',scale_units='xy', scale=1,headlength=8,headwidth=5)

    ax = fig.add_subplot(122, projection='3d')
    aug_data = aug_data.detach()
    ax.grid(None)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel(r'$x$', fontsize=ticksize, labelpad=-15)
    ax.set_ylabel(r'$y$', fontsize=ticksize, labelpad=-15)
    ax.set_zlabel(r'$z$', fontsize=ticksize, labelpad=-15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.plot(aug_data[:, 0], aug_data[:, 1], aug_data[:, 2], c=colors[-3])
    sub_data = output.detach()#aug_data[0:-1:60]
    F_ = Mitotic(sub_data)
    X, Y, Z = sub_data[:, 0], sub_data[:, 1], sub_data[:, 2]
    dX, dY, dZ = F_[:, 0], F_[:, 1], F_[:, 2]
    plt.quiver(X, Y, Z, dX, dY, dZ, color='k', )

    np.save('./data/Mitotic_quiverdata_sparse',{'M':sub_data,'F_M':F_,'D':input,'F_D':dinput})
    plt.show()
# uniformity()

def PRC_plot():
    import matplotlib
    rc_fonts = {
        "text.usetex": True,
        'text.latex.preview': True,  # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}""", r"""\usepackage{amsmath}""", r"""\usepackage{amsfonts}"""],
        'font.sans-serif': 'Times New Roman'
    }
    matplotlib.rcParams.update(rc_fonts)
    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'
    fontsize = 35
    fig = plt.figure(figsize=(4,3))
    plt.subplots_adjust(left=0.17, bottom=0.05, right=0.93, top=0.95)
    ax = plt.subplot(111)
    phi = torch.linspace(0,2*math.pi,100).view(-1,1)
    # phi = torch.linspace(-math.pi,math.pi,100).view(-1,1)
    dphi = auto_encoder.phase_sensitive_func(phi)
    phi = phi.detach().numpy()
    dphi = dphi.detach().numpy()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)
    plt.axhline(0,lw=4,c='k',zorder=0)
    plt.plot(phi[:],dphi[:,0],c=colors[0],lw=4,zorder=1)
    plt.text(2*math.pi-0.5,-0.45,r'$2\pi$',fontsize=fontsize)
    plt.xticks([])
    # plt.xticks([-math.pi,math.pi],[r'$-\pi$',r'$\pi$'],fontsize=ticksize)
    plt.yticks([-1,0,1],fontsize=fontsize)
    plt.xlim(0,2*math.pi)
    plt.ylim(-1.1,1.1)
    plt.tick_params(bottom=False,left=False)
    plt.savefig('./data/PRC_Mitotic.pdf')
    plt.show()
PRC_plot()