import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch

from functions import *

'''
hyperparameter
'''
D_in =  7 # input dimension
H1 = 20 * D_in  # hidden dimension
D_latent = 2 # latent dimension
edim = 7 # dimension of equant

'''
models 
'''
for k in range(1):
    # Q = float(format(k * 0.05, '.2f'))
    Q = 0.0
    auto_encoder = AE(D_in, H1, D_latent,w=0.43,edim=7)
    auto_encoder.load_state_dict(torch.load('./data/model_ECC_equant.pkl'))
    # auto_encoder.load_state_dict(torch.load('./data/params/model_ECC_Q_{}_H1_70.pkl'.format(Q)))
    model = ECC(auto_encoder)

    '''
    loading data
    '''
    data = torch.load('./data/train_ECC.pt')[80:]
    # data = torch.load('./data/orbit/train_ECC_Q_{}.pt'.format(Q))
    data.requires_grad_(True)

    f = ECC_varyQ(data,Q)
    latent,phi,dphi,reverse = auto_encoder.forward(data)
    ptp = torch.max(phi) - torch.min(phi)
    inner = torch.sum(dphi*f,dim=1)
    loss = torch.mean((reverse - data) ** 2) + torch.mean((inner - auto_encoder.w) ** 2)
                    #+ (math.pi * 2 - ptp) ** 2 + torch.relu(1.0 - model.w ** 2)
    print( "loss=", loss.item(),auto_encoder.w.item(),ptp.item())


'''
calculate
'''
def phase_reduction():
    '''
    :return: fourier coefficients of \Gamma, the function type of the \Gamma
    '''
    data_gamma = model.gamma(data)
    L = len(data_gamma)
    data_gamma = data_gamma[torch.argsort(data_gamma[:,0])]


    coef,func = model.fourier_regression(data_gamma)
    pred_gamma = func(data_gamma[:,0])
    _,func_kura = model.fourier_regression(data_gamma,1)
    pred_gamma_kura = func_kura(data_gamma[:,0])-coef[0]
    print(f'coef:{coef}')
    return coef,func,data_gamma,pred_gamma,pred_gamma_kura

# phase_reduction()
def run_model():
    start = timeit.default_timer()
    n = 10000
    dt = 0.02
    dim = 7
    num = 100

    coef, func,_,_,_ = phase_reduction()
    scale = 0.00
    delta_beta = torch.from_numpy(np.random.normal(0,scale,[num]))
    delta_w = model.delta_w(data,delta_beta)
    print(f'delta_w={delta_w}')
    # Q = 0.8
    simulate = ECC_dynamics(Q,delta_beta,delta_w, auto_encoder.w,func,model) # 0.1: , 0.01:0.43586340382844874

    setup_seed(1)

    # # warm up, running the dynamics to the limit cycle
    # x0 = torch.rand([1, dim * num])
    # # x0 = data[0:1, :].repeat(1, num) + torch.from_numpy(np.random.normal(0,1,[1,num*dim]))
    # t = torch.linspace(0,30,1000)
    # with torch.no_grad():
    #     x = odeint(simulate.forward,x0,t)[:,0,:]
    # # running on the limit cycle
    # x0 = x[-1:]
    # _,phi0,_ = auto_encoder.encoder(x0.view(num,dim).requires_grad_(True))
    # phi0 = phi0.view(1,-1)

    phi0 = torch.from_numpy(np.random.uniform(0,1.0*math.pi,num)).view(1,-1)
    # phi0 = torch.linspace(0,1.0*math.pi,num).view(1,-1)
    x0 = model.phi2x(phi0.T).view(1,-1)

    # x0 = data[0:1, :].repeat(1, num)
    # _,phi0,_ = auto_encoder.encoder(x0.view(num,dim).requires_grad_(True))
    # phi0 = phi0.view(1,-1)

    t = torch.linspace(0,n*dt,n)
    with torch.no_grad():
        x = odeint(simulate.forward,x0,t)[:,0,:]
        phi = odeint(simulate.phase_forward,phi0,t)[:,0,:]
    print(f'running time for num={num} is {timeit.default_timer()-start}')
    # _,_,_,reverse_x = auto_encoder.forward(x[:,:7].requires_grad_(True))
    # reverse_x = reverse_x.detach().numpy()
    x = x.detach()
    reverse_x = model.phi2x(phi).detach()
    phi = phi.detach()
    fontsize = 20
    ticksize = 18
    l = 1000
    # fig = plt.figure(figsize=(5,3))
    plt.subplot(121)
    for i in range(10):
        plt.plot(t[-l:], x[-l:, 1 + i * 7])
        plt.title('M={:.2f}'.format(cal_M(x[-1000:])))
        # plt.title('Q={:.2f}, scale={:.3f}, original'.format(Q,scale))
    plt.subplot(122)
    for i in range(10):
        plt.plot(t[-l:], reverse_x[-l:, 1 + i * 7],ls='--',c=colors[i])
        # plt.title('R={:.2f}'.format(cal_R(phi[-1000:])))
        # plt.title('Q={:.2f}, scale={:.3f}, Dec+phase'.format(Q, scale))
        plt.title('M={:.2f} R={:.2f}'.format(cal_M(reverse_x[-1000:]),cal_R(phi[-1000:])))
        # plt.plot(t[-l:], torch.sin(phi[-l:, i]), ls='--', c=colors[i])
    # plt.title(r'$\Delta\beta={},~Q={}$'.format(scale,Q),fontsize=fontsize)
    # plt.xlabel('Time',fontsize=fontsize)
    # plt.ylabel('mRNA level',fontsize=fontsize)
    # print(f'Initial freq={phi[0]}')
    np.save('./data/results/orbit_Q_{}_scale_{}'.format(Q,scale),{'orig':x.numpy(),'dec':reverse_x.numpy()})
    plt.show()

# run_model()

'''
visualizing
'''
def table_plot():
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
    fig = plt.figure(figsize=(5,3))
    plt.subplots_adjust(left=0.14, bottom=0.18, right=0.86, top=0.95, hspace=0.25, wspace=0.2)
    _,_,data_gamma,pred_gamma,pred_gamma_kura = phase_reduction()
    plt.plot(data_gamma[:,0],pred_gamma,c=colors[0],lw=3,label='ECC')
    plt.plot(data_gamma[:,0],pred_gamma_kura,c='gray',lw=3,alpha=0.5,label='Kuramoto')
    plt.xticks([-math.pi,0,math.pi],[r'$-\pi$','0',r'$\pi$'],fontsize=ticksize)
    # plt.yticks([-0.0003,0,0.0002],['-3','0','2'],fontsize=ticksize)
    plt.xlabel(r'$\phi$',fontsize=fontsize,labelpad=0)
    plt.ylabel(r'$\Gamma(\phi)(\times 10^{-3})$',fontsize=fontsize,labelpad=0)
    plt.legend(loc=2,fontsize=ticksize,frameon=False,handlelength=1.0)
    plt.show()

# table_plot()

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

    aug_data = aug_data.detach()
    latent_bd = latent_bd.detach()
    disk = generate_interior_data(latent_bd[0:-1:10,:2], torch.tensor([0.0, 0.0]), num_samples)
    aug_disk = augment(disk,edim) # the same space with aug_data
    aug_surface = auto_encoder.surface.inverse(aug_disk) # inverse of the disk D
    aug_surface = aug_surface.detach().numpy()
    orig_surface = generate_interior_data(aug_data[0:-1:10], equant, num_samples)
    latent_surface, _, _ = auto_encoder.surface(orig_surface)
    latent_surface = latent_surface.detach().numpy()
    orig_surface = orig_surface.detach().numpy()
    print(orig_surface.shape,aug_data.shape,aug_data[0:-1:10].shape)

    latent_bd = latent_bd.detach().numpy()
    pred_y_bd = pred_y_bd.detach().numpy()
    reverse = reverse.detach().numpy()
    latent = latent.detach().numpy()
    np.save(f'./data/Ecoli_equant_{num_samples}',{'equant':equant.detach().numpy(),'M':orig_surface,'C':aug_data,'L_M':aug_surface,
                             'D':disk,'S^1':latent,'L_S^1':latent_bd,'w':auto_encoder.w.detach().numpy(),'f':(10000-1)/(len(data)*200),
                                                  'phase':phi.detach().numpy(),'L_D':latent_surface,'loss_angle':loss_angle.detach().numpy(),
                                                     'loss_phase':loss_phase.detach().numpy()})
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
    ax.plot(orig_surface[:, 0], orig_surface[:, 1], orig_surface[:, 2], c=colors[-3], alpha=0.5)
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
    ax.plot(aug_surface[:,0],aug_surface[:,1],aug_surface[:,2],c=colors[-3],alpha=0.5)
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
    plt.plot(disk[:,0],disk[:,1],c=colors[-1],alpha=0.5)
    plt.xticks([-1,1],fontsize=ticksize)
    plt.yticks([-1, 1], fontsize=ticksize)
    plt.legend(loc=3,bbox_to_anchor=[-1.1,-0.35],fontsize=ticksize,ncol=3,handlelength=1.0,columnspacing=0.5,handletextpad=0.5,frameon=False)
    plt.title(r'$\text{Latent disk}~\mathbb{D}$', fontsize=fontsize)

    plt.show()

# plot_equant(data)

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
    fontsize = 30
    fig = plt.figure(figsize=(4,3))
    plt.subplots_adjust(left=0.25, bottom=0.03, right=0.93, top=0.93)
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
    plt.plot(phi[:],dphi[:,6],c=colors[0],lw=4,zorder=1)
    plt.text(2*math.pi-0.5,-0.015,r'$2\pi$',fontsize=fontsize)
    plt.xticks([])
    # plt.xticks([-math.pi,math.pi],[r'$-\pi$',r'$\pi$'],fontsize=ticksize)
    plt.yticks([-0.03,0.03],fontsize=fontsize)
    plt.xlim(0,2*math.pi)
    # plt.ylim(-1.1,1.3)
    plt.tick_params(bottom=False,left=False)
    plt.savefig('./data/PRC_Ecoli.pdf')
    plt.show()
# PRC_plot()