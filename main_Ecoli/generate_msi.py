import math
import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import fsolve,root
from functions import *

'''
hyperparameter
'''
D_in =  7 # input dimension
H1 = 10 * D_in  # hidden dimension
D_latent = 2 # latent dimension
edim = 7 # dimension of equant
for k in range(1):

    # Q=float(format(k*0.05,'.2f'))
    Q = 1.0
    '''
    models 
    '''
    auto_encoder = AE(D_in, H1, D_latent,w=0.0,edim=7)
    # auto_encoder.load_state_dict(torch.load('./data/ECC.pkl'))
    auto_encoder.load_state_dict(torch.load('./data/params/model_ECC_Q_{}_H1_{}.pkl'.format(Q,H1)))
    model = ECC(auto_encoder)

    '''
    loading data
    '''
    # data = torch.load('./data/train_ECC.pt')[80:]
    data = torch.load('./data/orbit/train_ECC_Q_{}.pt'.format(Q))
    data.requires_grad_(True)

    f=ECC_varyQ(data,Q)
    latent,phi,dphi,reverse = auto_encoder.forward(data)
    ptp = torch.max(phi) - torch.min(phi)
    inner = torch.sum(dphi*f,dim=1)
    loss = torch.mean((reverse - data) ** 2) + torch.mean((inner - auto_encoder.w) ** 2)
                    #+ (math.pi * 2 - ptp) ** 2 + torch.relu(1.0 - model.w ** 2)
    print( Q,k,"loss=", loss.item(),auto_encoder.w.item(),ptp.item())



'''
calculate
'''
def phase_reduction(model,data,deg=20):
    '''
    :return: fourier coefficients of \Gamma, the function type of the \Gamma
    '''
    data_gamma = model.gamma(data)
    L = len(data_gamma)
    data_gamma = data_gamma[torch.argsort(data_gamma[:,0])]


    coef,func = model.fourier_regression(data_gamma,deg)
    pred_gamma = func(data_gamma[:,0])
    _,func_kura = model.fourier_regression(data_gamma,1)
    pred_gamma_kura = func_kura(data_gamma[:,0])-coef[0]
    print(f'amplitude:{torch.max(data_gamma[:,1].abs())}')
    return coef,func,data_gamma,pred_gamma,pred_gamma_kura

# phase_reduction()
def run_model(model,data):
    start = timeit.default_timer()
    n = 10000
    dt = 0.02
    dim = 7
    num = 100

    coef, func,_,_,_ = phase_reduction(model,data)
    scale = 0.04 # deviation intensity of the beta
    delta_beta = torch.from_numpy(np.random.normal(0,scale,[num]))
    delta_beta = torch.linspace(-scale,scale,num)
    delta_w = model.delta_w(data,delta_beta)
    print(f'delta_w={delta_w}, std:{torch.std(delta_beta)}')
    # Q = 0.0
    simulate = ECC_dynamics(Q,delta_beta,delta_w, auto_encoder.w,func,model) # 0.1: , 0.01:0.43586340382844874

    setup_seed(1)

    # # warm up, running the dynamics to the limit cycle
    # x0 = torch.rand([1, dim * num])
    # # x0 = data[0:1, :].repeat(1, num) + torch.from_numpy(np.random.normal(0,0.5,[1,num*dim]))
    # t = torch.linspace(0,30,1000)
    # with torch.no_grad():
    #     x = odeint(simulate.forward,x0,t)[:,0,:]
    # # running on the limit cycle
    # x0 = x[-1:]
    # _,phi0,_ = auto_encoder.encoder(x0.view(num,dim).requires_grad_(True))
    # phi0 = phi0.view(1,-1)

    # phi0 = torch.linspace(0,math.pi,num).view(1,-1)
    # phi0 = torch.from_numpy(np.random.uniform(0,math.pi,num)).view(1,-1)
    # x0 = auto_encoder.decoder(phi0.T).view(1,-1)
    phi0 = torch.zeros([1, num])
    x0 = auto_encoder.decoder(phi0.T).view(1, -1)
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
    x = x.detach() # original orbit
    reverse_x = model.phi2x(phi).detach() # reconstructed orbit from phase trajectory
    phi = phi.detach()
    fontsize = 20
    ticksize = 18
    l = 1000
    # fig = plt.figure(figsize=(5,3))
    plt.subplot(121)
    for i in range(100):
        plt.plot(t[-l:], x[-l:, 1 + i * 7])
        plt.title('M={:.2f}'.format(cal_M(x[-1000:])))
        # plt.title('Q={:.2f}, scale={:.3f}, original'.format(Q,scale))
    plt.subplot(122)
    for i in range(100):
        plt.plot(t[-l:], reverse_x[-l:, 1 + i * 7],ls='--')
        # plt.title('R={:.2f}'.format(cal_R(phi[-1000:])))
        # plt.title('Q={:.2f}, scale={:.3f}, reconstruct'.format(Q, scale))
        plt.title('M={:.2f},R={}'.format(cal_M(reverse_x[-1000:]),cal_R(phi[-1000:])))
        # plt.plot(t[-l:], torch.sin(phi[-l:, i]), ls='--', c=colors[i])
    # plt.title(r'$\Delta\beta={},~Q={}$'.format(scale,Q),fontsize=fontsize)
    # plt.xlabel('Time',fontsize=fontsize)
    # plt.ylabel('mRNA level',fontsize=fontsize)
    # print(f'Initial freq={phi[0]}')

    plt.show()

# run_model(model,data)

def generate_results_int():
    start = timeit.default_timer()
    n = 10000
    dt = 0.02
    num = 100
    D_in = 7  # input dimension
    H1 = 10 * D_in  # hidden dimension
    D_latent = 2  # latent dimension
    res_M = torch.zeros([21,21])
    res_decM = torch.zeros([21,21])
    res_R = torch.zeros([21,21])
    for k in range(21):
        Q=float(format(k*0.05,'.2f'))
        # Q = 0.5
        '''
        models 
        '''
        auto_encoder = AE(D_in, H1, D_latent, w=0.0, edim=7)
        auto_encoder.load_state_dict(torch.load('./data/params/model_ECC_Q_{}_H1_{}.pkl'.format(Q, H1)))
        model = ECC(auto_encoder)
        '''
        loading data
        '''
        # data = torch.load('./data/train_ECC.pt')[80:]
        data = torch.load('./data/orbit/train_ECC_Q_{}.pt'.format(Q))
        data.requires_grad_(True)

        '''
        calculate results
        '''
        coef, func,data_gamma,pred_gamma,pred_gamma_kura = phase_reduction(model,data)
        # np.save('./data/results/prc_Q_{}'.format(Q),
        #         {'raw':data_gamma.detach().numpy(),'pred':pred_gamma.detach().numpy(),
        #          'kura':pred_gamma_kura.detach().numpy()})
        # for i,delta in enumerate(np.linspace(0,math.pi,21)):
        for i,scale in enumerate(np.linspace(0,0.2,21)):
        #     scale = 0.0 # deviation intensity of the beta
            delta_beta = torch.from_numpy(np.random.normal(0,scale**2,[num]))
            # delta_beta = torch.linspace(-scale**2,scale**2,num)
            delta_w = model.delta_w(data,delta_beta)
            print(f'ptp:{np.ptp(delta_beta.detach().numpy())}')

            simulate = ECC_dynamics(Q,delta_beta,delta_w, auto_encoder.w,func,model)

    #         def phase_dynamics(x):
    #             dx = simulate.phase_forward(0.0,torch.from_numpy(x).view(1,-1))-auto_encoder.w
    #             return dx[0].detach().numpy()*100.0
    #         roots = fsolve(phase_dynamics,np.zeros([num]))
    #         print(cal_R(torch.from_numpy(roots).view(1,-1)))
    #         R = cal_R(torch.from_numpy(roots).view(1,-1))
    #         res_R[k, i] = R
    # np.save('./data/results/Q beta pi num=100 max=0.2 R',
    #                     res_R)
    #         setup_seed(1)
    #         # phi0 = torch.linspace(0,math.pi,num).view(1,-1)
    #         # phi0 = torch.from_numpy(np.random.uniform(0,delta,num)).view(1,-1)
            phi0 = torch.zeros([1,num])
            x0 = auto_encoder.decoder(phi0.T).view(1,-1)
    #
            t = torch.linspace(0,n*dt,n)
            with torch.no_grad():
                x = odeint(simulate.forward,x0,t)[:,0,:]
                phi = odeint(simulate.phase_forward,phi0,t)[:,0,:]
    #         # print(f'Collecting data: Q={Q},delta={delta}, running time for num={num} is {timeit.default_timer()-start}')
            print(f'Collecting data: Q={Q},beta={scale}, running time for num={num} is {timeit.default_timer()-start}')
    #
            x = x.detach() # original orbit
            reverse_x = model.phi2x(phi).detach() # reconstructed orbit from phase trajectory
            phi = phi.detach()
            l = 1000
            orig_M = cal_M(x[-l:])
            dec_M = cal_M(reverse_x[-l:])
            R = cal_R(phi)
            res_M[k,i],res_decM[k,i],res_R[k,i] = orig_M,dec_M,R
    print(f'Done! Total running time is {timeit.default_timer() - start}')
    # np.save('./data/results/Q beta pi l=1000 num=100 max=0.2',
    #         {'orig_M':res_M.numpy(),'dec_M':res_decM.detach(),'R':res_R.detach()})

# generate_results_int()

def generate_results_ext():
    start = timeit.default_timer()
    n = 10000
    dt = 0.02
    num = 100
    D_in = 7  # input dimension
    H1 = 10 * D_in  # hidden dimension
    D_latent = 2  # latent dimension
    res_M = torch.zeros([21,21])
    res_decM = torch.zeros([21,21])
    res_R = torch.zeros([21,21])
    for k in range(21):
        Q=float(format(k*0.05,'.2f'))
        # Q = 0.5
        '''
        models 
        '''
        auto_encoder = AE(D_in, H1, D_latent, w=0.0, edim=7)
        auto_encoder.load_state_dict(torch.load('./data/params/model_ECC_Q_{}_H1_{}.pkl'.format(Q, H1)))
        model = ECC(auto_encoder)
        '''
        loading data
        '''
        data = torch.load('./data/orbit/train_ECC_Q_{}.pt'.format(Q))
        data.requires_grad_(True)

        '''
        calculate results
        '''
        coef, func,data_gamma,pred_gamma,pred_gamma_kura = phase_reduction(model,data)
        for i,scale in enumerate(np.linspace(0,0.2,21)):
        #     scale = 0.0 # deviation intensity of the beta
            delta_w = model.delta_w_ext(data)*scale**2

            simulate = ECC_dynamics_ext(Q,delta_w, auto_encoder.w,func,model,scale,auto_encoder)
            setup_seed(1)
            # phi0 = torch.linspace(0,math.pi,num).view(1,-1)
            # phi0 = torch.from_numpy(np.random.uniform(0,delta,num)).view(1,-1)
            phi0 = torch.zeros([1,num])
            x0 = auto_encoder.decoder(phi0.T).view(1,-1)

            t = torch.linspace(0,n*dt,n)
            # X = torch.zeros([n,num*7])
            # X[0:1,:] = x0
            # Y = torch.zeros([n,num*3])
            # Phi = torch.zeros([n,num])
            # Phi[0:1,:] = phi0
            # kappa = 15
            # w = torch.from_numpy(np.random.normal(0,1,[n,num*3]))
            # for j in range(n-1):
            #     x = X[j:j+1,:]
            #     y = Y[j:j+1,:]
            #     phi = Phi[j:j+1,:]
            #     dy = dt*(-kappa*y)+math.sqrt(dt)*w[j:j+1,:]*scale
            #     new_x = x + dt*(simulate.forward(0.0,x)) + simulate.forward_perturbed(dy,x)
            #     reverse,dphi = auto_encoder.dphi_reverse(phi.view(-1,1))
            #     new_phi = phi + dt * (simulate.phase_forward(0.0,phi)) #+ simulate.phase_perturbed(dy.view(-1,3),reverse.detach(),dphi.detach())
            #     new_y = y + dy
            #     X[j+1:j+2,:] = new_x
            #     Y[j+1:j+2,:] = new_y
            #     Phi[j+1:j+2,:] = new_phi

            with torch.no_grad():
                # X = odeint(simulate.forward_OU,x0,t,method='euler',options=dict(step_size=dt))[:,0,:]
                Phi = odeint(simulate.phase_forward_OU,phi0,t)[:,0,:]
            # # print(f'Collecting data: Q={Q},delta={delta}, running time for num={num} is {timeit.default_timer()-start}')
            print(f'Collecting data: Q={Q},beta={scale}, running time for num={num} is {timeit.default_timer()-start}')
            # X = X.detach() # original orbit
            # reverse_x = model.phi2x(Phi).detach() # reconstructed orbit from phase trajectory
            # Phi = Phi.detach()
            # l = 1000
            # orig_M = cal_M(X[-l:])
            # dec_M = cal_M(reverse_x[-l:])
            # R = cal_R(Phi[-l:])
            # res_M[k,i],res_decM[k,i],res_R[k,i] = orig_M,dec_M,R
    print(f'Done! Total running time is {timeit.default_timer() - start}')
    np.save('./data/results/Q beta pi l=1000 num=100 max=0.2 ext',
            {'orig_M':res_M.numpy(),'dec_M':res_decM.detach(),'R':res_R.detach()})


# generate_results_ext()


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
    # matplotlib.rcParams.update(rc_fonts)
    # matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'
    fontsize = 20
    ticksize = 18
    fig = plt.figure(figsize=(5,3))
    plt.subplots_adjust(left=0.14, bottom=0.18, right=0.86, top=0.95, hspace=0.25, wspace=0.2)
    _,_,data_gamma,pred_gamma,pred_gamma_kura = phase_reduction(model,data,20)
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
    aug_data = data # without augmentation, shape (T,dim)
    edim = aug_data.shape[1]
    latent, phi, dphi, reverse = auto_encoder.forward(data)
    equant = auto_encoder.surface.inverse(torch.zeros([1,edim]))[0]
    latent_ode, _, _ = auto_encoder.surface(aug_data)  # odeint(surface, data, t)[-1]
    pred_y_ode = auto_encoder.surface.inverse(latent_ode)  # odeint(surface.reverse, latent_ode, t)[-1]


    aug_data = aug_data.detach()
    latent_ode = latent_ode.detach()
    disk = generate_interior_data(latent_ode[0:-1:10,:2], torch.tensor([0.0, 0.0]), 3)
    aug_disk = augment(disk,edim) # the same space with aug_data
    aug_surface = auto_encoder.surface.inverse(aug_disk) # inverse of the disk D
    aug_surface = aug_surface.detach().numpy()
    orig_surface = generate_interior_data(aug_data[0:-1:10], equant, 3)
    orig_surface = orig_surface.detach().numpy()

    latent_ode = latent_ode.detach().numpy()
    pred_y_ode = pred_y_ode.detach().numpy()
    reverse = reverse.detach().numpy()
    latent = latent.detach().numpy()

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
    plt.scatter(equant[0], equant[1], s=80, marker='o', c=colors[-3], label='Equant')
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
    plt.plot(latent_ode[:, 0], latent_ode[:, 1], label=r'$g_{\theta}(C)$',c=colors[-1])
    plt.scatter(0, 0, s=80, marker='o', c=colors[-1], label='Origin')
    plt.plot(disk[:,0],disk[:,1],c=colors[-1],alpha=0.5)
    plt.xticks([-1,1],fontsize=ticksize)
    plt.yticks([-1, 1], fontsize=ticksize)
    plt.legend(loc=3,bbox_to_anchor=[-1.1,-0.35],fontsize=ticksize,ncol=3,handlelength=1.0,columnspacing=0.5,handletextpad=0.5,frameon=False)
    plt.title(r'$\text{Latent disk}~\mathbb{D}$', fontsize=fontsize)

    plt.show()

# plot_equant(data)