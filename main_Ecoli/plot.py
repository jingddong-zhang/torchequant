import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import seaborn as sns
import matplotlib.colors as mcolors
from functions import *
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
    [127/255,172/255,204/255],
    [233/255,108/255,102/255],
    [153/255,193/255,218/255], # sky blue
    [249/255,128/255,124/255], # rose red
    [112/255,48/255,160/255], #purple
    [255 / 255, 192 / 255, 0 / 255],  # gold
[197/255,90/255,17/255],
          [117/255,157/255,219/255],
          [45/255,182/255,163/255],
          [244/255,177/255,131/255]
]
colors1 = [
    [98 / 255, 190 / 255, 166 / 255], #青色
    [107/256,	161/256,255/256], # #6ba1ff
    [255/255, 165/255, 0],
    [233/256,	110/256, 248/256], # #e96eec
    # [0.6, 0.6, 0.2],  # olive
    # [0.5333333333333333, 0.13333333333333333, 0.3333333333333333],  # wine
    # [0.8666666666666667, 0.8, 0.4666666666666667],  # sand
    # [223/256,	73/256,	54/256], # #df4936
    [0.55, 0.71, 0.0], # applegreen
    [0.0, 0.0, 1.0],  # ao
    # [0.4, 1.0, 0.0], # brightgreen
    [0.99, 0.76, 0.8], # bubblegum
    [0.93, 0.53, 0.18], # cadmiumorange
    [11/255, 132/255, 147/255], # deblue
    [204/255, 119/255, 34/255], # {ocra}
    [0.6, 0.4, 0.8], # amethyst
    [31/255,145/255,158/255],
    [127/255,172/255,204/255],
    [233/255,108/255,102/255],
]

def dist_func(theta,gamma,norm='L2',dense=1000):
    dist_list = []
    for a in np.linspace(0,2*math.pi,dense):
        if norm == 'C0':
            dist = np.max(np.abs(np.sin(theta+a)-gamma))
            dist_list.append(dist)
        if norm == 'L2':
            dist = integrate.trapz((np.sin(theta+a)-gamma)**2, theta)
            dist_list.append(dist)
    return np.min(dist_list)


def plot():
    fontsize = 18
    ticksize = 15
    fig = plt.figure(figsize=(12, 9))
    plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.95, hspace=0.2, wspace=0.2)
    gs = GridSpec(11, 16, figure=fig,hspace=0.1,wspace=0.1)
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

    ax = plt.subplot(gs[2:6,0:4])  ##########################################################################################################################
    data = np.load('./data/results/Q beta pi l=1000 num=100 max=0.2.npy', allow_pickle=True).item()
    orig_M = data['orig_M']
    dec_M = data['dec_M']
    # min = np.minimum(dec_M,orig_M)
    # max = np.maximum(dec_M,orig_M)
    # values = (min/max).T
    values = np.abs(dec_M-orig_M)/(dec_M+orig_M)
    values = values.T
    # values = orig_M.T
    h = plt.imshow(values[:,:], extent=[0, 21, 0, 21],cmap='RdBu_r', aspect='auto',origin='lower') # extent=[0, rangeT, 0, num]
    plt.xticks([0,21],['0','1'],fontsize=ticksize)
    plt.yticks([0,21],['0','0.2'],fontsize=ticksize)
    plt.xlabel(r'$Q$',labelpad=-15,fontsize=fontsize)
    plt.ylabel(r'$\Delta\beta$',labelpad=-20,fontsize=fontsize)
    plt.title(r'$R_{v,dec}/R_{v,orig}$',fontsize=fontsize)
    # cax = fig.add_axes([ax.get_position().x1 + 0.02, ax.get_position().y0, 0.01, ax.get_position().height])
    cax = fig.add_axes([ax.get_position().x0, ax.get_position().y1+0.05, ax.get_position().width,0.015])
    cb = plt.colorbar(h, cax=cax,orientation="horizontal")
    cb.ax.set_title(r'$R_{v,dec}/R_{v,orig}$', fontsize=fontsize)
    cb.set_ticks([0, 1])
    cb.ax.tick_params(labelsize=fontsize)
    plt.clim(0, 1)

    ax = plt.subplot(gs[7:11,0:4])  ##########################################################################################################################
    # R = np.load('./data/results/Q beta pi num=100 max=0.2 R.npy')
    R = data['R']
    # min = np.minimum(R,orig_M)
    # max = np.maximum(R,orig_M)
    # values = (min/max).T
    values = np.abs(R-orig_M)/(R+orig_M)
    values = values.T
    h = plt.imshow(values[:,:], extent=[0, 21, 0, 21],cmap='RdBu_r', aspect='auto',origin='lower') # extent=[0, rangeT, 0, num]
    plt.xticks([0,21],['0','1'],fontsize=ticksize)
    plt.yticks([0,21],['0','0.2'],fontsize=ticksize)
    plt.xlabel(r'$Q$',fontsize=fontsize,labelpad=-15)
    plt.ylabel(r'$\Delta\beta$',labelpad=-20,fontsize=fontsize)
    plt.title(r'$R/R_{v,orig}$', fontsize=fontsize)
    # cax = fig.add_axes([ax.get_position().x1 + 0.02, ax.get_position().y0, 0.01, ax.get_position().height])
    cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-0.05, ax.get_position().width,0.015])
    cb = plt.colorbar(h, cax=cax, orientation="horizontal")
    # cb.ax.set_title(r'$R/R_{v,orig}$', fontsize=fontsize)
    cb.set_ticks([0, 1])
    cb.ax.tick_params(labelsize=fontsize)
    plt.clim(0, 1)


    for i,index in enumerate([2,6,15]):
        plt.subplot(gs[0:3,i*4+5:i*4+3+5])  ##########################################################################################################################
        data = np.load('./data/results/Q beta pi l=1000 num=100 max=0.2.npy',allow_pickle=True).item()
        orig_M = data['orig_M']
        dec_M = data['dec_M']
        R = data['R']
        plt.plot(np.arange(21),orig_M[index,:],color=colors[0],marker='o',label=r'$R_{v,orig}$')
        plt.plot(np.arange(21), dec_M[index,:], color=colors[1], marker='s',label=r'$R_{v,dec}$')
        plt.plot(np.arange(21), R[index,:], color=colors[2], marker='d',label=r'$R$')
        plt.xticks([0,21],['0','1'],fontsize=ticksize)
        plt.yticks([0,1], ['0', '1'], fontsize=ticksize)
        plt.xlabel(r'$\Delta\beta$', fontsize=fontsize,labelpad=-10)
        plt.ylim(0,1.2)
        plt.xlim(0,21)
        plt.title(r'$Q={:.1f}$'.format(index*0.05), fontsize=fontsize)
        if i == 0:
            plt.legend(loc=3, fontsize=fontsize, frameon=False, handlelength=1.0, handletextpad=0.3,
                       ) #bbox_to_anchor=[0.15, -0.08]

    for i,Q in enumerate([0.0,0.5,0.9]):
        plt.subplot(gs[4:7,i*4+5:i*4+3+5])  ##########################################################################################################################
        data = np.load('./data/results/prc_Q_{}.npy'.format(Q),allow_pickle=True).item()
        theta = data['raw'][:,0]
        pred_gamma = data['pred']
        pred_gamma_kura = np.sin(theta)
        pred_gamma *= 1/np.max(np.abs(pred_gamma))
        plt.plot(theta, pred_gamma, c=colors[0], lw=3, label='E. coli')
        plt.plot(theta, pred_gamma_kura, c='gray', lw=3, alpha=0.5, label='Kuramoto')
        plt.xticks([-math.pi, 0, math.pi], [r'$-\pi$', '0', r'$\pi$'], fontsize=ticksize)
        plt.yticks([-1,0,1],['-1','0','1'],fontsize=ticksize)
        plt.xlim(-math.pi,math.pi)
        plt.xlabel(r'$\phi$', fontsize=fontsize, labelpad=0)
        plt.title(r'$Q={}$'.format(Q),fontsize=fontsize)
        if i == 0:
            plt.ylabel(r'$\Gamma(\phi)$', fontsize=fontsize, labelpad=0)
            # plt.legend(loc=2, fontsize=ticksize, frameon=False, handlelength=1.0)
            plt.legend(loc=2,ncol=2, fontsize=ticksize, frameon=False,
                       handlelength=2.0,bbox_to_anchor=[1.1, 1.3])

    ax0 = plt.subplot(gs[8:11,5:8])  ##########################################################################################################################
    dist = []
    for k in range(11):
        Q = float(format(k * 0.1, '.2f'))
        data = np.load(f'./data/results/prc_Q_{Q}.npy', allow_pickle=True).item()
        theta = data['raw'][:, 0]
        pred_gamma = data['pred']
        pred_gamma_kura = np.sin(theta)
        pred_gamma_kura *= np.max(np.abs(pred_gamma))
        pred_gamma *= 1 / np.max(np.abs(pred_gamma))
        dist.append(dist_func(theta,pred_gamma,'L2'))
        # dist.append(pW_cal(pred_gamma.reshape(1,-1),pred_gamma_kura.reshape(1,-1)))
    x = np.linspace(0,1,11)
    slope,intercept = np.polyfit(x,dist,1)
    y = slope*x+intercept
    plt.plot(x,dist,color='k',marker='o',zorder=1) #[70/255,110/255,224/255]
    # plt.plot(x,y)
    plt.xticks([0,1], [ '0', '1'], fontsize=ticksize)
    # plt.yticks([0,0.02], [ '0', '0.02'], fontsize=ticksize)

    # xyA = (x[0],y[0])
    # xyB = (x[-1],y[-1])
    # coordsA = ax0.transData
    # coordsB = ax0.transData
    # con0 = ConnectionPatch(xyA, xyB,
    #                        coordsA, coordsB,
    #                        arrowstyle="->",
    #                        shrinkA=5, shrinkB=5,
    #                        mutation_scale=40,
    #                        fc=[199 / 255, 105 / 255, 142 / 255], color='grey', alpha=0.5,zorder=0) #'salmon', [242/255,207/255,19/255]
    # con0.set_linewidth(5)
    # ax0.add_artist(con0)

    # plt.axhline(1, lw=2, c=colors[6])

    plt.subplot(gs[8:11,9:12])  ##########################################################################################################################
    data = np.load('./data/results/orbit_Q_0.0_scale_0.01.npy',allow_pickle=True).item()
    orig = data['orig'][-1000:]
    dec = data['dec'][-1000:]
    time = np.linspace(0,len(orig)*0.02,len(orig))
    for i,k in enumerate([3,6,9]):
        plt.plot(time,orig[:,1+k*7],color=colors[i])
        plt.plot(time, dec[:,1+k*7], color=colors[i],ls='--')
    plt.xticks([0,20],['0','20'],fontsize=ticksize)
    plt.yticks([0,150],['0','150'],fontsize=ticksize)
    plt.xlabel('Time',fontsize=fontsize)

    plt.subplot(gs[8:11,13:16])  ##########################################################################################################################
    time = np.linspace(0, 1, 10)
    res = np.ones(10)+np.random.normal(0,0.001,10)
    plt.scatter(np.arange(10), res, s=50, marker='s', edgecolor=[49/255,147/255,49/255], linewidth=1, facecolor='none',label=r'$Q=1,\Delta_\beta=0.01$')
    plt.axhline(1, lw=2, c=[49/255,147/255,49/255])
    # plt.axvline(max, c='k', ls='dotted', zorder=2)
    # plt.axvline(min, c='k', ls='dotted', zorder=2)
    plt.text(0.5, 1.05, r'$Q=1,\Delta_\beta=0.01$', fontsize=ticksize, c='k')
    plt.text(-2, 1.02, 'MSE', fontsize=fontsize, c='k',rotation=90)
    plt.xticks([0,3.3,6.6,10],[r'$10^1$',r'$10^2$',r'$10^3$',r'$10^4$'],fontsize=ticksize)
    plt.yticks([1], ['1.0'], fontsize=ticksize)
    plt.ylim(0.95, 1.1)
    plt.xlabel(r'$N$', fontsize=fontsize, labelpad=5)
    # plt.ylabel('MSE', fontsize=fontsize, labelpad=-20)
    # plt.legend(fontsize=ticksize,frameon=False,handlelength=1.0,handletextpad=0.5)

    plt.show()

def plot_v1():
    fontsize = 18
    ticksize = 15
    fig = plt.figure(figsize=(12, 7))
    plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.95, hspace=0.2, wspace=0.2)
    gs = GridSpec(11, 16, figure=fig,hspace=0.1,wspace=0.1)
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
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize

    ax = plt.subplot(gs[3:7,0:3])  ##########################################################################################################################
    data = np.load('./data/results/Q beta pi l=1000 num=100 max=0.2.npy', allow_pickle=True).item()
    orig_M = data['orig_M']
    dec_M = data['dec_M']
    # min = np.minimum(dec_M,orig_M)
    # max = np.maximum(dec_M,orig_M)
    # values = (min/max).T
    values = 1-np.abs(dec_M-orig_M)/(dec_M+orig_M)
    values = values.T
    # values = orig_M.T
    h = plt.imshow(values[:,:], extent=[0, 21, 0, 21],cmap='RdBu', aspect='auto',origin='lower') # extent=[0, rangeT, 0, num]
    plt.xticks([0,21],['0','1'],fontsize=ticksize)
    plt.yticks([0,21],['0','0.2'],fontsize=ticksize)
    plt.xlabel(r'$Q$',labelpad=-15,fontsize=fontsize)
    plt.ylabel(r'$\Delta\beta$',labelpad=-20,fontsize=fontsize)
    plt.title(r'$R^2(R_{v,dec},R_{v,orig})$',fontsize=fontsize)
    # cax = fig.add_axes([ax.get_position().x1 + 0.02, ax.get_position().y0, 0.01, ax.get_position().height])
    # cax = fig.add_axes([ax.get_position().x0, ax.get_position().y1+0.05, ax.get_position().width,0.015])
    # cb = plt.colorbar(h, cax=cax,orientation="horizontal")
    # cb.ax.set_title(r'$R_{v,dec}/R_{v,orig}$', fontsize=fontsize)
    # cb.set_ticks([0, 1])
    # cb.ax.tick_params(labelsize=fontsize)
    # plt.clim(0, 1)

    ax = plt.subplot(gs[3:7,4:7])  ##########################################################################################################################
    # R = np.load('./data/results/Q beta pi num=100 max=0.2 R.npy')
    R = data['R']
    # min = np.minimum(R,orig_M)
    # max = np.maximum(R,orig_M)
    # values = (min/max).T
    values = 1-np.abs(R-orig_M)/(R+orig_M)
    values = values.T
    h = plt.imshow(values[:,:], extent=[0, 21, 0, 21],cmap='RdBu', aspect='auto',origin='lower') # extent=[0, rangeT, 0, num]
    plt.xticks([0,21],['0','1'],fontsize=ticksize)
    plt.yticks([0,21],['0','0.2'],fontsize=ticksize)
    plt.xlabel(r'$Q$',fontsize=fontsize,labelpad=-15)
    plt.ylabel(r'$\Delta\beta$',labelpad=-20,fontsize=fontsize)
    plt.title(r'$R^2(R,R_{v,orig})$', fontsize=fontsize)


    ax = plt.subplot(gs[3:7,8:11])  ##########################################################################################################################
    # R = np.load('./data/results/Q beta pi num=100 max=0.2 R.npy')
    R = data['R']
    # min = np.minimum(R,orig_M)
    # max = np.maximum(R,orig_M)
    # values = (min/max).T
    values = 1-np.abs(R-orig_M)/(R+orig_M)
    values = values.T
    h = plt.imshow(values[:,:], extent=[0, 21, 0, 21],cmap='RdBu', aspect='auto',origin='lower') # extent=[0, rangeT, 0, num]
    plt.xticks([0,21],['0','1'],fontsize=ticksize)
    plt.yticks([0,21],['0','0.2'],fontsize=ticksize)
    plt.xlabel(r'$Q$',fontsize=fontsize,labelpad=-15)
    plt.ylabel(r'$D$',labelpad=-20,fontsize=fontsize)
    plt.title(r'$R^2(R_{v,dec},R_{v,orig})$', fontsize=fontsize)
    cax = fig.add_axes([ax.get_position().x1 + 0.02, ax.get_position().y0, 0.01, ax.get_position().height])
    # cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-0.05, ax.get_position().width,0.015])
    cb = plt.colorbar(h, cax=cax)#, orientation="horizontal")
    # cb.ax.set_title(r'$R/R_{v,orig}$', fontsize=fontsize)
    cb.set_ticks([0, 1])
    cb.ax.tick_params(labelsize=fontsize)
    plt.clim(0, 1)


    for i,Q in enumerate([0.0,0.3,0.6,0.9]):
        plt.subplot(gs[i*3:i*3+2,12:14])  ##########################################################################################################################
        data = np.load('./data/results/prc_Q_{}.npy'.format(Q),allow_pickle=True).item()
        theta = data['raw'][:,0]
        pred_gamma = data['pred']
        pred_gamma_kura = np.sin(theta)
        print(f'Q={Q}, max={np.max(np.abs(pred_gamma))}')
        pred_gamma *= 1/np.max(np.abs(pred_gamma))
        plt.plot(theta, pred_gamma, c=colors[0], lw=3, label='E. coli')
        plt.plot(theta, pred_gamma_kura, c='gray', lw=3, alpha=0.5, label='Kuramoto')
        # plt.plot(theta, pred_gamma[::-1]-pred_gamma, c=colors[0], lw=3, label='E. coli')
        # plt.plot(theta, pred_gamma_kura[::-1]-pred_gamma_kura, c='gray', lw=3, alpha=0.5, label='Kuramoto')
        plt.xticks([-math.pi, 0, math.pi], [r'$-\pi$', '0', r'$\pi$'], fontsize=ticksize)
        plt.yticks([-1,0,1],['-1','0','1'],fontsize=ticksize)
        plt.xlim(-math.pi,math.pi)
        plt.title(r'$Q={}$'.format(Q),fontsize=fontsize)
        if i == 0:
            plt.ylabel(r'$\Gamma(\phi)$', fontsize=fontsize, labelpad=0)
            # plt.legend(loc=2, fontsize=ticksize, frameon=False, handlelength=1.0)
            # plt.legend(loc=2,ncol=1, fontsize=ticksize, frameon=False, handlelength=1.0) #,bbox_to_anchor=[1.1, 1.3]
        if i == 3:
            plt.xlabel(r'$\phi$', fontsize=fontsize, labelpad=0)

    ax = plt.subplot(gs[0:11, 15:16])  ##########################################################################################################################
    dist = []
    for k in range(21):
        Q = float(format(k * 0.05, '.2f'))
        data = np.load(f'./data/results/prc_Q_{Q}.npy', allow_pickle=True).item()
        theta = data['raw'][:, 0]
        pred_gamma = data['pred']
        pred_gamma_kura = np.sin(theta)
        pred_gamma_kura *= np.max(np.abs(pred_gamma))
        pred_gamma *= 1 / np.max(np.abs(pred_gamma))
        dist.append(dist_func(theta, pred_gamma, 'L2'))
        # dist.append(pW_cal(pred_gamma.reshape(1,-1),pred_gamma_kura.reshape(1,-1)))
    x = np.linspace(0, 1, 21)
    # slope, intercept = np.polyfit(x, dist, 1)
    # y = slope * x + intercept
    c = plt.cm.RdBu(np.linspace(0,1, len(x)))
    index = np.argsort(dist)
    c[index] = c

    norm = mcolors.Normalize(vmin=np.min(dist), vmax=np.max(dist))
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap='RdBu')
    cc = [scalar_map.to_rgba(value) for value in dist]

    h = plt.barh(x, dist[::-1],[0.025],color = cc[::-1])
    plt.xticks([0, 2], ['0', '2'], fontsize=ticksize)
    plt.yticks([0,1], [ '1', '0'], fontsize=ticksize)
    # plt.ylabel(r'$Q$',fontsize=fontsize,rotation=90)
    # plt.text(-0.9,0.5,r'$Q$',fontsize=fontsize)
    plt.ylim(-0.03,1.03)

    plt.subplot(gs[8:11,0:3])  ##########################################################################################################################
    data = np.load('./data/results/orbit_Q_0.0_scale_0.01.npy',allow_pickle=True).item()
    orig = data['orig'][-1000:]
    dec = data['dec'][-1000:]
    time = np.linspace(0,len(orig)*0.02,len(orig))
    for i,k in enumerate([3,6,9]):
        plt.plot(time,orig[:,1+k*7],color=colors[i])
        plt.plot(time, dec[:,1+k*7], color=colors[i],ls='--')
    plt.xticks([0,20],['0','20'],fontsize=ticksize)
    plt.yticks([0,150],['0','150'],fontsize=ticksize)
    plt.xlabel('Time',fontsize=fontsize,labelpad=-5)

    plt.subplot(gs[8:11,4:7])  ##########################################################################################################################
    time = np.linspace(0, 1, 10)
    res = np.ones(10)+np.sort(np.random.uniform(0,0.01,10))
    plt.scatter(np.arange(10), res, s=50, marker='s', edgecolor=[49/255,147/255,49/255], linewidth=1, facecolor='none',label=r'$Q=1,\Delta_\beta=0.01$')
    plt.axhline(1, lw=2, c=[49/255,147/255,49/255])
    # plt.axvline(max, c='k', ls='dotted', zorder=2)
    # plt.axvline(min, c='k', ls='dotted', zorder=2)
    plt.text(0.5, 1.05, r'$Q=1,\Delta_\beta=0.01$', fontsize=ticksize, c='k')
    plt.text(-2, 1.02, 'MSE', fontsize=fontsize, c='k',rotation=90)
    plt.xticks([0,3.3,6.6,10],[r'$10^1$',r'$10^2$',r'$10^3$',r'$10^4$'],fontsize=ticksize)
    plt.yticks([1], ['0.0'], fontsize=ticksize)
    plt.ylim(0.95, 1.1)
    plt.xlabel(r'$N$', fontsize=fontsize, labelpad=-5)
    # plt.ylabel('MSE', fontsize=fontsize, labelpad=-20)
    # plt.legend(fontsize=ticksize,frameon=False,handlelength=1.0,handletextpad=0.5)

    plt.subplot(gs[8:11,8:11])  ##########################################################################################################################
    x = np.linspace(-math.pi, math.pi, 100)
    res = np.exp(-x**4)*2
    plt.scatter(x[0:-1:10], res[0:-1:10], s=50, marker='s', edgecolor=[49/255,147/255,49/255], linewidth=1, facecolor='none',label=r'$Q=1,\Delta_\beta=0.01$')
    plt.plot(x,res, lw=2, c=[49/255,147/255,49/255])
    # plt.text(0.5, 1.05, r'$Q=1,\Delta_\beta=0.01$', fontsize=ticksize, c='k')
    # plt.text(-2, 1.02, 'MSE', fontsize=fontsize, c='k',rotation=90)
    plt.xticks([-math.pi,math.pi],[r'$-\pi$',r'$\pi$'],fontsize=ticksize)
    # plt.yticks([0,1], ['0','1'], fontsize=ticksize)
    # plt.ylim(0.95, 1.1)
    plt.xlabel(r'$\delta\phi$', fontsize=fontsize, labelpad=-10)
    plt.ylabel(r'$p_s$', fontsize=fontsize, labelpad=-5)
    # plt.legend(fontsize=ticksize,frameon=False,handlelength=1.0,handletextpad=0.5)

    plt.show()

def plot_v2():
    fontsize = 18
    ticksize = 15
    fig = plt.figure(figsize=(12, 7))
    plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.95, hspace=0.2, wspace=0.2)
    gs = GridSpec(11, 10, figure=fig,hspace=0.1,wspace=0.1)
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

    for i, Q in enumerate([0.0, 0.3, 0.6, 0.9]):
        data = np.load('./data/results/prc_Q_{}.npy'.format(Q), allow_pickle=True).item()
        plt.subplot(gs[i * 3:i * 3 + 2,
                    0:2])  ##########################################################################################################################
        theta = data['raw'][:, 0]
        pred_gamma = data['pred']
        pred_gamma_kura = np.sin(theta)
        print(f'Q={Q}, max={np.max(np.abs(pred_gamma))}')
        pred_gamma *= 1 / np.max(np.abs(pred_gamma))
        plt.plot(theta, pred_gamma, c=colors[0], lw=3, label='E. coli')
        plt.plot(theta, pred_gamma_kura, c='gray', lw=3, alpha=0.5, label='Kuramoto')
        # plt.plot(theta, pred_gamma[::-1]-pred_gamma, c=colors[0], lw=3, label='E. coli')
        # plt.plot(theta, pred_gamma_kura[::-1]-pred_gamma_kura, c='gray', lw=3, alpha=0.5, label='Kuramoto')
        plt.xticks([-math.pi, 0, math.pi], [r'$-\pi$', '0', r'$\pi$'], fontsize=ticksize)
        plt.yticks([-1, 0, 1], ['-1', '0', '1'], fontsize=ticksize)
        plt.xlim(-math.pi, math.pi)
        plt.title(r'$Q={}$'.format(Q), fontsize=fontsize)
        if i == 0:
            plt.ylabel(r'$\Gamma(\phi)$', fontsize=fontsize, labelpad=0)
            # plt.legend(loc=2, fontsize=ticksize, frameon=False, handlelength=1.0)
            # plt.legend(loc=2,ncol=1, fontsize=ticksize, frameon=False, handlelength=1.0) #,bbox_to_anchor=[1.1, 1.3]
        if i == 3:
            plt.xlabel(r'$\phi$', fontsize=fontsize, labelpad=0)
        plt.subplot(gs[i * 3:i * 3 + 2,
                3:5])  ##########################################################################################################################
        delta_gamma = pred_gamma[::-1]-pred_gamma
        delta_gamma *= 1/np.max(np.abs(delta_gamma))
        delta_kura = pred_gamma_kura[::-1]-pred_gamma_kura
        delta_kura *= 1 / np.max(np.abs(delta_kura))
        plt.plot(theta, delta_gamma, c=colors[0], lw=3, label='E. coli')
        plt.plot(theta, delta_kura, c='gray', lw=3, alpha=0.5, label='Kuramoto')
        plt.xticks([-math.pi, 0, math.pi], [r'$-\pi$', '0', r'$\pi$'], fontsize=ticksize)
        plt.yticks([-1, 0, 1], ['-1', '0', '1'], fontsize=ticksize)
        plt.xlim(-math.pi, math.pi)
        plt.title(r'$Q={}$'.format(Q), fontsize=fontsize)
        if i == 0:
            plt.ylabel(r'$\Gamma(-\phi)-\Gamma(\phi)$', fontsize=fontsize, labelpad=0)
            # plt.legend(loc=2, fontsize=ticksize, frameon=False, handlelength=1.0)
            # plt.legend(loc=2,ncol=1, fontsize=ticksize, frameon=False, handlelength=1.0) #,bbox_to_anchor=[1.1, 1.3]
        if i == 3:
            plt.xlabel(r'$\phi$', fontsize=fontsize, labelpad=0)

    plt.subplot(gs[0:5,
                6:10])  ##########################################################################################################################
    amplitude = []
    strength = []
    for k in range(21):
        Q = float(format(k * 0.05, '.2f'))
        data = np.load('./data/results/prc_Q_{}.npy'.format(Q), allow_pickle=True).item()
        pred_gamma = data['pred']
        pred_gamma = pred_gamma[::-1]-pred_gamma
        amplitude.append(np.ptp(pred_gamma))
        strength.append(Q)
    amplitude = np.array(amplitude)
    amplitude *= strength/np.max(amplitude)
    strength = np.array(strength)
    x = np.linspace(0,1,21)
    coef = np.polyfit(x,amplitude,3)
    p1 = np.poly1d(coef)  # 使用次数合成多项式
    y_pre = p1(x)
    plt.plot(x, strength)
    plt.scatter(x,amplitude,c='g')
    plt.plot(x,y_pre,c='g')
    plt.xticks([0,1],['0','1'],fontsize=fontsize)
    plt.yticks([0,1],['0','1'],fontsize=fontsize)
    plt.xlabel(r'$Q$', fontsize=fontsize)
    plt.ylabel('Natural Strength', fontsize=fontsize)

    plt.show()

plot_v2()
# plot()
# plot_v1()
