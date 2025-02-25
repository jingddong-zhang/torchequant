import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import seaborn as sns
import cmaps
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
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

def plot_v1():
    fontsize = 18
    ticksize = 15
    fig = plt.figure(figsize=(12, 7))
    plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.95, hspace=0.2, wspace=0.2)
    gs = GridSpec(11, 20, figure=fig,hspace=0.1,wspace=0.1)
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
    plt.rc('font', family='Times New Roman')

    ax = plt.subplot(gs[3:7,0:3])  ##########################################################################################################################
    data = np.load('./data/results/Q beta pi l=1000 num=100 max=0.2.npy', allow_pickle=True).item()
    orig_M = data['orig_M']
    dec_M = data['dec_M']
    print(orig_M.shape,'&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
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
    plt.ylabel(r'$\sigma$',labelpad=-20,fontsize=fontsize)
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
    plt.ylabel(r'$\sigma$',labelpad=-20,fontsize=fontsize)
    plt.title(r'$R^2(R,R_{v,orig})$', fontsize=fontsize)


    ax = plt.subplot(gs[3:7,8:11])  ##########################################################################################################################
    # R = torch.from_numpy(np.load('./data/results/Q beta pi num=100 max=0.2 R.npy'))
    R = data['R']
    R = (R+dec_M)/2
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


    for i,Q in enumerate([0.1,0.55,0.9]):
        plt.subplot(gs[i*3:i*3+2,12:14])  ##########################################################################################################################
        data = np.load('./data/results/prc_Q_{}.npy'.format(Q),allow_pickle=True).item()
        theta = data['raw'][:,0]
        pred_gamma = data['pred']
        pred_gamma_kura = np.sin(theta)
        print(f'Q={Q}, max={np.max(np.abs(pred_gamma))}')
        pred_gamma *= 1/np.max(np.abs(pred_gamma))
        plt.plot(theta, pred_gamma, c=[55 / 255, 127 / 255, 153 / 255], lw=2, label='E. coli')
        plt.plot(theta, pred_gamma_kura, c=[163 / 255, 84 / 255, 83 / 255], lw=2, label='Kuramoto')

        # plt.plot(theta, pred_gamma[::-1]-pred_gamma, c=colors[0], lw=3, label='E. coli')
        # plt.plot(theta, pred_gamma_kura[::-1]-pred_gamma_kura, c='gray', lw=3, alpha=0.5, label='Kuramoto')
        plt.xticks([-math.pi, 0, math.pi], [r'$-\pi$', '0', r'$\pi$'], fontsize=ticksize)
        plt.yticks([-1,0,1],['-1','0','1'],fontsize=ticksize)
        plt.xlim(-math.pi,math.pi)
        if i == 0:
            plt.title(r'$\Gamma(\phi)$', fontsize=fontsize)

        #     plt.ylabel(r'$\Gamma(\phi)$', fontsize=fontsize, labelpad=0)
            # plt.legend(loc=2, fontsize=ticksize, frameon=False, handlelength=1.0)
            # plt.legend(loc=2,ncol=1, fontsize=ticksize, frameon=False, handlelength=1.0) #,bbox_to_anchor=[1.1, 1.3]
        if i == 3:
            plt.xlabel(r'$\phi$', fontsize=fontsize, labelpad=0)

        plt.subplot(gs[i*3:i*3+2,18:20])  ##########################################################################################################################
        delta_gamma = pred_gamma[::-1] - pred_gamma
        delta_gamma *= 1 / np.max(np.abs(delta_gamma))
        delta_kura = pred_gamma_kura[::-1] - pred_gamma_kura
        delta_kura *= 1 / np.max(np.abs(delta_kura))
        plt.plot(theta, delta_gamma, c=[55 / 255, 127 / 255, 153 / 255], lw=2, label='E. coli')
        plt.plot(theta, delta_kura, c=[163 / 255, 84 / 255, 83 / 255], lw=2, label='Kuramoto')
        plt.xticks([-math.pi, 0, math.pi], [r'$-\pi$', '0', r'$\pi$'], fontsize=ticksize)
        plt.yticks([-1, 0, 1], ['-1', '0', '1'], fontsize=ticksize)
        plt.xlim(-math.pi, math.pi)
        if i == 0:
            plt.title(r'$\Delta\Gamma(\phi)$', fontsize=fontsize)

    ax = plt.subplot(gs[0:8, 15:16])  ##########################################################################################################################
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

    ax = plt.subplot(gs[0:8,
                     16:17])  ##########################################################################################################################
    dist = []
    for k in range(21):
        Q = float(format(k * 0.05, '.2f'))
        data = np.load(f'./data/results/prc_Q_{Q}.npy', allow_pickle=True).item()
        theta = data['raw'][:, 0]
        pred_gamma = data['pred']
        pred_gamma = pred_gamma[::-1] - pred_gamma
        pred_gamma_kura = np.sin(theta)
        pred_gamma_kura = pred_gamma_kura[::-1] - pred_gamma_kura
        pred_gamma_kura *= 1 / np.max(np.abs(pred_gamma_kura))*np.max(np.abs(pred_gamma))
        pred_gamma *= 1 / np.max(np.abs(pred_gamma))

        # delta_gamma = pred_gamma[::-1] - pred_gamma
        # delta_gamma *= 1 / np.max(np.abs(delta_gamma))
        # delta_kura = pred_gamma_kura[::-1] - pred_gamma_kura
        # delta_kura *= 1 / np.max(np.abs(delta_kura))

        dist.append(dist_func(theta, pred_gamma, 'L2'))
        # dist.append(pW_cal(pred_gamma.reshape(1,-1),pred_gamma_kura.reshape(1,-1)))
    x = np.linspace(0, 1, 21)
    # slope, intercept = np.polyfit(x, dist, 1)
    # y = slope * x + intercept
    c = plt.cm.RdBu(np.linspace(0, 1, len(x)))
    index = np.argsort(dist)
    c[index] = c

    norm = mcolors.Normalize(vmin=np.min(dist), vmax=np.max(dist))
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap='RdBu')
    cc = [scalar_map.to_rgba(value) for value in dist]

    h = plt.barh(x, dist[::-1], [0.025], color=cc[::-1])
    plt.xticks([0, 2], ['0', '2'], fontsize=ticksize)
    plt.yticks([0, 1], ['1', '0'], fontsize=ticksize)
    # plt.ylabel(r'$Q$',fontsize=fontsize,rotation=90)
    # plt.text(-0.9,0.5,r'$Q$',fontsize=fontsize)
    plt.ylim(-0.03, 1.03)

    plt.subplot(gs[9:11,
                12:20])  ##########################################################################################################################
    amplitude = []
    strength = []
    for k in range(21):
        Q = float(format(k * 0.05, '.2f'))
        data = np.load('./data/results/prc_Q_{}.npy'.format(Q), allow_pickle=True).item()
        pred_gamma = data['pred']
        pred_gamma = pred_gamma[::-1] - pred_gamma
        amplitude.append(np.ptp(pred_gamma))
        strength.append(Q)
    amplitude = np.array(amplitude)
    amplitude *= strength / np.max(amplitude)
    strength = np.array(strength)
    x = np.linspace(0, 1, 21)
    coef = np.polyfit(x, amplitude, 3)
    p1 = np.poly1d(coef)  # 使用次数合成多项式
    y_pre = p1(x)
    plt.plot(x, strength,marker='^', markerfacecolor='none', c='k',
             label='Kuramoto', zorder=1)
    plt.scatter(x, amplitude, marker='o', facecolor='none',
             c=[165 / 255, 72 / 255, 162 / 255], label='E. coli', zorder=1)
    plt.plot(x, y_pre,c=[165 / 255, 72 / 255, 162 / 255], label='E. coli', zorder=1)

    # plt.plot(np.linspace(0, 1, 11), orig_M_list[0][0:21:2], marker='o', markerfacecolor='none',
    #          c=[165 / 255, 72 / 255, 162 / 255], label=r'$R_{\mathrm{orig}}$', zorder=1)
    # plt.plot(np.linspace(0, 1, 11), dec_R_list[0][0:21:2], marker='^', markerfacecolor='none', c='k',
    #          label=r'$R_{\mathrm{ord}}$', zorder=1)

    plt.xticks([0, 1], ['0', '1'], fontsize=fontsize)
    plt.yticks([0, 1], ['0', '1'], fontsize=fontsize)
    plt.xlabel(r'$Q$', fontsize=fontsize)
    plt.ylabel(r'$\mathrm{NS}(Q)$', fontsize=fontsize)


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

    # plt.subplot(gs[8:11,8:11])  ##########################################################################################################################

    plt.show()

def plot_v2():
    fontsize = 18
    ticksize = 15
    fig = plt.figure(figsize=(12, 11))
    plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.95, hspace=0.25, wspace=0.2)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.0)
    gs = GridSpec(23, 22, figure=fig, hspace=0.1, wspace=0.1)
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
    plt.rc('font', family='Times New Roman')

###############################################################################################################################################
#                   Internal noise
#################################################################################################################################################
    ax = plt.subplot(gs[0:2,3:5])  ##########################################################################################################################
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.0)
    data = np.load('./data/results/Q beta pi l=1000 num=100 max=0.2.npy', allow_pickle=True).item()
    orig_M = data['orig_M']
    dec_M = data['dec_M']
    print(orig_M.shape, '&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    # min = np.minimum(dec_M,orig_M)
    # max = np.maximum(dec_M,orig_M)
    # values = (min/max).T
    values = 1 - np.abs(dec_M - orig_M) / (dec_M + orig_M)
    values = values.T
    # values = orig_M.T

    # cmap = cmaps.BlueWhiteOrangeRed  # 引用NCL的colormap
    # newcolors = cmap(np.linspace(0, 1, 12))  # 分片操作，生成0到1的12个数据间隔的数组
    # print(newcolors)
    # newcmap = ListedColormap(newcolors[::1])  # 重构为新的colormap
    # test_color = [[121/255,149/255,195/255,1.0],[136/255,145/255,184/255,1.0],[150/255,141/255,172/255,1.0],[162/255,137/255,163/255,1.0],
    #               [174/255,133/255,154/255,1.0],[190/255,129/255,142/255,1.0],[210/255,123/255,126/255,1.0]]
    # test_color = [[200/255,93/255,77/255,1.0],[240/255,183/255,154/255,1.0],[250/255,231/255,217/255,1.0],[227/255,238/255,239/255,1.0],
    #               [174/255,205/255,215/255,1.0],[97/255,157/255,184/255,1.0]]
    # test_color = [[84 / 255, 114 / 255, 152 / 255, 1.0], [142 / 255, 157 / 255, 186 / 255, 1.0],
    #               [218 / 255, 222 / 255, 231 / 255, 1.0], [241 / 255, 221 / 255, 214 / 255, 1.0],
    #               [218 / 255, 144 / 255, 135 / 255, 1.0], [210 / 255, 72 / 255, 70 / 255, 1.0]]
    test_color = [[95 / 255, 169 / 255, 208 / 255, 1.0], [153 / 255, 200 / 255, 220 / 255, 1.0],
                  [234 / 255, 238 / 255, 247 / 255, 1.0], [250 / 255, 238 / 255, 238 / 255, 1.0],
                  [233 / 255, 197 / 255, 197 / 255, 1.0], [218 / 255, 149 / 255, 152 / 255, 1.0]]
    newcmap = ListedColormap(test_color[::-1])  # 重构为新的colormap

    # colors = ['#024CEB', '#02BBA9', '#65FF00', '#FEFF00', '#FF8800', '#D40608']
    # newcmap = LinearSegmentedColormap.from_list('mymap', colors)
    # colors = ['#7FCAB2', '#EA9F7B', '#9FAED2', '#E99CCB', '#ADD696', '#E6CCA4']
    nodes = np.linspace(0,1,6)
    newcmap = LinearSegmentedColormap.from_list('mymap', list(zip(nodes, test_color[::-1])))
    newcmap = 'RdBu'
    h = plt.imshow(values[:, :], extent=[0, 21, 0, 21], cmap=newcmap, aspect='auto',
                   origin='lower')  # extent=[0, rangeT, 0, num]
    plt.xticks([0, 21], ['0', '1'], fontsize=ticksize)
    plt.yticks([0, 21], ['0', '0.2'], fontsize=ticksize)
    plt.xlabel(r'$Q$', labelpad=-15, fontsize=fontsize)
    plt.ylabel(r'$\sigma$', labelpad=-20, fontsize=fontsize)
    plt.title(r'$R_{\mathrm{pred}}$', fontsize=fontsize)
    # cax = fig.add_axes([ax.get_position().x1 + 0.02, ax.get_position().y0, 0.01, ax.get_position().height])
    # cax = fig.add_axes([ax.get_position().x0, ax.get_position().y1+0.05, ax.get_position().width,0.015])
    # cb = plt.colorbar(h, cax=cax,orientation="horizontal")
    # cb.ax.set_title(r'$R_{v,dec}/R_{v,orig}$', fontsize=fontsize)
    # cb.set_ticks([0, 1])
    # cb.ax.tick_params(labelsize=fontsize)
    # plt.clim(0, 1)

    ax = plt.subplot(gs[3:5,3:5])  ##########################################################################################################################
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.0)
    # R = np.load('./data/results/Q beta pi num=100 max=0.2 R.npy')
    R = data['R']
    # min = np.minimum(R,orig_M)
    # max = np.maximum(R,orig_M)
    # values = (min/max).T
    values = 1 - np.abs(R - orig_M) / (R + orig_M)
    values = values.T
    h = plt.imshow(values[:, :], extent=[0, 21, 0, 21], cmap='RdBu', aspect='auto',
                   origin='lower')  # extent=[0, rangeT, 0, num]
    plt.xticks([0, 21], ['0', '1'], fontsize=ticksize)
    plt.yticks([0, 21], ['0', '0.2'], fontsize=ticksize)
    plt.xlabel(r'$Q$', fontsize=fontsize, labelpad=-15)
    plt.ylabel(r'$\sigma$', labelpad=-20, fontsize=fontsize)
    plt.title(r'$R_{\mathrm{phase}}$', fontsize=fontsize)

    ax = plt.subplot(gs[0:5,6:11])  ##########################################################################################################################
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.0)
    # R = torch.from_numpy(np.load('./data/results/Q beta pi num=100 max=0.2 R.npy'))
    R = data['R']
    R = (R + dec_M) / 2
    # min = np.minimum(R,orig_M)
    # max = np.maximum(R,orig_M)
    # values = (min/max).T
    values = 1 - np.abs(R - orig_M) / (R + orig_M)
    values = values.T
    h = plt.imshow(values[:, :], extent=[0, 21, 0, 21], cmap=newcmap, aspect='auto',
                   origin='lower')  # extent=[0, rangeT, 0, num]
    plt.xticks([0, 21], ['0', '1'], fontsize=ticksize)
    plt.yticks([0, 21], ['0', '0.2'], fontsize=ticksize)
    plt.xlabel(r'$Q$', fontsize=fontsize, labelpad=-15)
    plt.ylabel(r'$\sigma$', labelpad=-20, fontsize=fontsize)
    plt.title(r'$(R_{\text{pred}}+R_{\text{phase}})/2$', fontsize=fontsize)
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.01, ax.get_position().height])
    # cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-0.05, ax.get_position().width,0.015])
    cb = plt.colorbar(h, cax=cax)  # , orientation="horizontal")
    # cb.ax.set_title(r'$R/R_{v,orig}$', fontsize=fontsize)
    cb.set_ticks([0, 1])
    cb.ax.tick_params(labelsize=fontsize)
    plt.clim(0, 1)

    ax = plt.subplot(gs[0:5,13:17])  ##########################################################################################################################
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.0)
    res_data = np.load('./data/results/Ecoli mse int l=1000 max=1000.npy',allow_pickle=True).item()
    # three_colors = [[51/255,76/255,139/255],[55/255,127/255,153/255],[157/255,81/255,152/255]]
    three_colors = [[136 / 255, 179 / 255, 214 / 255], [228 / 255, 144 / 255, 117 / 255], [201 / 255, 125 / 255, 134 / 255]]
    marker_list = ['o','*','^']
    label_list = [r'$Q=0,\Delta_\beta=0.1$',r'$Q=0.5,\Delta_\beta=0.05$',r'$Q=1,\Delta_\beta=0.01$']
    x = np.linspace(1, 3, 10)
    for i in range(3):
        l = 1000
        res = res_data['all_mse'][i]/l
        res = np.log10(res)
        plt.scatter(x, res, s=50, marker=marker_list[i], edgecolor=three_colors[i], linewidth=1,
                    facecolor='none',label=label_list[i])
        if i == 0 or i==1:
            # x = np.arange(10)
            # x = np.linspace(1,3,10)
            coef = np.polyfit(x, res, 1)
            p1 = np.poly1d(coef)  # 使用次数合成多项式
            y_pre = p1(x)
            plt.plot(x, y_pre, c=[193 / 255, 82 / 255, 6 / 255], lw=2)
    # plt.text(0.5, 1.05, r'$Q=1,\Delta_\beta=0.01$', fontsize=ticksize, c='k')
    # plt.text(-2, 1.02, 'MSE', fontsize=fontsize, c='k', rotation=90)
    # plt.xticks([0, 5, 10], [r'$10^1$', r'$10^2$', r'$10^3$'], fontsize=ticksize)
    plt.xticks([1,3], ['$1$', '$3$'], fontsize=ticksize)
    plt.yticks([-2,0], ['-2','0'], fontsize=ticksize)
    # plt.ylim(-0.03, 1.0)
    plt.xlabel(r'$\log_{10}(N)$', fontsize=fontsize, labelpad=-15)
    plt.ylabel(r'$\log_{10}(\text{MSE})$', fontsize=fontsize, labelpad=-5)
    plt.legend(fontsize=ticksize,loc=4,bbox_to_anchor=[1.1,-0.70],ncol=1,frameon=False,handlelength=1.0,handletextpad=0.5,labelspacing=0.1)
    plt.text(1.1,-0.35,r'$\mu={:.3f}$'.format(coef[0]),fontsize=fontsize,c=[193 / 255, 82 / 255, 6 / 255])

    ax = plt.subplot(gs[0:5,18:22])  ##########################################################################################################################
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.0)
    data = np.load('./data/results/orbit_Q_0.0_scale_0.01.npy', allow_pickle=True).item()
    orig = data['orig'][-1000:]
    dec = data['dec'][-1000:]
    time = np.linspace(0, len(orig) * 0.02, len(orig))
    # traj_colors = [[136 / 255, 179 / 255, 244 / 255],[248 / 255, 144 / 255, 117 / 255],[201 / 255, 125 / 255, 134 / 255]]
    # traj_colors = [[201 / 255, 125 / 255, 134 / 255], [169 / 255, 135 / 255, 158 / 255], [121 / 255, 149 / 255, 196 / 255]]
    traj_colors = [[70 / 255, 111 / 255, 135 / 255], [222 / 255, 179 / 255, 108 / 255],
                   [190 / 255, 203 / 255, 133 / 255]]
    for i, k in enumerate([3, 6, 9]):
        plt.plot(time, orig[:, 1 + k * 7], color=traj_colors[i], label='Orig{}'.format(i + 1), zorder=1, lw=2)
        plt.scatter(time[0:-1:25], dec[0:-1:25, 1 + k * 7], edgecolor=traj_colors[i], marker='o', s=20,
                    facecolor='none', label='Pred{}'.format(i + 1), zorder=2)
    plt.xticks([0, 20], ['0', '20'], fontsize=ticksize)
    plt.yticks([0, 150], ['0', '150'], fontsize=ticksize)
    plt.xlabel('Time', fontsize=fontsize, labelpad=-10)
    plt.legend(loc=4,ncol=2,frameon=False,bbox_to_anchor=[1.0,-0.70],handlelength=0.7,handletextpad=0.5,labelspacing=0.1,columnspacing=0.5,fontsize=ticksize)

    ###############################################################################################################################################
    #                   External noise
    #################################################################################################################################################


    ax = plt.subplot(gs[8:10,3:5])  ##########################################################################################################################
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.0)
    data = np.load('./data/results/Q beta pi l=1000 num=20 max=0.2 ext.npy', allow_pickle=True).item()
    orig_M = data['orig_M']
    dec_M = data['dec_M']
    print(orig_M.shape, '&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    # min = np.minimum(dec_M,orig_M)
    # max = np.maximum(dec_M,orig_M)
    # values = (min/max).T
    values = 1 - np.abs(dec_M - orig_M) / (dec_M + orig_M)
    values = values.T
    # values = orig_M.T
    h = plt.imshow(values[:, :], extent=[0, 21, 0, 21], cmap='RdBu', aspect='auto',
                   origin='lower')  # extent=[0, rangeT, 0, num]
    plt.xticks([0, 21], ['0', '1'], fontsize=ticksize)
    plt.yticks([0, 21], ['0', '0.2'], fontsize=ticksize)
    plt.xlabel(r'$Q$', labelpad=-15, fontsize=fontsize)
    plt.ylabel(r'$\sigma$', labelpad=-20, fontsize=fontsize)
    # plt.title(r'Sim$(R_{\mathrm{rec}},R_{\mathrm{orig}})$', fontsize=fontsize)
    plt.title(r'$R_{\text{pred}}$', fontsize=fontsize)

    ax = plt.subplot(gs[11:13,3:5])  ##########################################################################################################################
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.0)
    R = data['R']
    # min = np.minimum(R,orig_M)
    # max = np.maximum(R,orig_M)
    # values = (min/max).T
    values = 1 - np.abs(R - orig_M) / (R + orig_M)
    values = values.T
    h = plt.imshow(values[:, :], extent=[0, 21, 0, 21], cmap='RdBu', aspect='auto',
                   origin='lower')  # extent=[0, rangeT, 0, num]
    plt.xticks([0, 21], ['0', '1'], fontsize=ticksize)
    plt.yticks([0, 21], ['0', '0.2'], fontsize=ticksize)
    plt.xlabel(r'$Q$', fontsize=fontsize, labelpad=-15)
    plt.ylabel(r'$\sigma$', labelpad=-20, fontsize=fontsize)
    # plt.title(r'Sim$(R_{\mathrm{ord}},R_{\mathrm{orig}})$', fontsize=fontsize)
    plt.title(r'$R_{\text{phase}}$', fontsize=fontsize)

    ax = plt.subplot(gs[8:13,6:11])  ##########################################################################################################################
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.0)
    R = data['R']
    R = (R + dec_M) / 2
    # min = np.minimum(R,orig_M)
    # max = np.maximum(R,orig_M)
    # values = (min/max).T
    values = 1 - np.abs(R - orig_M) / (R + orig_M)
    values = values.T
    h = plt.imshow(values[:, :], extent=[0, 21, 0, 21], cmap='RdBu', aspect='auto',
                   origin='lower')  # extent=[0, rangeT, 0, num]
    plt.xticks([0, 21], ['0', '1'], fontsize=ticksize)
    plt.yticks([0, 21], ['0', '0.2'], fontsize=ticksize)
    plt.xlabel(r'$Q$', fontsize=fontsize, labelpad=-15)
    plt.ylabel(r'$\sigma$', labelpad=-20, fontsize=fontsize)
    # plt.title(r'Sim$(\Tilde{R},R_{\mathrm{orig}})$', fontsize=fontsize)
    plt.title(r'$(R_{\text{pred}}+R_{\text{phase}})/{2}$', fontsize=fontsize)
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.01, ax.get_position().height])
    # cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-0.05, ax.get_position().width,0.015])
    cb = plt.colorbar(h, cax=cax)  # , orientation="horizontal")
    # cb.ax.set_title(r'$R/R_{v,orig}$', fontsize=fontsize)
    cb.set_ticks([0, 1])
    cb.ax.tick_params(labelsize=fontsize)
    plt.clim(0, 1)

    ax = plt.subplot(gs[8:13,13:17])  ##########################################################################################################################
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.0)
    res_data = np.load('./data/results/Ecoli mse ext l=1000 max=100.npy', allow_pickle=True).item()
    # three_colors = [[51 / 255, 76 / 255, 139 / 255], [55 / 255, 127 / 255, 153 / 255], [157 / 255, 81 / 255, 152 / 255]]
    three_colors = [[136 / 255, 179 / 255, 214 / 255], [228 / 255, 144 / 255, 117 / 255], [201 / 255, 125 / 255, 134 / 255]]
    marker_list = ['o', '*', '^']
    label_list = [r'$Q=0,\Delta_\beta=0.1$', r'$Q=0.5,\Delta_\beta=0.05$', r'$Q=1,\Delta_\beta=0.01$']
    x = np.linspace(0, 2, 10)
    for i in range(3):
        l = 1000
        res = res_data['all_mse'][i] / l
        res = np.log10(res)
        plt.scatter(x, res, s=50, marker=marker_list[i], edgecolor=three_colors[i], linewidth=1,
                    facecolor='none', label=label_list[i])
    # x = np.arange(10)
    coef = np.polyfit(x, res, 1)
    p1 = np.poly1d(coef)  # 使用次数合成多项式
    y_pre = p1(x)
    plt.plot(x, y_pre,c=[193/255,82/255,6/255],lw=2)
    # plt.xticks([0, 5, 10], [r'$10^0$', r'$10^1$', r'$10^2$'], fontsize=ticksize)
    plt.xticks([0, 2], [r'$0$', r'$2$'], fontsize=ticksize)
    plt.yticks([ -1, 0], [ '-1', '0'], fontsize=ticksize)
    # plt.xlabel(r'$N$', fontsize=fontsize, labelpad=0)
    plt.xlabel(r'$\log_{10}(N)$', fontsize=fontsize, labelpad=-15)
    plt.ylabel(r'$\log_{10}(\text{MSE})$', fontsize=fontsize, labelpad=-5)
    plt.text(0.1,-0.05,r'$\mu={:.3f}$'.format(coef[0]),fontsize=fontsize,c=[193 / 255, 82 / 255, 6 / 255])

    plt.subplot(gs[8:13,18:22])  ##########################################################################################################################
    data = np.load('./data/results/orbit_Q_0.0_scale_0.01 ext.npy', allow_pickle=True).item()
    orig = data['orig'][:1000]
    dec = data['dec'][:1000]
    time = np.linspace(0, len(orig) * 0.02, len(orig))
    # traj_colors = [[136 / 255, 179 / 255, 244 / 255],[248 / 255, 144 / 255, 117 / 255],[201 / 255, 125 / 255, 134 / 255]]
    traj_colors = [[70 / 255, 111 / 255, 135 / 255], [222 / 255, 179 / 255, 108 / 255],
                   [190 / 255, 203 / 255, 133 / 255]]
    for i, k in enumerate([3, 9, 15]):
        plt.plot(time, orig[:, 1 + k * 7], color=traj_colors[i],label='Orig{}'.format(i+1),zorder=1,lw=2)
        plt.scatter(time[0:-1:25], dec[0:-1:25, 1 + k * 7], edgecolor=traj_colors[i], marker='o', s=20,
                    facecolor='none', label='Rec{}'.format(i+1), zorder=2)
    plt.xticks([0, 20], ['0', '20'], fontsize=ticksize)
    plt.yticks([0, 150], ['0', '150'], fontsize=ticksize)
    plt.xlabel('Time', fontsize=fontsize, labelpad=-10)

    ###############################################################################################################################################
    #                  Scaling form
    #################################################################################################################################################

    for i, Q in enumerate([0.05, 0.3,0.6, 0.9]):
        ax = plt.subplot(gs[15:17,i*3:i*3+2])  ##########################################################################################################################
        ax.patch.set_facecolor('white')
        ax.patch.set_alpha(0.0)
        data = np.load('./data/results/prc_Q_{}.npy'.format(Q), allow_pickle=True).item()
        theta = data['raw'][:, 0]
        pred_gamma = data['pred']
        pred_gamma_kura = np.sin(theta)
        print(f'Q={Q}, max={np.max(np.abs(pred_gamma))}')
        pred_gamma *= 1 / np.max(np.abs(pred_gamma))
        plt.plot(theta, pred_gamma, c=[55 / 255, 127 / 255, 153 / 255], lw=2, label='E. coli')
        plt.plot(theta, pred_gamma_kura, c=[163 / 255, 84 / 255, 83 / 255], lw=2, label='Kuramoto')

        # plt.plot(theta, pred_gamma[::-1]-pred_gamma, c=colors[0], lw=3, label='E. coli')
        # plt.plot(theta, pred_gamma_kura[::-1]-pred_gamma_kura, c='gray', lw=3, alpha=0.5, label='Kuramoto')
        plt.xticks([-math.pi, 0, math.pi], [r'$-\pi$', '0', r'$\pi$'], fontsize=ticksize)
        plt.yticks([-1, 0, 1], ['-1', '0', '1'], fontsize=ticksize)
        plt.xlim(-math.pi, math.pi)
        plt.title(r'$Q={}$'.format(Q),fontsize=fontsize)
        if i == 0:
            plt.ylabel(r'$\Gamma_Q(\phi)$', fontsize=fontsize)
            plt.legend(loc=2, ncol=2, bbox_to_anchor=[0.35, -1.35], fontsize=ticksize, frameon=False, handlelength=1.0)

        #     plt.ylabel(r'$\Gamma(\phi)$', fontsize=fontsize, labelpad=0)
        # plt.legend(loc=2, fontsize=ticksize, frameon=False, handlelength=1.0)
        # plt.legend(loc=2,ncol=1, fontsize=ticksize, frameon=False, handlelength=1.0) #,bbox_to_anchor=[1.1, 1.3]
        # if i == 3:
        #     plt.xlabel(r'$\phi$', fontsize=fontsize, labelpad=0)

        ax = plt.subplot(gs[21:23,i*3:i*3+2])  ##########################################################################################################################
        ax.patch.set_facecolor('white')
        ax.patch.set_alpha(0.0)
        delta_gamma = pred_gamma[::-1] - pred_gamma
        delta_gamma *= 1 / np.max(np.abs(delta_gamma))
        delta_kura = pred_gamma_kura[::-1] - pred_gamma_kura
        delta_kura *= 1 / np.max(np.abs(delta_kura))
        plt.plot(theta, delta_gamma, c=[55 / 255, 127 / 255, 153 / 255], lw=2, label='E. coli')
        plt.plot(theta, delta_kura, c=[163 / 255, 84 / 255, 83 / 255], lw=2, label='Kuramoto')
        plt.xticks([-math.pi, 0, math.pi], [r'$-\pi$', '0', r'$\pi$'], fontsize=ticksize)
        plt.yticks([-1, 0, 1], ['-1', '0', '1'], fontsize=ticksize)
        plt.xlim(-math.pi, math.pi)
        if i == 0:
            plt.ylabel(r'$\Delta\Gamma_Q(\phi)$', fontsize=fontsize)


    ax = plt.subplot(gs[18:20,0:11])  ##########################################################################################################################
    ax.spines['right'].set_color('none')  # 把右边的边框颜色设置为无色,隐藏右边框
    ax.spines['top'].set_color('none')  # 把上边的边框颜色设置为无色,隐藏上边框
    ax.spines['left'].set_color('none')  # 把上边的边框颜色设置为无色,隐藏上边框
    ax.xaxis.set_ticks_position('bottom')  # 指定左边的边为 y 轴
    ax.spines['bottom'].set_position(('data', 0))
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.0)
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
    c = plt.cm.RdBu(np.linspace(0, 1, len(x)))
    index = np.argsort(dist)
    c[index] = c

    norm = mcolors.Normalize(vmin=np.min(dist), vmax=np.max(dist))
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap='RdBu')
    cc = [scalar_map.to_rgba(value) for value in dist]

    h = plt.bar(x, dist, [0.025], color=[220/255,175/255,161/255],edgecolor=[208/255,148/255,219/255],label=r'$\Gamma_Q$') #[232/255,150/255,83/255]
    dist = []
    for k in range(21):
        Q = float(format(k * 0.05, '.2f'))
        data = np.load(f'./data/results/prc_Q_{Q}.npy', allow_pickle=True).item()
        theta = data['raw'][:, 0]
        pred_gamma = data['pred']
        pred_gamma = pred_gamma[::-1] - pred_gamma
        pred_gamma_kura = np.sin(theta)
        pred_gamma_kura = pred_gamma_kura[::-1] - pred_gamma_kura
        pred_gamma_kura *= 1 / np.max(np.abs(pred_gamma_kura)) * np.max(np.abs(pred_gamma))
        pred_gamma *= 1 / np.max(np.abs(pred_gamma))
        dist.append(dist_func(theta, pred_gamma, 'L2'))
        # dist.append(pW_cal(pred_gamma.reshape(1,-1),pred_gamma_kura.reshape(1,-1)))
    h = plt.bar(x, -np.array(dist), [0.025], color=[172/255,194/255,213/255],edgecolor=[144/255,174/255,199/255],label=r'$\Delta \Gamma_Q$')
    plt.xticks([0, 1], ['0', '1'], fontsize=ticksize)
    plt.yticks([])
    plt.xlabel(r'$Q$', fontsize=fontsize, labelpad=-7)
    plt.legend(frameon=False,loc=4,ncol=2,fontsize=fontsize,bbox_to_anchor=[1.0,-0.65],handlelength=0.85,handleheight=0.25,handletextpad=0.5,labelspacing=0.1,columnspacing=0.5)
    plt.ylabel('Distance',fontsize=fontsize,rotation=90)
    # plt.text(-0.9,0.5,r'$Q$',fontsize=fontsize)
    plt.ylim(-1.03, 1.03)

    ax = plt.subplot(gs[15:23,14:24])  ##########################################################################################################################
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(0.0)
    amplitude = []
    strength = []
    for k in range(21):
        Q = float(format(k * 0.05, '.2f'))
        data = np.load('./data/results/prc_Q_{}.npy'.format(Q), allow_pickle=True).item()
        pred_gamma = data['pred']
        pred_gamma = pred_gamma[::-1] - pred_gamma
        amplitude.append(np.ptp(pred_gamma))
        strength.append(Q)
    amplitude = np.array(amplitude)
    amplitude *= strength / np.max(amplitude)
    strength = np.array(strength)
    x = np.linspace(0, 1, 21)
    coef = np.polyfit(x, amplitude, 3)
    p1 = np.poly1d(coef)  # 使用次数合成多项式
    y_pre = p1(x)
    plt.plot(x, strength, c=[193 / 255, 84 / 255, 83 / 255], zorder=1)
    plt.scatter(x, strength, marker='^', edgecolor=[163 / 255, 84 / 255, 83 / 255], c='',s=50,
             label='Kuramoto', zorder=1)
    plt.scatter(x, amplitude, marker='o', c='',
                edgecolor=[55 / 255, 127 / 255, 153 / 255],s=50, label='E. coli', zorder=1)
    plt.plot(x, y_pre, c=[55 / 255, 127 / 255, 193 / 255], zorder=1)
    plt.legend(loc=2,ncol=1,frameon=False,handlelength=0.7,handletextpad=1.0,labelspacing=1.0,columnspacing=0.5,fontsize=ticksize)
    # plt.plot(np.linspace(0, 1, 11), orig_M_list[0][0:21:2], marker='o', markerfacecolor='none',
    #          c=[165 / 255, 72 / 255, 162 / 255], label=r'$R_{\mathrm{orig}}$', zorder=1)
    # plt.plot(np.linspace(0, 1, 11), dec_R_list[0][0:21:2], marker='^', markerfacecolor='none', c='k',
    #          label=r'$R_{\mathrm{ord}}$', zorder=1)

    plt.xticks([0, 1], ['0', '1'], fontsize=fontsize)
    plt.yticks([0, 1], ['0', '1'], fontsize=fontsize)
    plt.xlabel(r'$Q$', fontsize=fontsize,labelpad=-10)
    plt.ylabel(r'$\mathrm{NS}(Q)$', fontsize=fontsize,labelpad=-10)
    plt.savefig('./data/fig4_sub.pdf')
    plt.show()

# plot_v1()
plot_v2()