import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange
from functions import *

class Cdk(nn.Module):

    def __init__(self):
        super(Cdk, self).__init__()
        self.GF = 1.0
        self.K_agf = 0.1
        self.k_dap1 = 0.15
        self.eps = 17.0
        self.v_sap1 = 1.0
        #################################### Antagonistic regulation exerted by pRB and E2F
        self.k_de2f = 0.002
        self.k_de2fp = 1.1
        self.k_dprb = 0.01
        self.k_dprbp = 0.06
        self.k_dprbpp = 0.04
        self.k_pc1 = 0.05
        self.k_pc2 = 0.5
        self.k_pc3 = 0.025
        self.k_pc4 = 0.5
        self.K_1 = 0.1
        self.K_2 = 0.1
        self.K_3 = 0.1
        self.K_4 = 0.1
        self.V_1 = 2.2
        self.V_2 = 2.0
        self.V_3 = 1.0
        self.V_4 = 2.0
        self.K_1e2f = 5.0
        self.K_2e2f = 5.0
        self.V_1e2f = 4.0
        self.V_2e2f = 0.75
        self.v_se2f = 0.15
        self.v_sprb = 0.8
        #################################### Module Cyclin D/Cdk4-6: G1 phase
        self.Cdk4_tot = 1.5
        self.K_i7 = 0.1
        self.K_i8 = 2.0
        self.k_cd1 = 0.4
        self.k_cd2 = 0.005
        self.k_decom1 = 0.1
        self.k_com1 = 0.175
        self.k_c1 = 0.15
        self.k_c2 = 0.05
        self.k_ddd = 0.005
        self.K_dd = 0.1
        self.K_1d = 0.1
        self.K_2d = 0.1
        self.V_dd = 5.0
        self.V_m1d = 1.0
        self.V_m2d = 0.2
        #################################### Module Cyclin E/Cdk2: G1 phase and transition G1/S
        self.a_e = 0.25
        self.Cdk2_tot = 2.0
        self.i_b1 = 0.5
        self.K_i9 = 0.1
        self.K_i10 = 2.0
        self.k_ce = 0.29
        self.k_c3 = 0.2
        self.k_c4 = 0.1
        self.k_decom2 = 0.1
        self.k_com2 = 0.2
        self.k_dde = 0.005
        self.k_ddskp2 = 0.005
        self.k_dpe = 0.075
        self.k_dpei = 0.15
        self.K_de = 0.1
        self.K_dceskp2 = 2.0
        self.K_dskp2 = 0.5
        self.K_cdh1 = 0.4
        self.K_1e = 0.1
        self.K_2e = 0.1
        self.K_5e = 0.1
        self.K_6e = 0.1
        self.V_de = 3.0
        self.V_dskp2 = 1.1
        self.V_m1e = 2.0
        self.V_m2e = 1.4
        self.V_m5e = 5.0
        self.V_6e = 0.8
        self.v_spei = 0.13
        self.v_sskp2 = 0.15
        self.x_e1 = 1.0
        self.x_e2 = 1.0
        ################################## Module Cyclin A/Cdk2: S phase and transition S/G2
        self.a_a = 0.2
        self.i_b2 = 0.5
        self.K_i11 = 0.1
        self.K_i12 = 2.0
        self.K_i13 = 0.1
        self.K_i14 = 2.0
        self.k_ca = 0.0375
        self.k_decom3 = 0.1
        self.k_com3 = 0.2
        self.k_c5 = 0.15
        self.k_c6 = 0.125
        self.k_dda = 0.005
        self.k_ddp27 = 0.06
        self.k_ddp27p = 0.01
        self.k_dcdh1a = 0.1
        self.k_dcdh1i = 0.2
        self.k_dpa = 0.075
        self.k_dpai = 0.15
        self.K_da = 1.1
        self.K_dp27p = 0.1
        self.K_dp27skp2 = 0.1
        self.K_acdc20 = 2.0
        self.K_1a = 0.1
        self.K_2a = 0.1
        self.K_1cdh1 = 0.01
        self.K_2cdh1 = 0.01
        self.K_5a = 0.1
        self.K_6a = 0.1
        self.K_1p27 = 0.5
        self.K_2p27 = 0.5
        self.V_dp27p = 5.0
        self.V_da = 2.5
        self.V_m1a = 2.0
        self.V_m2a = 1.85
        self.V_m5a = 4.0
        self.V_6a = 1.0
        self.v_scdh1a = 0.11
        self.v_spai = 0.105
        self.v_s1p27 = 0.8
        self.v_s2p27 = 0.1
        self.V_1cdh1 = 1.25
        self.V_2cdh1 = 8.0
        self.V_1p27 = 100.0 #100.0
        self.V_2p27 = 0.1
        self.x_a1 = 1.0
        self.x_a2 = 1.0
        ################################## Module Cyclin B/Cdk1: G2 phase and transition G2/M
        self.a_b = 0.11
        self.Cdk1_tot = 0.5
        self.i_b = 0.75
        self.i_b3 = 0.5
        self.k_c7 = 0.12
        self.k_c8 = 0.2
        self.k_decom4 = 0.1
        self.k_com4 = 0.25
        self.k_dcdc20a = 0.05
        self.k_dcdc20i = 0.14
        self.k_ddb = 0.005
        self.k_dpb = 0.1
        self.k_dpbi = 0.2
        self.k_dwee1 = 0.1
        self.k_dwee1p = 0.2
        self.K_db = 0.005
        self.K_dbcdc20 = 0.2
        self.K_dbcdh1 = 0.1
        self.k_sw = 5.0
        self.K_1b = 0.1
        self.K_2b = 0.1
        self.K_3b = 0.1
        self.K_4b = 0.1
        self.K_5b = 0.1
        self.K_6b = 0.1
        self.K_7b = 0.1
        self.K_8b = 0.1
        self.v_cb = 0.05
        self.V_db = 0.06
        self.V_m1b = 3.9
        self.V_m2b = 2.1
        self.v_scdc20i = 0.1
        self.V_m3b = 8.0
        self.V_m4b = 0.7
        self.V_m5b = 5.0
        self.V_6b = 1.0
        self.V_m7b = 1.2
        self.V_m8b = 1.0
        self.v_spbi = 0.12
        self.v_swee1 = 0.06
        self.x_b1 = 1.0
        self.x_b2 = 1.0
        # Additional parameters
        self.ATR_tot = 0.5
        self.Chk1_tot = 0.5
        self.Cdc45_tot = 0.5
        self.k_aatr = 0.022
        self.k_datr = 0.15
        self.k_dpol = 0.2
        self.k_dprim = 0.15
        self.k_spol = 0.8
        self.k_sprim = 0.05
        self.K_1cdc45 = 0.02
        self.K_2cdc45 = 0.02
        self.K_1chk = 0.5
        self.K_2chk = 0.5
        self.Pol_tot = 0.5
        self.V_1cdc45 = 0.8
        self.V_2cdc45 = 0.12
        self.V_1chk = 4.0
        self.V_2chk = 0.1
        self.K_dw = 0.5
        self.K_iw = 1.0
        self.n = 4.0
        self.v_dw = 0.5
        self.v_sw = 0.0



    def forward(self, t, state):
        dstate = torch.zeros_like(state)
        L = state.shape[1] # numbers of variables
        dim = 39
        AP1 =  state[:,0:L:dim]
        pRB,pRBc1,pRBp,pRBc2,pRBpp,E2F,E2Fp = state[:,1:L:dim],state[:,2:L:dim],state[:,3:L:dim],state[:,4:L:dim],state[:,5:L:dim],state[:,6:L:dim],state[:,7:L:dim]
        Cd,Mdi,Md,Mdp27= state[:,8:L:dim],state[:,9:L:dim],state[:,10:L:dim],state[:,11:L:dim]
        Ce,Mei,Me,Skp2,Mep27,Pei,Pe= state[:,12:L:dim],state[:,13:L:dim],state[:,14:L:dim],state[:,15:L:dim],state[:,16:L:dim],state[:,17:L:dim],state[:,18:L:dim]
        Ca,Mai,Ma,Map27,p27,p27p,Cdh1i,Cdh1a,Pai,Pa = state[:,19:L:dim],state[:,20:L:dim],state[:,21:L:dim],state[:,22:L:dim],state[:,23:L:dim],state[:,24:L:dim]\
        , state[:, 25:L:dim],state[:,26:L:dim],state[:,27:L:dim],state[:,28:L:dim]
        Cb,Mbi,Mb,Mbp27,Cdc20i,Cdc20a,Pbi,Pb,Wee1,Wee1p = state[:,29:L:dim],state[:,30:L:dim],state[:,31:L:dim],state[:,32:L:dim],state[:,33:L:dim],state[:,34:L:dim] \
            , state[:, 35:L:dim],state[:,36:L:dim],state[:,37:L:dim],state[:,38:L:dim]
        # Cdc45,Pol,Primer,ATR,Chk1 = state[:,39:L:dim],state[:,40:L:dim],state[:,41:L:dim],state[:,42:L:dim],state[:,43:L:dim]
        # Mw = state[:,44:L:dim]
        Mw,Chk1 = 0.01,0.00
        # Mitotic stimulation by growth factor, GF
        dstate[:,0:L:dim] = self.v_sap1*self.GF/(self.K_agf+self.GF)-self.k_dap1*AP1
        # Antagonistic regulation exerted by pRB and E2F
        dstate[:,1:L:dim] = self.v_sprb-self.k_pc1*pRB*E2F+self.k_pc2*pRBc1-self.V_1*pRB/(self.K_1+pRB)*(Md+Mdp27)+self.V_2*pRBp/(self.K_2+pRBp)-self.k_dprb*pRB
        dstate[:,2:L:dim] = self.k_pc1*pRB*E2F-self.k_pc2*pRBc1
        dstate[:, 3:L:dim] = self.V_1*pRB/(self.K_1+pRB)*(Md+Mdp27)-self.V_2*pRBp/(self.K_2+pRBp)-self.V_3*pRBp*(self.K_3+pRBp)*Me+self.V_4*pRBpp/(self.K_4+pRBpp)-self.k_pc3*pRBp*E2F+self.k_pc4*pRBc2-self.k_dprbp*pRBp
        dstate[:, 4:L:dim] = self.k_pc3*pRBp*E2F-self.k_pc4*pRBc2
        dstate[:, 5:L:dim] = self.V_3*pRBp/(self.K_3+pRBp)*Me-self.V_4*pRBpp/(self.K_4+pRBpp)-self.k_dprbpp*pRBpp
        dstate[:, 6:L:dim] = self.v_se2f-self.k_pc1*pRB*E2F+self.k_pc2*pRBc1-self.k_pc3*pRBp*E2F+self.k_pc4*pRBc2-self.V_1e2f*Ma*E2F/(self.K_1e2f+E2F)+self.V_2e2f*E2Fp/(self.K_2e2f+E2Fp)-self.k_de2f*E2F
        dstate[:, 7:L:dim] = self.V_1e2f*Ma*E2F/(self.K_1e2f+E2F)-self.V_2e2f*E2Fp/(self.K_2e2f+E2Fp)-self.k_de2fp*E2Fp
        # Module cyclin D/Cdk4-6: G1 phase
        dstate[:, 8:L:dim] = self.k_cd1*AP1+self.k_cd2*E2F*self.K_i7/(self.K_i7+pRB)*self.K_i8/(self.K_i8+pRBp)-self.k_com1*Cd*(self.Cdk4_tot-Mdi-Md-Mdp27)+self.k_decom1*Mdi-self.V_dd*Cd/(self.K_dd+Cd)-self.k_ddd*Cd
        dstate[:, 9:L:dim] = self.k_com1*Cd*(self.Cdk4_tot-Md-Mdi-Mdp27)-self.k_decom1*Mdi+self.V_m2d*Md/(self.K_2d+Md)-self.V_m1d*Mdi/(self.K_1d+Mdi)
        dstate[:, 10:L:dim] = self.V_m1d*Mdi/(self.K_1d+Mdi)-self.V_m2d*Md/(self.K_2d+Md)-self.k_c1*Md*p27+self.k_c2*Mdp27
        dstate[:, 11:L:dim] = self.k_c1*Md*p27-self.k_c2*Mdp27
        # Module cyclin E/Cdk2: G1 phase and G1/S transition
        dstate[:, 12:L:dim] = self.k_ce*E2F*self.K_i9/(self.K_i9+pRB)*self.K_i10/(self.K_i10+pRBp)-self.k_com2*Ce*(self.Cdk2_tot-Mei-Me-Mep27-Mai-Ma-Map27)+self.k_decom2*Mei-self.V_de*Skp2/(self.K_dceskp2+Skp2)*Ce/(self.K_de+Ce)-self.k_dde*Ce
        dstate[:, 13:L:dim] = self.k_com2*Ce*(self.Cdk2_tot-Mei-Me-Mep27-Mai-Ma-Map27)-self.k_decom2*Mei+self.V_m2e*(Wee1+self.i_b1)*Me/(self.K_2e+Me)-self.V_m1e*Pe*Mei/(self.K_1e+Mei)
        dstate[:, 14:L:dim] = self.V_m1e*Pe*Mei/(self.K_1e+Mei)-self.V_m2e*(Wee1+self.i_b1)*Me/(self.K_2e+Me)-self.k_c3*Me*p27+self.k_c4*Mep27
        dstate[:, 15:L:dim] = self.v_sskp2-self.V_dskp2*Skp2/(self.K_dskp2+Skp2)*(Cdh1a)/(self.K_cdh1+Cdh1a)-self.k_ddskp2*Skp2
        dstate[:, 16:L:dim] = self.k_c3*Me*p27-self.k_c4*Mep27
        dstate[:, 17:L:dim] = self.v_spei+self.V_6e*(self.x_e1+self.x_e2*Chk1)*Pe/(self.K_6e+Pe)-self.V_m5e*(Me+self.a_e)*Pei/(self.K_5e+Pei)-self.k_dpei*Pei
        dstate[:, 18:L:dim] = self.V_m5e*(Me+self.a_e)*Pei/(self.K_5e+Pei)-self.V_6e*(self.x_e1+self.x_e2*Chk1)*Pe/(self.K_6e+Pe)-self.k_dpe*Pe
        # Module cyclin A/Cdk2: S phase and S/G2 transition
        dstate[:, 19:L:dim] = self.k_ca*E2F*self.K_i11/(self.K_i11+pRB)*self.K_i12/(self.K_i12+pRBp)-self.k_com3*Ca*(self.Cdk2_tot-Mei-Me-Mep27-Mai-Ma-Map27)+self.k_decom3*Mai-self.V_da*Ca/(self.K_da+Ca)*Cdc20a/(self.K_acdc20+Cdc20a)-self.k_dda*Ca
        dstate[:, 20:L:dim] = self.k_com3*Ca*(self.Cdk2_tot-Mei-Me-Mep27-Mai-Ma-Map27)-self.k_decom3*Mai+self.V_m2a*(Wee1+self.i_b2)*Ma/(self.K_2a+Ma)-self.V_m1a*Pa*Mai/(self.K_1a+Mai)
        dstate[:, 21:L:dim] = self.V_m1a*Pa*Mai/(self.K_1a+Mai)-self.V_m2a*(Wee1+self.i_b2)*Ma/(self.K_2a+Ma)-self.k_c5*Ma*p27+self.k_c6*Map27
        dstate[:, 22:L:dim] = self.k_c5*Ma*p27-self.k_c6*Map27
        dstate[:, 23:L:dim] = self.v_s1p27+self.v_s2p27*E2F*self.K_i13/(self.K_i13+pRB)*self.K_i14/(self.K_i14+pRBp)-self.k_c1*Md*p27+self.k_c2*Mdp27-self.k_c3*Me*p27+self.k_c4*Mep27-self.k_c5*Ma*p27+self.k_c6*Map27-self.k_c7*Mb*p27+self.k_c8*Mbp27-self.V_1p27*Me*p27/(self.K_1p27+p27)+self.V_2p27*p27p/(self.K_2p27+p27p)-self.k_ddp27*p27
        dstate[:, 24:L:dim] = self.V_1p27*Me*p27/(self.K_1p27+p27)-self.V_2p27*p27p/(self.K_2p27+p27p)-self.V_dp27p*Skp2/(self.K_dp27skp2+Skp2)*p27p/(self.K_dp27p+p27p)-self.k_ddp27p*p27p
        dstate[:, 25:L:dim] = self.V_2cdh1*Cdh1a/(self.K_2cdh1+Cdh1a)*(Ma+Mb)-self.V_1cdh1*Cdh1i/(self.K_1cdh1+Cdh1i)-self.k_dcdh1i*Cdh1i
        dstate[:, 26:L:dim] = self.v_scdh1a+self.V_1cdh1*Cdh1i/(self.K_1cdh1+Cdh1i)-self.V_2cdh1*Cdh1a/(self.K_2cdh1+Cdh1a)*(Ma+Mb)-self.k_dcdh1a*Cdh1a
        dstate[:, 27:L:dim] = self.v_spai+self.V_6a*(self.x_a1+self.x_a2*Chk1)*Pa/(self.K_6a+Pa)-self.V_m5a*(Ma+self.a_a)*Pai/(self.K_5a+Pai)-self.k_dpai*Pai
        dstate[:, 28:L:dim] = self.V_m5a*(Ma+self.a_a)*Pai/(self.K_5a+Pai)-self.V_6a*(self.x_a1+self.x_a2*Chk1)*Pa/(self.K_6a+Pa)-self.k_dpa*Pa
        # Module cyclin B/Cdk1: G2 phase and G2/M transition
        dstate[:, 29:L:dim] = self.v_cb-self.k_com4*Cb*(self.Cdk1_tot-Mbi-Mb-Mbp27)+self.k_decom4*Mbi-self.V_db*Cb/(self.K_db+Cb)*(Cdc20a/(self.K_dbcdc20+Cdc20a)+Cdh1a/(self.K_dbcdh1+Cdh1a)-self.k_ddb*Cb)
        dstate[:, 30:L:dim] = self.k_com4*Cb*(self.Cdk1_tot-Mbi-Mb-Mbp27)-self.k_decom4*Mbi+self.V_m2b*(Wee1+self.i_b3)*Mb/(self.K_2b+Mb)-self.V_m1b*Pb*Mbi/(self.K_1b+Mbi)
        dstate[:, 31:L:dim] = self.V_m1b*Pb*Mbi/(self.K_1b+Mbi)-self.V_m2b*(Wee1+self.i_b3)*Mb/(self.K_2b+Mb)-self.k_c7*Mb*p27+self.k_c8*Mbp27
        dstate[:, 32:L:dim] = self.k_c7*Mb*p27-self.k_c8*Mbp27
        dstate[:, 33:L:dim] = self.v_scdc20i-self.V_m3b*Mb*Cdc20i/(self.K_3b+Cdc20i)+self.V_m4b*Cdc20a/(self.K_4b+Cdc20a)-self.k_dcdc20i*Cdc20i
        dstate[:, 34:L:dim] = self.V_m3b*Mb*Cdc20i/(self.K_3b+Cdc20i)-self.V_m4b*Cdc20a/(self.K_4b+Cdc20a)-self.k_dcdc20a*Cdc20a
        dstate[:, 35:L:dim] = self.v_spbi+self.V_6b*(self.x_b1+self.x_b2*Chk1)*Pb/(self.K_6b+Pb)-self.V_m5b*(Mb+self.a_b)*Pbi/(self.K_5b+Pbi)-self.k_dpbi*Pbi
        dstate[:, 36:L:dim] = self.V_m5b*(Mb+self.a_b)*Pbi/(self.K_5b+Pbi)-self.V_6b*(self.x_b1+self.x_b2*Chk1)*Pb/(self.K_6b+Pb)-self.k_dpb*Pb
        dstate[:, 37:L:dim] = self.v_swee1+self.k_sw*Mw-self.V_m7b*(Mb+self.i_b)*Wee1/(self.K_7b+Wee1)+self.V_m8b*Wee1p/(self.K_8b+Wee1p)-self.k_dwee1*Wee1
        dstate[:, 38:L:dim] = self.V_m7b*(Mb+self.i_b)*Wee1/(self.K_7b+Wee1)-self.V_m8b*Wee1p/(self.K_8b+Wee1p)-self.k_dwee1p*Wee1p
        # Additional variables
        # dstate[:, 39:L:dim] = self.V_1cdc45*Me*(self.Cdc45_tot-Cdc45)/(self.K_1cdc45+self.Cdc45_tot-Cdc45)-self.V_2cdc45*Cdc45/(self.K_2cdc45+Cdc45)-self.k_spol*(self.Pol_tot-Pol)*Cdc45+self.k_dpol*Pol
        # dstate[:, 40:L:dim] = self.k_spol*(self.Pol_tot-Pol)*Cdc45-self.k_dpol*Pol
        # dstate[:, 41:L:dim] = self.k_sprim*Pol-self.k_dprim*Primer-self.k_aatr*(self.ATR_tot-ATR)*Primer+self.k_datr*ATR
        # dstate[:, 42:L:dim] = self.k_aatr*(self.ATR_tot-ATR)*Primer-self.k_datr*ATR
        # dstate[:, 43:L:dim] = self.V_1chk*ATR*(self.Chk1_tot-Chk1)/(self.K_1chk+self.Chk1_tot-Chk1)-self.V_2chk*Chk1/(self.K_2chk+Chk1)
        # dstate[:, 44:L:dim] = -self.v_dw*Mw/(self.K_dw+Mw)
        return dstate * self.eps


def plot():
    import scipy.io
    dim = 44
    data = scipy.io.loadmat('./data/cell cycle.mat')
    X = data['y_deter']
    vector = data['vector']
    t = data['t0']
    print(X.shape,vector.shape)
    result_args = np.argpartition(X[:, 3], -10)[-10:]
    index_list = []
    for i in range(10):
        for j in range(i, 10):
            res = np.abs(result_args[i] - result_args[j])
            if res > 300 and res < 800:
                index_list.append(res)
    print(result_args, '\n', int(np.mean(index_list)))
    L = int(np.mean(index_list))

    torch.save(torch.from_numpy(X[-L:]),'./data/train_Cdk.pt')
    torch.save(torch.from_numpy(vector[-L:]), './data/train_Cdk_func.pt')
    # print(vector.shape,X[-L:].shape)
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.plot(np.arange(len(X)), X[:, i])
        # plt.plot(np.arange(len(X)), X[:, 3])
        # plt.plot(np.arange(len(X)), X[:, 8])
        # plt.plot(X[-L:,3],X[-L:,7])



    plt.show()

plot()
