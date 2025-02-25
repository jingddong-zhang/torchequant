import logging



from functions import *
'''
loading data
'''
# data = torch.load('./data/train_ECC.pt')[80:]


'''
For learning 
'''
D_in =  7 # input dimension
H1 = 20 * D_in  # hidden dimension
D_latent = 2 # latent dimension
edim = 7 # dimension of equant
out_iters = 0

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_filename = 'train_log.txt'
file_handler = logging.FileHandler(log_filename)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

while out_iters < 1:
    # break
    start = timeit.default_timer()
    max_iters = 10000
    learning_rate = 0.01
    for k in range(21):
        Q = float(format(k * 0.05, '.2f'))
        data = torch.load('./data/orbit/train_ECC_Q_{}.pt'.format(Q))
        data = data.requires_grad_(True)
        estimate_w = 2*math.pi/(0.02*len(data))
        model = AE(D_in, H1, D_latent,w=estimate_w, edim=7)
        Loss = []
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        f = ECC_varyQ(data,Q)
        # latent, phi, dphi, reverse = model.forward(data)
        i = 0
        while i < max_iters:
            # break
            latent,phi,dphi,reverse = model.forward(data)
            ptp = torch.max(phi) - torch.min(phi)
            inner = torch.sum(dphi*f,dim=1)
            loss_AE = torch.mean((reverse - data) ** 2)
            loss_phase = torch.mean((inner - model.w) ** 2)
            loss = loss_AE + loss_phase + (math.pi * 2 - ptp) ** 2
                    # + (math.pi * 2 - ptp) ** 2 + torch.relu(1.0 - model.w ** 2)
            if i%100 == 0:
                print(Q, i, "loss=", loss.item(),loss_phase.item(),model.w.item(),ptp.item())
                logger.info(f"Phase, Q={Q}, i = {i}, loss={loss.item():.4f}, loss_phase={loss_phase.item():.4f},w={model.w.item():.4f},ptp={ptp.item():.4f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss.append(loss.item())
            if loss.item() == min(Loss):
                best_model = model.state_dict()
            if loss<0.08 and loss_phase<=0.009:
                print(Q, i, "loss=", loss.item(),loss_phase.item(),model.w.item(),ptp.item())
                logger.info(f"Phase break, Q={Q}, i = {i}, loss={loss.item():.4f}, loss_phase={loss_phase.item():.4f},w={model.w.item():.4f},ptp={ptp.item():.4f}")
                break
            i += 1
        # torch.save(model.state_dict(), './data/params/model_ECC_Q_{}.pkl'.format(Q))

        model.load_state_dict(best_model)
        max_iters = 3000
        learning_rate = 0.001
        optimizer = torch.optim.Adam(model.surface.parameters(),
                                     lr=learning_rate)
        Loss = []
        i = 0
        # aug_data = torch.cat((data,torch.zeros([len(data),1])),dim=1)
        aug_data = data # without augmentation

        while i < max_iters:
            # break
            zero = torch.zeros([1,edim])  # reference of disk
            pred_y,_,_ = model.surface(aug_data)
            equant = model.surface.inverse(zero)
            line_reference_data = aug_data-equant
            data_angle = angle_3d(line_reference_data)
            loss_angle = data_angle.std()
            # ptp = torch.max(data_angle)-torch.min(data_angle)
            cir = model.enc(data)
            cir = cir / torch.norm(cir, p=2, dim=1).view(-1, 1)  # Constrain the latent state on the circle by construction
            aug_cir = augment(cir)
            loss_bound = torch.mean((pred_y - aug_cir) ** 2)
            loss = loss_bound + .1*loss_angle
            if i%100 == 0:
                print(Q,i, "loss=", loss.item(), loss_bound.item(),loss_angle.item())
                logger.info(f"Equant, Q={Q}, i = {i}, loss={loss.item():.4f}, loss_bound={loss_bound.item():.4f},loss_angle={loss_angle.item():.4f}")
            Loss.append(loss.item())
            if loss.item() == min(Loss):
                best_equant = equant[0] #model.reference_point
                best_model = model.state_dict()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss <= 1e-5:
                logger.info(f"Equant break, Q={Q}, i = {i}, loss={loss.item():.4f}, loss_bound={loss_bound.item():.4f},loss_angle={loss_angle.item():.4f}")
                break
            i += 1
        print(f'best_equant:{best_equant}')
        # model.load_state_dict(best_model)
        # best_equant = model.surface.inverse(zero)[0]
        torch.save(best_model, './data/params/model_ECC_Q_{}.pkl'.format(Q))
        stop = timeit.default_timer()
        print('\n')
        print("Total time: ", stop - start)



    out_iters += 1

file_handler.close()

'''
9 realnvps: loss= 0.006912082497405244 0.030506761058374483
9 realnvps: loss= 0.016504904375797064 0.01334908846697062
7 realnvps: loss= 0.008428481489430243 0.023727878733632322
5 realnvps: loss= 0.011117235659182245 0.024764629551937208
4 realnvps: loss= 0.03880689595853918 0.020492370677357565
'''