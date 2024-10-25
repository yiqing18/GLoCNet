import os
os.environ["CUDA_VISIBLE_DEVICES"] =''
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler
import ReM_PointNet_DGCNNall as RM
from ReM_Pointdataset import train_data_preprocess,test_data_preprocess
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio

config = {
    'embed_dim': 256,
    'keypoint_encoder': [32, 64, 128,256],
    'layer_names': ['cross','self'] * 4,
    'output_dim':2,
    'k':18,
}
## path
train_img_path = './EXP/train'
test_data_path = './EXP/test'

## net
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net1 = RM.DGCNN(config).to(device)
net1.load_state_dict(torch.load('./model/ReM_pointu.pkl'),strict=False)
net2 = RM.CrossScan(config).to(device)


eps = 0.0001
# optimizer1 = optim.Adam(net1.parameters(), lr=1e-4)
# scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=8, gamma=0.1,last_epoch=-1)
optimizer2 = optim.Adam(net2.parameters(), lr=1e-4)
scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=8, gamma=0.1,last_epoch=-1)
epochs = 2
writer = SummaryWriter('./logs_tensforboard')

batch = 1

## dataloader
train_data = train_data_preprocess(train_img_path, nfeatures=1000)
train_dataloader = DataLoader(train_data, batch_size=batch, shuffle=True)
train_curve = list()
train_acc = list()
log_interval = len(train_dataloader)


def evaluateRPF(CorrectIndex, OurIndex, size):
    tmp = np.zeros((1, size))
    OurIndex = np.array(OurIndex)
    tmp[:, OurIndex] = 1
    tmp[:, CorrectIndex] = tmp[:, CorrectIndex] + 1
    OurCorrect = np.where(tmp == 2)[1]
    NumCorrectIndex = len(CorrectIndex)
    NumOurIndex = len(OurIndex.T)
    NumOurCorrect = len(OurCorrect)

    # if NumOurIndex > 0:
    corrRate = NumCorrectIndex / size
    precision = NumOurCorrect / (NumOurIndex + eps)
    recall = NumOurCorrect / NumCorrectIndex
    f1 = 2*precision*recall / (precision+recall+eps)

    return precision, recall, f1, corrRate

def Test(test_img,k):
    net1.eval()
    net2.eval()
    point1 = test_img["point1"]
    BATCH_SIZE = len(point1)
    gt = test_img['GroundTruth']
    test_data = test_data_preprocess(test_img, gt)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
    thre_ratio = 0.9
    for data in test_dataloader:
        data_config = {key: data[key].to(device) for key in data}
        match = data_config['matches']
        output1= net1.forward(data_config)

        pred_coarse_re = output1.argmax(dim=1)
        selected_index = torch.where(pred_coarse_re == 1)[0]


        p1 = data_config['kp1'].to(torch.float32)
        p2 = data_config['kp2'].to(torch.float32)

        output, selected_index2 = net2.forward(p1, p2, selected_index, k, match[0])
        threshold = (torch.max(output)-torch.min(output))*thre_ratio+torch.min(output)
        out_selec_index = torch.where(output<threshold)[0].detach().cpu().numpy()
        selec_index_final = selected_index2[out_selec_index]
        final_result = np.zeros(BATCH_SIZE).astype(np.int64)
        se = selec_index_final

        final_result[se] = 1


        return final_result, gt

## train
def main():
    for epoch in range(0, epochs):
            loss_mean = 0.
            acc_mean = 0.
            correct = 0.
            total = 0.
            kl = 6

            net1.train()
            net2.train()
            for i, data in enumerate(train_dataloader):
                data_config = data
                data_config = {key:data_config[key].to(device) for key in data_config}
                match = data_config['matches']

                # optimizer1.zero_grad()
                optimizer2.zero_grad()
                output1 = net1.forward(data_config)

                pred_coarse_re = output1.argmax(dim=1)
                selected_index = torch.where(pred_coarse_re == 1)[0]
                criti1 = RM.Stage1_loss()
                loss1 = criti1(output1, match[0])

                gt_new = match[0, selected_index]
                p1 = data_config['kp1'].to(torch.float32)
                p2 = data_config['kp2'].to(torch.float32)
                output, selected_index2,output_auxi = net2.forward(p1, p2, selected_index, kl, match[0])

                label = match[0,selected_index2]
                criti2 = RM.Stage2_loss()
                loss2 = criti2(output.unsqueeze(1), label,output_auxi)

                loss = 0.2 * loss1 + loss2

                loss.requires_grad_(True)

                loss.backward()
                # optimizer1.step()
                optimizer2.step()

                loss_mean =loss_mean + loss.item()
                train_curve.append(loss.item())
                writer.add_scalar('train loss', loss.item(), i)

                if (i+1) % log_interval ==0:
                    loss_mean = loss_mean / log_interval
                    print("Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] loss:{:.4f}".format(
                        epoch, epochs, i+1, len(train_dataloader),loss_mean
                    ))
                    train_curve.append(loss_mean)
                    loss_mean = 0.
                    count = 0
                    ResultFeatureMatching = np.zeros((len(os.listdir(test_data_path)), 3))
                    with torch.no_grad():
                        for path in os.listdir(test_data_path):
                            test_img = scio.loadmat(os.path.join(test_data_path, path))        
                            output, gt_new = Test(test_img, kl)
                            OurIndex = np.where(output == 1)[0]
                            CorrectIndex = np.where(gt_new == 1)[0]
                            P, R, F1, _ = evaluateRPF(CorrectIndex, OurIndex, output.shape[0])
                            ResultFeatureMatching[count] = np.hstack((P, R, F1))
                            count += 1
                        mean_value = np.mean(ResultFeatureMatching, axis=0)
                        print(mean_value)
                    if (epoch%10)==0:
                        # torch.save(net1.state_dict(),'./model/ReM_pointu{}.pkl'.format(epoch))
                        torch.save(net2.state_dict(),'./model/ReM_pointd{}.pkl'.format(epoch))

            # scheduler1.step()
            scheduler2.step()
    writer.close()

if __name__=='__main__':
    main()




