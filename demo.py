import os
os.environ["CUDA_VISIBLE_DEVICES"] ='4'
import torch
from torch.utils.data import DataLoader
import ReM_PointNet_DGCNNall as RM
from ReM_Pointdataset import test_data_plot
import scipy.io as scio
import numpy as np

config = {
    'embed_dim': 256,
    'keypoint_encoder': [32, 64, 128,256],
    'layer_names': ['cross','self'] * 4,
    'output_dim':2,
    'k':18,
}

test_data_path = './EXP/1.mat'


## load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net1 = RM.DGCNN(config).to(device)
net1.load_state_dict(torch.load('./model/ReM_pointu.pkl'),strict=False)
net2 = RM.CrossScan(config).to(device)
net2.load_state_dict(torch.load('./model/ReM_pointd.pkl'),strict=False)  

eps = 0.0001
thre_ratio = 0.9

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
    precision = NumOurCorrect / (NumOurIndex + eps)
    recall = NumOurCorrect / NumCorrectIndex
    f1 = 2*precision*recall / (precision+recall+eps)
    # else:
    #     precision, recall, f1, corrRate = 0, 0, 0, 0
    return precision, recall, f1


def Test(test_img,kl):
    net1.eval()
    net2.eval()
    point1 = test_img["point1"]
    bs = len(point1)
    gt = test_img['GroundTruth']

    test_data = test_data_plot(test_img, gt)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=bs)
    for data in test_dataloader:
        data_config = {key: data[key].to(device) for key in data}
        output1 = net1.forward(data_config)

        pred_coarse_re = output1.argmax(dim=1)
        selected_index = torch.where(pred_coarse_re == 1)[0]

        # point
        p1 = data_config['kp1'].to(torch.float32)
        p2 = data_config['kp2'].to(torch.float32)

        # similarity
        output, selected_index2 = net2.forward(p1, p2, selected_index,kl)

        threshold = (torch.max(output)-torch.min(output))*thre_ratio+torch.min(output)
        out_selec_index = torch.where(output<threshold)[0].detach().cpu().numpy()

        selec_index_final = selected_index2[out_selec_index]
        final_result = np.zeros(bs).astype(np.int64)
        se = selec_index_final
        final_result[se] = 1

        return final_result, gt

if __name__=='__main__':
    c = 6
    test_img = scio.loadmat(test_data_path)

    output,gt = Test(test_img, c)

    OurIndex = np.where(output==1)[0]
    CorrectIndex = np.where(gt == 1)[0]
    P, R, F1 = evaluateRPF(CorrectIndex, OurIndex,output.shape[0])

    print('Precision:{}, Recall:{}, F1:{}'.format(P,R,F1))







