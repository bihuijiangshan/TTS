import torch
import os
from torch.utils.data import DataLoader
from DataSet import NumTrain_DataSet, collate_length
from Models import TextToSpeechModel
from Feature_Utils import get_spectrograms


def write_line2log(log_dict: dict, filedir, isprint: True):
    # 记录损失
    strp = ''
    with open(filedir, 'a', encoding='utf-8') as f:
        for key, value in log_dict.items():
            witem = '{}'.format(key) + ':{},'.format(value)
            strp += witem
        f.write(strp)
        f.write('\n')
    if isprint:
        print(strp)
    pass


def start_train(root_dir, train_scp_path):
    # 创建文件夹和文本文档，创建数据集对象，创建dataloader，用绝对值损失函数，Adam优化；测试
    os.makedirs("Experiment")
    os.makedirs("Experiment/model_save")
    with open("Experiment/train_log.txt", 'w', encoding='utf-8') as f:
        pass
    with open("Experiment/eval_log.txt", 'w', encoding='utf-8') as f:
        pass
    # 训练用的设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("训练用的设备", device)
    # 初始化数据集
    datas = NumTrain_DataSet(root_dir=root_dir, traintxtpath=train_scp_path, output_dir='Experiment')
    d_loader = DataLoader(datas, batch_size=5, shuffle=True, drop_last=True, collate_fn=collate_length)

    # 初始化模型
    print("词典token个数：", datas.token_dict_len)
    # token_nums加1，有零填充的token,一共15个token
    model = TextToSpeechModel(token_nums=datas.token_dict_len+1, embedding_dim=512).to(device)
    # 初始化损失函数
    loss_f = torch.nn.L1Loss().to(device)  # 绝对值损失
    # loss_f = torch.nn.MSELoss().to(device)
    # Adam优化
    optim = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

    # 训练时候的变量记录
    epoch_num = 0  # 轮数
    train_iter = 0  # 步数

    while 1:
        epoch_num += 1
        for batch in d_loader:
            train_iter += 1
            text_padded, mel_padded = batch
            text_padded, mel_padded = text_padded.to(device), mel_padded.to(device)
            output = model(text_padded, mel_padded)
            loss_l1 = loss_f(output, mel_padded)

            optim.zero_grad()  # 梯度清空
            loss_l1.backward()  # 计算梯度
            optim.step()  # 更新参数

            loss_d = {"epoch": epoch_num, "step": train_iter, "l1_loss": loss_l1.item()}
            write_line2log(loss_d, "Experiment/train_log.txt", True)

            # 保存模型，隔多少步保存一次
            if train_iter % 1000 == 0:
                torch.save(model.state_dict(), "Experiment/model_save/train_{:06}.pth".format(train_iter))

            # 测试
            if train_iter % 1000 == 0:
                eval_loss_es = []
                # 用eval.txt测试模型效果
                with open("Experiment/测试集语音路径.txt", encoding='utf-8') as f:
                    for l in f.readlines():
                        l = l.strip('\n')  # 去除换行符
                        wav_name, text = l.split('|')[0], l.split('|')[1]
                        labels = torch.LongTensor([datas.token_set.index(x)+1 for x in list(text)])
                        labels = labels.unsqueeze(0).to(device)  # [1,T1]
                        mel, _ = get_spectrograms(wav_name)
                        mel = torch.FloatTensor(mel)
                        mel = mel.unsqueeze(0).to(device)  # [1,80,T2]

                        eval_outmel = model(labels, mel)
                        eval_loss = loss_f(eval_outmel, mel)
                        eval_loss_es.append(eval_loss.item())
                    # 汇总5条测试语音的loss
                    eval_loss_es = sum(eval_loss_es) / len(eval_loss_es)
                    loss_d = {"epoch": epoch_num, "step": train_iter, "eval_l1_loss": eval_loss_es}
                    write_line2log(loss_d, "Experiment/eval_log.txt", True)
    pass


if __name__ == '__main__':
    # 每次运行该函数之前必须删除 Experiment
    root_dir = r'D:\abc\yyhc\num_train'
    traintxtpath = r'D:\abc\yyhc\num_train\train.txt'

    start_train(root_dir, traintxtpath)
    pass