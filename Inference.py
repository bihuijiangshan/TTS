import torch
import numpy as np
import os
from pathlib import Path
from Models import TextToSpeechModel
import matplotlib.pyplot as plt
from Feature_Utils import melspectrogram2wav
import soundfile


def inference_a_wav(infe_text):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    modelstate_path = 'Experiment/model_save/train_005500.pth'
    mel_factor = 16
    # 创建文件
    result_savedir = 'Experiment/inference_result'
    print("推理结果保存在", result_savedir)
    os.makedirs(result_savedir, exist_ok=True)

    token_dict_path = 'Experiment/token_dict.txt'

    # 读取词典，以方便知道 文字对应的下标
    texts = []
    inds = []
    with open(token_dict_path, encoding='utf-8') as f:
        for l in f.readlines():
            text, ind = l.split('|')[0], l.split('|')[1]
            texts.append(text)
            inds.append(int(ind))

    # 读取模型
    model = TextToSpeechModel(token_nums=len(inds)+1, embedding_dim=512).to(device)
    model.load_state_dict(torch.load(modelstate_path, map_location=device))
    model.eval()

    # 推理的文字
    textlabels = [inds[texts.index(s)] for s in infe_text]  # 转化成对应下标数组
    print("text labels:", textlabels)
    textlabels = torch.LongTensor([textlabels]).to(device)  # [1,T]
    infe_len = textlabels.shape[-1] * mel_factor
    inference_mel = model.inference(textlabels, infe_len)

    # 设定推理的文件名 = 文本 + 模型名
    infe_name = infe_text + "_" + Path(modelstate_path).stem

    # 存储推理语音mel谱和图像，画出推理的语音的频谱图像
    npy_mel = inference_mel.squeeze().detach().cpu().numpy()  # 将语音从张量转换为cpu的npy格式
    plt.figure()
    plt.imshow(npy_mel, cmap='Greens')  # [80,L = 16 * text len]
    plt.savefig(result_savedir + f"/{infe_name}.png")
    plt.show()

    np.save(result_savedir + f"/{infe_name}.npy", npy_mel)

    # 转为时域波形，存储.wav
    wavform = melspectrogram2wav(npy_mel.T)
    print("时域信号：", wavform.shape)
    soundfile.write(result_savedir + f"/{infe_name}.wav", wavform, 16000)
    pass


if __name__ == '__main__':

    # 推理你想要听的汉字
    inference_a_wav("七八九五")
    pass