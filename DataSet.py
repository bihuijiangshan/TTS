import torch
from pathlib import Path
from torch.utils.data import Dataset
from Feature_Utils import get_spectrograms


# padding、lengths记录 、右填充，与最长的相同
def collate_length(batch):

    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x[0].shape[-1] for x in batch]),
        dim=0, descending=True)  # 对每一个batch中每个样本的文本序列长度进行排序，存储索引
    max_input_len = input_lengths[0]
    num_mels = batch[0][1].size(0)

    text_padded = torch.zeros((len(batch), max_input_len)).long()  # 填充至长度与最长的相同

    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]][0]
        text_padded[i, :text.size(0)] = text  # 填充到张量中

    # 对音频排序、填充
    max_target_len = max([x[1].size(1) for x in batch])
    if max_target_len % 1 != 0:
        max_target_len += 1 - max_target_len % 1
        assert max_target_len % 1 == 0

    mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
    mel_padded.zero_()

    for i in range(len(ids_sorted_decreasing)):
        mel = batch[ids_sorted_decreasing[i]][1]
        mel_padded[i, :, :mel.size(1)] = mel
    return text_padded, mel_padded


class NumTrain_DataSet(Dataset):
    def __init__(self, root_dir, traintxtpath, output_dir):

        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)

        token_set = []  # 保存全部语音对应的文本的token
        self.datas = []
        with open(traintxtpath, encoding='utf-8') as f:
            for l in f.readlines():
                l = l.strip('\n')  # 去除换行符
                wav_name, text = l.split('|')[0], l.split('|')[1]
                self.datas.append((self.root_dir / wav_name, text))
                token_set += list(text)  # 将第K条语音的文本直接转列表，添加到字符集
        self.token_set = list(set(token_set))  # 获取词典，使用set取集合
        self.token_dict_len = len(self.token_set)

        # 将词典以 txt文档的格式保存到硬盘里
        with open(self.output_dir / "token_dict.txt", 'w', encoding='utf-8') as f:
            for tok in self.token_set:
                f.write(f"{tok}|{self.token_set.index(tok)+1}\n")  # 写入token和对应下标，0是填充符，给label的下标全部加1

        # 拿出5条作为测试集，输出到output文件下
        self.train_datas = self.datas[5:]
        self.eval_datas = self.datas[:5]
        with open(self.output_dir / "测试集语音路径.txt", 'w', encoding='utf-8') as f:
            for d in self.eval_datas:
                f.write(f"{d[0]}|{d[1]}\n")
        print('从训练集取5条作为测试集.')
        print("写入测试集.txt到:", output_dir)

        self.L = len(self.train_datas)

    def __getitem__(self, idx):
        # 获取索引idx的数据
        wav_path, text_list = self.train_datas[idx]
        # 读取mel谱
        mel, _ = get_spectrograms(str(wav_path))
        mel = torch.FloatTensor(mel)  # mel谱转为张量

        # 将标签文本转化为下标的表示
        labels = [self.token_set.index(x)+1 for x in text_list]
        labels = torch.LongTensor(labels)

        return labels, mel

    def __len__(self):
        return self.L