import torch
from torch import nn
from torch.nn import functional as F


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))  # 初始化卷积层的权重

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


# 三层一维卷积+relu激活函数+dropout
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(3):  # 3层卷积模块
            conv_layer = nn.Sequential(
                ConvNorm(512, 512, kernel_size=5, stride=1, padding=int((5 - 1) / 2), dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, x):
        # 输入的x的维度：B,text length ,D]，返回的x的维度：[B,text length ,D]
        x = x.permute(0, 2, 1).contiguous()  # [B,T,D] -> [B,D,T]
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)  # [B,D,T] -> [B,T,D]
        return x


# 前馈：两层一维卷积+残差
class PositionwiseFeedForward(nn.Module):
    # 前馈
    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=kernel_size[0], padding=(kernel_size[0] - 1) // 2)
        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=kernel_size[1], padding=(kernel_size[1] - 1) // 2)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)  # residual残差
        return output


# LSTM+一维卷积
class Decoder(nn.Module):
    # 两个前馈层模块
    def __init__(self):
        super().__init__()

        self.rnn_layers = nn.ModuleList()
        self.FFT_blocks = nn.ModuleList()
        # 先初始化第一层RNN和FFT。
        self.rnn_layers.append(torch.nn.LSTM(input_size=512, hidden_size=512, batch_first=True))
        self.FFT_blocks.append(PositionwiseFeedForward(d_in=512, d_hid=512, kernel_size=[9, 1]))

        # 初始化第 2-(decoder_layernum-1)层
        for i in range(2):
            self.rnn_layers.append(torch.nn.LSTM(input_size=512, hidden_size=512, batch_first=True))
            self.FFT_blocks.append(PositionwiseFeedForward(d_in=512, d_hid=512, kernel_size=[9, 1]))

        # 初始化最后一层，size维度变回80
        self.rnn_layers.append(torch.nn.LSTM(input_size=512, hidden_size=80, batch_first=True))
        self.FFT_blocks.append(PositionwiseFeedForward(d_in=80, d_hid=80, kernel_size=[9, 1]))

    def forward(self, x):
        # 输入x的维度：[B,T,D]，是encoder的输出经过length regulator的向量，T已经是melspec的长度
        for k in range(len(self.rnn_layers)):
            x, _ = self.rnn_layers[k](x)
            x = self.FFT_blocks[k](x)
        x = x.transpose(1, 2)
        return x


class Length_Regulator(nn.Module):  # 扩充文本张量再填充或裁剪，使其与语音张量大小相同
    def __init__(self, expand_factor):
        # 输入：文本张量[B,Text_len,D]，mel_len，输出mel谱张量[B,mel_len,D]
        super().__init__()
        # 记录每个文字要扩充的melspec长度
        self.expand_factor = expand_factor

    def forward(self, text_memory, mel_len):
        # 根据一个固定的数值来进行扩充
        # 把text_memory中的text len直接倍数扩增到mel len
        mel_chunks = []
        text_len = text_memory.shape[1]
        for t in range(text_len):
            # 取出第t个时刻的文本向量，并重复factor遍
            t_vec = text_memory[:, t, :].unsqueeze(1)  # [B,1,D]
            # 将t_vec重复expand_factor次
            t_vec = t_vec.repeat(1, self.expand_factor, 1)  # [B,1,D] -> [B,self.expand_factor,D]
            # 第t个文本已经扩充到了其对应的mel谱的长度。
            mel_chunks.append(t_vec)

        # 修正输出的长度，[B,self.expand_factor * text len,D] -> [B,melspec len,512]
        # 在时间维度上，拼接所有时间帧
        mel_chunks = torch.cat(mel_chunks, dim=1)
        B, cat_mel_len, D = mel_chunks.shape
        # 确认其长度与给定的目标melspec一致
        if cat_mel_len < mel_len:
            # 填充拼接一个全零张量
            pad_t = torch.zeros((B, mel_len-cat_mel_len, D), device=text_memory.device)
            mel_chunks = torch.cat([mel_chunks, pad_t], dim=1)
        else:
            mel_chunks = mel_chunks[:, :mel_len, :]
        return mel_chunks  # [B,melspec,512]

        pass


class TextToSpeechModel(nn.Module):
    def __init__(self, token_nums, embedding_dim):

        super().__init__()

        self.embedding_layer = torch.nn.Embedding(token_nums, embedding_dim, padding_idx=0)  # 将文本嵌入为张量
        self.encoder = Encoder()
        self.length_regulator = Length_Regulator(expand_factor=16)
        self.decoder = Decoder()

    def forward(self, text_label, mel):
        # 输入text_label的维度: [B,T]，mel的维度:[B,D,T]
        mel_len = mel.shape[-1]
        text_emb = self.embedding_layer(text_label)
        text_memory = self.encoder(text_emb)
        decoder_input = self.length_regulator(text_memory, mel_len)
        decoder_output = self.decoder(decoder_input)
        return decoder_output

    def inference(self, text_label, mel_len):
        text_emb = self.embedding_layer(text_label)
        text_memory = self.encoder(text_emb)
        decoder_input = self.length_regulator(text_memory, mel_len)
        decoder_output = self.decoder(decoder_input)
        return decoder_output