import librosa
import numpy as np
import scipy.signal as signal
import copy

sr = 16000  # 采样率
n_fft = 800  # fft点数
hop_length = 200  # 步长
win_length = 800  # 窗长
n_mels = 80  # mel频带数
power = 1.2  # 预测幅度的放大指数
n_iter = 100  # 谱重构的迭代次数
preemphasis = .97  # 预加重系数
max_db = 100  # 最大能量
ref_db = 20  # 参考能量的幅度值
top_db = 15  # 幅度压缩的阈值


def get_spectrograms(fpath):
    # 返回归一化的log梅尔频谱和log幅度谱，梅尔谱：(T, n_mels)的张量，幅度谱：(T, 1+n_fft/2)的张量
    y, sr = librosa.load(fpath, sr=16000)  # 加载音频文件
    y, _ = librosa.effects.trim(y, top_db=top_db)  # 裁剪静音片段
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])  # 预加重
    linear = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # 幅度谱
    mag = np.abs(linear)  # (1+n_fft//2, T)
    # 梅尔谱
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # 归一化
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    return mel, mag  # (80,T)


def melspectrogram2wav(mel):
    # 梅尔谱生成波形
    # 转置
    mel = mel.T
    # 反归一化
    mel = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db
    # 转为线性幅度谱
    mel = np.power(10.0, mel * 0.05)
    m = _mel_to_linear_matrix(sr, n_fft, n_mels)  # 将mel滤波器组转为恒定的、频率为Hz的滤波器组
    mag = np.dot(m, mel)

    # 重构音频
    wav = griffin_lim(mag)
    # 去预加重
    wav = signal.lfilter([1], [1, -preemphasis], wav)
    # 裁剪
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def _mel_to_linear_matrix(sr, n_fft, n_mels):
    # 计算将经过Mel频率滤波器组处理的信号转换回线性频率尺度的矩阵
    m = librosa.filters.mel(sr, n_fft, n_mels)  # 生成mel滤波器组
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)  # m滤波后频谱的协方差矩阵
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]  # 每个元素的值是除以该列的元素之和的倒数
    return np.matmul(m_t, np.diag(d))


def griffin_lim(spectrogram):
    # 将频谱图转换为音频
    X_best = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        X_t = librosa.istft(X_best, hop_length, win_length=win_length, window="hann")
        est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = librosa.istft(X_best, hop_length, win_length=win_length, window="hann")
    y = np.real(X_t)

    return y