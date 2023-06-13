from matplotlib import pyplot as plt
from pathlib import Path


def plot_loss_bylogtxt(txt_path, ver):

    # 寻找txt中全部数据项，构造一个字典
    with open(txt_path, encoding='utf-8') as f2:
        lines = f2.readlines()
        # 找到记录loss的行
        start_index = -1
        for line in lines:
            start_index += 1
            if "*****" == line[:5]:
                start_index += 1  #记录了loss记录开始的行数
                break
        loss_lines = lines[0:]

        # 寻找数据项的key
        loss_keys = []
        for line in loss_lines:
            l_keys = [k_v.split(':')[0] for k_v in line.split(',')[:-1]]
            loss_keys += l_keys
        loss_keys = list(set(loss_keys))  # 取集合得到字典的key
        # 创建字典，保存全部数据项的值和键
        loss_dict = {}
        for k in loss_keys:
            loss_dict[k] = []

        # 将每一个键对应的一组数值加入到列表中，这些数值要逐行去读取loss_lines
        for line in loss_lines:
            l_keys_values = [(k_v.split(':')[0], k_v.split(':')[1]) for k_v in line.split(',')[:-1]]
            # 将每个 A:B,添加到字典。
            for a_b in l_keys_values:
                loss_dict[a_b[0]].append(float(a_b[1]))

        # 作图并保存
        save_file_dir = Path('Experiment/Loss_Curves_Figures_{}'.format(ver, ver))
        save_file_dir.mkdir(parents=True, exist_ok=True)
        # 作图
        for k, v in loss_dict.items():
            plt.figure()
            plt.title(str(k))
            # 去除v中的异常值
            if "D" in k:
                v = [x for x in v if x < 10]
            plt.plot(range(len(v)), v)
            plt.savefig(str(save_file_dir / f"{k}.png"))
            plt.show()

            print("{},max:{},min:{}".format(k, max(v), min(v)))
    pass


if __name__ == '__main__':
    ver = 'v1'
    pa = 'Experiment/train_log.txt'
    plot_loss_bylogtxt(pa, ver)
    pass