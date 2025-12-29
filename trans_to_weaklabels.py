# 将一个多标签数据集变成一个缺失多标签数据集
import pandas as pd
import numpy as np
import random
import warnings

from pandas import DataFrame
import copy as cp

# 禁止所有警告
warnings.filterwarnings("ignore")

# 对DataFrame数据集df随机缺失一些数据处理
# 输入：df-数据集  labelsNum-数据集标签个数  missing_label_ratio-标签矩阵的缺失率
def randomLossLabelsData(df, labelsNum, missing_label_ratio):
    featuresNum = df.shape[1] - labelsNum
    df_features = df.iloc[:, :featuresNum] # 数据列矩阵
    df_labels = df.iloc[:, -labelsNum:]  # 标签列矩阵

    # 随机选择一些标签设置为缺失
    rng = np.random.default_rng(seed=random.randrange(1, 100000))  # 使用随机数生成器
    mask = rng.uniform(size=df_labels.shape) < missing_label_ratio
    lossLocList = [[x, y] for x, row in enumerate(mask) for y, value in enumerate(row) if value]  # 将缺失值的位置保存下来
    # print(lossLocList)
    df_labels[mask] = '?'  # 使用 ? 表示标签缺失
    # print('df_labels=', df_labels, df_labels.shape)

    # 合并特征和有缺失标签的DataFrame
    df_missing_labels = pd.concat([df_features, df_labels], axis=1)
    return df_missing_labels, lossLocList  # 返回缺失后的数据集和缺失值的位置二维列表

def randomNoiseLabelData(df, dfName, labNum, NoisePer):
    for i in range(0, len(NoisePer)):
        inputCsvFileName = dfName
        outCsvFileName = inputCsvFileName.replace('.csv', '_p_each.csv')
        data = DataFrame(cp.deepcopy(df.iloc[:, 0:df.shape[1]]))
        temMat = np.array(cp.deepcopy(data.iloc[:, -labNum:]))
        n, m = len(temMat), len(temMat[0])  # n, m分别是二维矩阵matrix的行和列
        noisyNum = m * NoisePer[i] // 100
        # print(1, '*', m, '的矩阵噪声数：', noisyNum, '\n')
        for row in range(0, n):
            temPos = set()
            while len(temPos) < noisyNum:
                col = random.randint(0, m - 1)
                temPos.add((row, col))  # 记录噪声的位置
                if temMat[row][col] == 1:
                    temMat[row][col] = 0
                elif temMat[row][col] == 0:
                    temMat[row][col] = 1
        temp1 = cp.deepcopy(data.iloc[:, -labNum:])
        temp1.iloc[:, -labNum:] = temMat
        data_new = data.iloc[:, 0:data.shape[1] - labNum].join(DataFrame(temp1))
        print('data_new:', data_new)
        # 将DataFrame写入新的CSV文件
        outCsvFileName_new = outCsvFileName.replace(".csv", "_") + str(NoisePer[i]) + '.csv'
        data_new.to_csv('./weaklabel_datasets/' + outCsvFileName_new, index=False)


if __name__ == '__main__':
    dataName = 'EXAMPLE'

    labNum = 7  # 标签个数
    souNum = 4  # 数据源个数

    MissRat = [0.3, 0.6]
    noiseRat = [0.3, 0.6]
    missfilePath = './weaklabel_datasets/'

    for r in range(0, len(MissRat)):
        lossLoc_2d = []
        for s in range(0, souNum):
            fileName = dataName + '_' + str(s) + '.csv'
            print('fileName: ', fileName)
            df = pd.read_csv('./datasets/' + fileName)
            df_missing_labels, lossLocList = randomLossLabelsData(df, labNum, MissRat[r])
            print('df_missing_labels=', df_missing_labels, df_missing_labels.shape)
            lossLoc_2d.append(lossLocList)
            fileName_new = fileName.replace(".csv", "_") + str(MissRat[r]) + '.csv'
            df_missing_labels.to_csv(missfilePath + fileName_new, index=False)

            randomNoiseLabelData(df_missing_labels, fileName_new, labNum, noiseRat)

