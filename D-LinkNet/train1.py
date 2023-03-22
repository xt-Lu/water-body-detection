import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 读取数据
csv_path = "npp1.csv"
df1 = pd.read_csv(csv_path)
df1.index = df1["index"]
# 转置矩阵
df = pd.DataFrame(df1.values.T, index=df1.columns, columns=df1.index)

# 设定固定参数
split_fraction = 0.76
train_split = int(split_fraction * int(df.shape[0]))
step = 1
past = 4
future = 1
learning_rate = 0.0002
batch_size = 6
epochs = 1000
sequence_length = 4


# 对所选择数据根据公式进行归一化
def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


length = df.shape[1]
features = df.drop(index="index")
# 为后续反归一化保存原始数组
data1 = features
data2 = data1.loc['2021']
features.head()
features = normalize(features.values, train_split)
features = pd.DataFrame(features)
features.head()

# train_data为训练集 val_data为验证集
train_data = features.loc[0: train_split - 1]
val_data = features.loc[train_split:]

# 设定训练集
end = train_split - past - future + 1
start = past

# array形式训练集
x_train = train_data.iloc[0:16][[i for i in range(length)]].values
y_train = train_data.iloc[start:][[i for i in range(length)]].values

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

# 设定验证数据集
x_val = val_data.iloc[:4][[i for i in range(length)]].values
y_val = val_data.iloc[4:][[i for i in range(length)]].values

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)  # (9,2,120750)
print("Target shape:", targets.numpy().shape)  # (9,1,120750)

# 进行训练
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
# 第一层网络
lstm1 = keras.layers.LSTM(units=32, return_sequences=True)(inputs)
# 第二层网络
lstm2 = keras.layers.LSTM(units=64)(lstm1)
# 全连接层 输出个数
outputs = keras.layers.Dense(120750)(lstm2)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

# 回调来定期保存检查点
path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=50)
modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)

# 可视化函数损失情况
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(len(loss))
plt.plot(epochs, loss, "b", label="Training loss")
plt.plot(epochs, val_loss, "r", label="Validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# 选择一组代入数据进行预测
for x, y in dataset_val.take(1):
    print()

data = model.predict(x)
data = pd.DataFrame(data)
data.columns = df1.index
# 反归一化
data_mean = data1.mean(axis=0)
data_std = data1.std(axis=0)
data_predict = data
data = data * data_std + data_mean
data_predict = data

# 将series转换为DataFrame
data2 = pd.DataFrame(data2)
data2 = pd.DataFrame(data2.values.T, index=data2.columns, columns=data2.index)

# 均方根误差
MSE = np.square(np.subtract(data_predict, data2))
MSE = MSE.mean(1)
rsme = math.sqrt(MSE)
print("均方根误差:\n")
print(rsme)  # 66.52

# 重构数组
result = np.zeros([436, 647])
for i in list(df.columns.values):
    a = data[i]
    index_x = int(i / 647)
    index_y = i % 647
    result[index_x, index_y] = data[i]

MSE = np.square(np.subtract(data_predict, data2)).mean()
rsme = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(rsme)
outputpath = r"D:\Desktop\result2.csv"
result = pd.DataFrame(result)
result.to_csv(outputpath, sep=',', index=False, header=True)
