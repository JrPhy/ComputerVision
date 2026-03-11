顧名思義就是利用[摺積 Convolution](https://github.com/JrPhy/opencv/blob/master/%E6%91%BA%E7%A9%8DConvolution.md) 與神經網路 Neural Network 來做辨識，其步驟為
1. 利用摺積提取特徵
2. 利用池化保留最強特徵
3. 得到輸出值

在此以數字辨識為例，當[前處理](https://github.com/JrPhy/opencv/blob/master/%E6%95%B8%E5%AD%97%E8%BE%A8%E8%AD%98.md)做完後就開始利用捲積提取特徵

## 一、利用摺積提取特徵
摺積在此的用處就會去提取特徵。一般來說經過摺積後矩陣大小會變小，如下圖所示

![img](https://docs.opencv.org/4.x/convolution-example-matrix.gif)

做摺積的過程在此稱為「特徵提取」。當然特徵不會只有一個，所以會做好幾次摺積，就會有好幾個特徵。所以在 CNN 有一張很有名的圖，就是從一張 w*h *1 的圖片，經過特徵粹取後會變成 m *n *d 的矩陣，其中 m < w, n < h, d > 1，d 即為做了幾次摺積

![img](https://learnopencv.com/wp-content/uploads/2023/01/Convolutional-Neural-Networks-1024x611.png)

假設以做完前處理且得到數字 5 的矩陣為 X

$$\
X =
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 255 & 255 & 255 & 0 & 0 & 0 \\
0 & 255 & 0 & 0 & 0 & 0 & 0 \\
0 & 255 & 255 & 255 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 255 & 0 & 0 \\
0 & 255 & 255 & 255 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
,
K =
\begin{bmatrix}
-1 & -1 & -1 \\
0 & 0 & 0 \\
1 & 1 & 1
\end{bmatrix}
\$$

$$\
X*K = F =
\begin{bmatrix}
0 & 255 & 255 & 255 & 0 \\
0 & -255 & 0 & -255 & 0 \\
0 & 255 & 255 & 255 & 0 \\
0 & -255 & 0 & -255 & 0 \\
0 & 255 & 255 & 255 & 0
\end{bmatrix}
\$$

可以看到原本為 7x7 的矩陣經過摺積後就變成 5x5。當然 kernel 不只一個，會有許多的 kernel 一直做。

## 二、保留特徵
做完提取後得到的矩陣值可能有正有負，所以會用 ReLU ( f(x) = max(0, x) ) 來移除負值，移除後那些神經元就不再更新，就可能會失去一些微小的特徵，如「筆劃缺口」或「背景對比」特徵，所以現在常用 leak-ReLU ( f(x) = x if x > 0, ax if x <= 0, a 為一個微小的數 )來保留這部分，然後再做池化 pooling，也就是保留最顯著特徵，大多是只取一個更小矩陣範圍內的最大值，所以矩陣會變得更小。假設池化矩陣為 2x2，那麼池化後的矩陣 P 為

$$\
F =
\begin{bmatrix}
0 & 255 & 255 & 255 & 0 \\
0 & -255 & 0 & -255 & 0 \\
0 & 255 & 255 & 255 & 0 \\
0 & -255 & 0 & -255 & 0 \\
0 & 255 & 255 & 255 & 0
\end{bmatrix}
\
,
P =
\begin{bmatrix}
255 & 255 & 255 \\
255 & 255 & 255 \\
255 & 255 & 255
\end{bmatrix}
\$$

## 三、全連接層
再數字辨識中有 0~ 9 個數字要辨識，所以做完特徵提取後在丟進[神經網路](https://github.com/JrPhy/MachineLearning/blob/master/Neural%20Network/Neural_Network.md)中得到一個線性輸出，最後經過 SOFTMAX 來看那個數字的機率最大。在連接層中有個權重矩陣 W，大小為要辨識的數量與特徵維度決定，在此例中要辨識的數量有 10 個，P 的維度為 9，在此會將 P 攤平成 9x1 的矩陣，所以 W 為 10x9 的矩陣，然後乘上 P 就可得到 y，為一個 10x1 的矩陣，其 index 就表示這張圖可能為 0 ~9 的機率。在此會再另外加上一個偏移來允許一些雜訊，所以即為

$$\ y = Wx + b \$$

然後可以利用反向傳播來一直更新 W 與 b 得到準確的結果。

由於 openCV 沒有 NN 的模組，所以通常是拿來做前處理，之後使用 tensorflow 或 PyTorch 來做 NN。
```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 1. 載入 MNIST 資料集
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 2. 前處理
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32") / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 3. 建立 CNN 模型 (使用 LeakyReLU)
model = models.Sequential([
    layers.Conv2D(32, (3,3), input_shape=(28,28,1)),
    layers.LeakyReLU(alpha=0.01),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3)),
    layers.LeakyReLU(alpha=0.01),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3)),
    layers.LeakyReLU(alpha=0.01),

    layers.Flatten(),
    layers.Dense(64),
    layers.LeakyReLU(alpha=0.01),
    layers.Dense(10, activation="softmax")
])

# 4. 編譯模型
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# 5. 訓練模型
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 6. 測試模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)
```
