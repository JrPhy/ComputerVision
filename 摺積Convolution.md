在工程數學中，摺積就是兩個函數 f, g，其中一個移動時與另一個函數重疊時的面積，[維基百科](https://zh.wikipedia.org/zh-tw/%E5%8D%B7%E7%A7%AF#/media/File:Convolution_of_box_signal_with_itself2.gif)的圖有很好的展示。定義如下
## 一、定義
$$ (f* g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t-\tau)d\tau = \int_{-\infty}^{\infty} f(t-\tau)g(\tau)d\tau = (g *f)(t)$$

![img](https://upload.wikimedia.org/wikipedia/commons/6/6a/Convolution_of_box_signal_with_itself2.gif)

在電腦中也是一樣的計算方式。假設有兩陣列 signal 與 kernel，會要求 signal 的長度 >= kernel，且做完摺積後的長度與 signal 相同，例如
```
signal = [1, 2, 3, 4, 5]
kernel = [1, 1, 1]
(signal*kernel) = [3, 6, 9, 12, 9]
```
可以看到這樣算會被放大，所以最後還會需要除上 kernel 元素的總和，另外在計算時要注意邊界。計算步驟如下
```
   [1, 2, 3, 4, 5]
[1, 1, 1]              (1*1+2*1)/3 = 1
-----------------------------------
   [1, 2, 3, 4, 5]
   [1, 1, 1]           (1*1+2*1+3*1)/3 = 2
-----------------------------------
   [1, 2, 3, 4, 5]
      [1, 1, 1]        (2*1+3*1+4*1)/3 = 3
-----------------------------------
   [1, 2, 3, 4, 5]
         [1, 1, 1]     (3*1+4*1+5*1)/3 = 4
-----------------------------------
   [1, 2, 3, 4, 5]
            [1, 1, 1]  (4*1+5*1)/3 = 3
-----------------------------------
```
## 二、程式碼
```C++
#include <iostream>
#include <vector>
using namespace std;

vector<float> conv1D(const vector<float>& signal, const vector<float>& kernel) {
    int n = signal.size();
    int k = kernel.size();
    int pad = k / 2;
    vector<float> result(n, 0);

    for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int j = 0; j < k; j++) {
            int idx = i + j - pad;
            if (idx >= 0 && idx < n) {
                sum += signal[idx] * kernel[j];
            }
        }
        result[i] = sum;
    }
    return result;
}
```
在圖片因為是二維圖片，所以更常使用二維的摺積。\
![img](https://upload.wikimedia.org/wikipedia/commons/1/19/2D_Convolution_Animation.gif)
```C++
#include <iostream>
#include <vector>
using namespace std;

vector<vector<int>> conv2D(const vector<vector<int>>& img, const vector<vector<int>>& kernel) {
    int h = img.size(), w = img[0].size();
    int kh = kernel.size(), kw = kernel[0].size();
    int pad_h = kh / 2, pad_w = kw / 2;

    vector<vector<int>> result(h, vector<int>(w, 0));

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int sum = 0;
            for (int m = 0; m < kh; m++) {
                for (int n = 0; n < kw; n++) {
                    int x = i + m - pad_h;
                    int y = j + n - pad_w;
                    if (x >= 0 && x < h && y >= 0 && y < w) {
                        sum += img[x][y] * kernel[m][n];
                    }
                }
            }
            result[i][j] = sum;
        }
    }
    return result;
}
```
## 三、加速
雖然是二維的圖片，但實務上讀進來時都是放在一維陣列中，所以操作時也都是用一維陣列的操作方式，在此我們需要一些轉換關係，假設圖片大小為 height, width，kernel 大小為 ksize，不過 kernel 大小為奇數，所以範圍是 -(kernel-1)/2 ~ -(kernel-1)/2，將 -(kernel-1)/2 寫成 pad，就可得到
```
img[i][j] --> img[i*width+j]
kernel[m][n] --> kernel[(m + pad) * ksize + (n + pad)]
x = i + m, y = j + n
img[x][y] * kernel[m][n] --> img[x * width + y] * kernel[(m + pad) * ksize + (n + pad)];
```
```C++
vector<int> conv2D_1Darray(const vector<int>& img, int width, int height,
                           const vector<int>& kernel, int ksize) {
    int pad = ksize / 2;
    vector<int> result(img.size(), 0);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int sum = 0;
            for (int m = -pad; m <= pad; m++) {
                for (int n = -pad; n <= pad; n++) {
                    int x = i + m;
                    int y = j + n;
                    if (x >= 0 && x < height && y >= 0 && y < width) {
                        sum += img[x * width + y] * kernel[(m + pad) * ksize + (n + pad)];
                    }
                }
            }
            result[i * width + j] = sum;
        }
    }
    return result;
}
```
以一張 4000 萬畫素的照片為例，二維資料儲存處理起來時間是 2.27582 秒，一維資料儲存處理起來時間是 1.65627 秒，同一台機器上快了約 22%。
```C++
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using namespace chrono;

// 2D Convolution
vector<vector<int>> conv2D(const vector<vector<int>>& img, const vector<vector<int>>& kernel) {
    int h = img.size(), w = img[0].size();
    int kh = kernel.size(), kw = kernel[0].size();
    int pad_h = kh / 2, pad_w = kw / 2;
    vector<vector<int>> result(h, vector<int>(w, 0));

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int sum = 0;
            for (int m = 0; m < kh; m++) {
                for (int n = 0; n < kw; n++) {
                    int x = i + m - pad_h;
                    int y = j + n - pad_w;
                    if (x >= 0 && x < h && y >= 0 && y < w) {
                        sum += img[x][y] * kernel[m][n];
                    }
                }
            }
            result[i][j] = sum;
        }
    }
    return result;
}

// 1D Convolution
vector<int> conv2D_1Darray(const vector<int>& img, int width, int height,
                           const vector<int>& kernel, int ksize) {
    int pad = ksize / 2;
    vector<int> result(img.size(), 0);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int sum = 0;
            for (int m = -pad; m <= pad; m++) {
                for (int n = -pad; n <= pad; n++) {
                    int x = i + m;
                    int y = j + n;
                    if (x >= 0 && x < height && y >= 0 && y < width) {
                        sum += img[x * width + y] * kernel[(m + pad) * ksize + (n + pad)];
                    }
                }
            }
            result[i * width + j] = sum;
        }
    }
    return result;
}

int main() {
    int width = 2880, height = 2800;
    vector<vector<int>> img2D(height, vector<int>(width, 1)); // 測試用影像 (二維)
    vector<int> imgFlat(width * height, 1);                   // 測試用影像 (一維)
    vector<vector<int>> kernel2D(5, vector<int>(5, 1));       // 5x5 kernel (二維)
    vector<int> kernelFlat(25, 1);                            // 5x5 kernel (一維)

    auto t1 = high_resolution_clock::now();
    auto r1 = conv2D(img2D, kernel2D);
    auto t2 = high_resolution_clock::now();
    cout << "標準二維卷積耗時: " << duration<double>(t2 - t1).count() << " 秒\n";
    auto t11 = duration<double>(t2 - t1).count();

    auto t5 = high_resolution_clock::now();
    auto r3 = conv2D_1Darray(imgFlat, width, height, kernelFlat, 5);
    auto t6 = high_resolution_clock::now();
    cout << "高效版一維陣列卷積耗時: " << duration<double>(t6 - t5).count() << " 秒\n";
    auto t12 = duration<double>(t2 - t1).count();
    cout  << 1-t12/t11 << " %\n";
    return 0;
}
```
