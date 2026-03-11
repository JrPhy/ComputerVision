一張圖片裡面往往會有包含許多的物體，影像辨識就會希望找出不同的物體並標出來，所以就會將圖片分割成好幾個小塊，再針對這些區域去做 CNN。不過 CNN 本身需要耗費大量的計算資源，所以會希望只對有可能感興趣的地方做 CNN，稱為 Region with CNN (RCNN)，比較有名的套件為 YOLO (You Only Look Once)

## 一、R-CNN
算法分為4個步驟
1. Selective Search：輸入一張影像，輸出約 2,000 個候選框。
2. Warp：將這些大小不一的框強制縮放 (Resize) 為固定尺寸（例如 227 * 227）。
3. CNN：將縮放後的圖像送入卷積神經網絡提取特徵。
4. SVM & Bbox Regressor：最後進行分類與邊框修正。

![img](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*p4hOwFS6AVAGzpfl65B8xg.png)
[source](https://medium.com/lifes-a-struggle/object-detection-r-cnn-a07c2d95d6a0)
其中的 Selective Search 就是標出可能是同一物體的部分，利用 Felzenszwalb 演算法，根據像素的顏色、強度等特徵，將影像切分成數千個細小的原始區域。雖然已經能降低計算量，但實際上還是很多，而且主要都是 if-else 只能在 CPU 去做。後來又發展出 Faster R-CNN 完全放棄 Selective Search。
![img](https://learnopencv.com/wp-content/uploads/2017/09/dogs-golden-retriever-top-250-proposals.jpg)
[source](https://learnopencv.com/selective-search-for-object-detection-cpp-python/#elementor-toc__heading-anchor-13)

## 二、Faster R-CNN
相較於 Selective Search，Faster R-CNN 使用多個預設大小與長寬比的框去做 CNN，先判斷框內是否有物體，如果有框到物體則會去微調位置來讓物體盡量落於框內，然後再去做 CNN 跟分類。
```C++
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::dnn;

int main() {
    // 1. 設定模型路徑 (以 TensorFlow 格式為例)
    string modelWeights = "faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";
    string modelConfig = "faster_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    
    // 2. 載入網路
    Net net = readNetFromTensorflow(modelWeights, modelConfig);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU); // 若有 GPU 可改為 DNN_TARGET_CUDA

    // 3. 讀取影像
    Mat frame = imread("test_image.jpg");
    if (frame.empty()) {
        cerr << "無法讀取影像！" << endl;
        return -1;
    }

    // 4. 影像預處理 (Blob)
    // Faster R-CNN 通常不需要像 YOLO 那樣手動 resize，DNN 模組會處理
    Mat blob = blobFromImage(frame, 1.0, Size(), Scalar(), true, false);
    net.setInput(blob);

    // 5. 執行前向傳播 (Forward Pass)
    // Faster R-CNN 的輸出通常是一個 [1, 1, N, 7] 的張量
    Mat detections = net.forward();

    // 6. 解析結果
    float confidenceThreshold = 0.5;
    Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > confidenceThreshold) {
            int classId = static_cast<int>(detectionMat.at<float>(i, 1));
            
            // 取得邊框座標 (正規化後的座標，需乘以原圖寬高)
            int left   = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int top    = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int right  = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

            // 繪製結果
            rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);
            string label = format("Class %d: %.2f", classId, confidence);
            putText(frame, label, Point(left, top - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        }
    }
    imshow("Faster R-CNN Detection", frame);
    waitKey(0);
    return 0;
}
```
![img](https://github.com/JrPhy/ComputerVision/blob/master/img/fasterRCNN.png)

可以看到相較於 RCNN，Faster R-CNN 找出來的框就更少了，所以整個效能就會提升很多。

## 三、YOLO
