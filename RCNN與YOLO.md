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

## 三、Mask R-CNN
Faster R-CNN 已經能很好的框出不同的物體，但若還需要更精確的描出物體，則還需要改進此方法。CNN 會一直縮小圖片的大小，到最後就只剩下一個 HEATMAP，最後做 Max pooling 就會讓框偏移。而M ask R-CNN 改採用雙線性插值法(Bilinear Interpolation)來改善 RoI Pooling，稱之為 RoIAlign，RoIAlign 會讓遮罩位置更準確。從相較於上圖，下圖可以更精準地涵蓋貓與狗。
```C++
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::dnn;

int main() {
    // 1. 設定路徑
    string modelWeights = "mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";
    string modelConfig = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";

    // 2. 載入網路
    Net net = readNetFromTensorflow(modelWeights, modelConfig);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    Mat frame = imread("test_mask.jpg");
    if (frame.empty()) return -1;

    // 3. 預處理
    Mat blob = blobFromImage(frame, 1.0, Size(), Scalar(), true, false);
    net.setInput(blob);

    // 4. 前向傳播 (取得兩個輸出層)
    // "detection_out_final" 是座標, "detection_masks" 是像素遮罩
    vector<String> outNames = {"detection_out_final", "detection_masks"};
    vector<Mat> outs;
    net.forward(outs, outNames);

    Mat outDetections = outs[0]; // [1, 1, N, 7]
    Mat outMasks = outs[1];      // [N, 90, 15, 15] (90 是類別數)

    // 5. 解析與繪製
    float scoreThresh = 0.8;
    int numDetections = outDetections.size[2];
    
    for (int i = 0; i < numDetections; i++) {
        float score = outDetections.at<float>(0, 0, i, 2);
        if (score > scoreThresh) {
            int classId = (int)outDetections.at<float>(0, 0, i, 1);
            int left   = (int)(outDetections.at<float>(0, 0, i, 3) * frame.cols);
            int top    = (int)(outDetections.at<float>(0, 0, i, 4) * frame.rows);
            int right  = (int)(outDetections.at<float>(0, 0, i, 5) * frame.cols);
            int bottom = (int)(outDetections.at<float>(0, 0, i, 6) * frame.rows);

            // 限制範圍避免溢出
            left = max(0, min(left, frame.cols - 1));
            top = max(0, min(top, frame.rows - 1));
            right = max(0, min(right, frame.cols - 1));
            bottom = max(0, min(bottom, frame.rows - 1));
            Rect box(left, top, right - left + 1, bottom - top + 1);

            // 6. 處理 Mask (像素對齊)
            Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));
            resize(objectMask, objectMask, Size(box.width, box.height));
            Mat mask = (objectMask > 0.5); // 二值化

            // 7. 渲染遮罩 (塗上顏色)
            Mat roi = frame(box);
            roi.setTo(Scalar(0, 255, 0), mask); // 將偵測到的物體塗成綠色
            rectangle(frame, box, Scalar(255, 0, 0), 2);
        }
    }

    imwrite("mask_output_cpp.jpg", frame);
    cout << "C++ Mask R-CNN 推論完成！" << endl;
    return 0;
}
```
![img](https://github.com/JrPhy/ComputerVision/blob/master/img/fasterRCNN.png)

當然效能也是輸給 Faster R-CNN 的，這就取決於要使用在什麼領域，如果只是計數可以用 Faster R-CNN 就好，如果要更精確地知道圖片是什麼則建議用 Mask R-CNN。

## 四、YOLO
全名 You Only Look Once，因其即時性所以獲得廣泛的應用。YOLO 的會將一張圖片切割成 S x S 個方格，每個方格以自己為中心點各自去判斷 B 個bounding boxes 中包含物體的 confidence score 跟種類，並算出 ```conf. score = Pr(Object) * IOU (groundtruth)```。如果該b-box的原生grid cell不含有物體，則理想conf. score應為0，否則理想conf. score的分數應和IOU相同。(最佳的情況是不含有物體則 Pr(Object) = 0 ；含有物體則 Pr(Object) = 1)。

![img](https://camo.githubusercontent.com/f2a5cce1c325ebaa7b96761fdf444025ca9743c97b3cc9195c22feb9225e8361/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313030302f312a34596350747176396d794a31767970676235774b6e412e706e67)
```c++
%%writefile yolo_inference.cpp
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::dnn;

int main() {
    // 1. 載入 ONNX 模型
    Net net = readNetFromONNX("yolov8n.onnx");
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // 2. 讀取與前處理 (Letterbox 縮放至 640x640)
    Mat frame = imread("bus.jpg");
    Mat blob = blobFromImage(frame, 1/255.0, Size(640, 640), Scalar(0,0,0), true, false);
    net.setInput(blob);

    // 3. 執行前向傳播
    Mat output = net.forward(); 
    // YOLOv8 輸出 shape 通常是 [1, 84, 8400]
    
    // 4. 解析 Output (這部分涉及座標與信心度提取)
    Mat detection(84, 8400, CV_32F, output.ptr<float>());
    detection = detection.t(); // 轉置為 [8400, 84]

    for (int i = 0; i < detection.rows; i++) {
        float confidence = detection.at<float>(i, 4); // 取得信心值
        if (confidence > 0.5) {
            // 提取中心點與寬高 (cx, cy, w, h)
            float cx = detection.at<float>(i, 0);
            float cy = detection.at<float>(i, 1);
            float w  = detection.at<float>(i, 2);
            float h  = detection.at<float>(i, 3);
            
            // 轉換為左上角座標並還原縮放比例...
            cout << "發現物件，信心度: " << confidence << endl;
        }
    }

    cout << "C++ 推論完成！" << endl;
    return 0;
}
```
![img](https://github.com/JrPhy/ComputerVision/blob/master/img/yolo.png)

YOLO 目前有許多版本，不過截至 2025 年整體考量最多的還是 YOLOv8，

## 五、模型評估
在實際應用時資料會被分成兩個維度，輸入資料中分成正確跟錯誤兩個，輸出結果也會被分成正確跟錯誤兩個，所以可以寫成下方矩陣

|   | 診斷有 | 診斷沒有 | 
| --- | --- | --- | 
| 有問題 | TP | FN |
| 沒問題 | FP | TN |

正確：有問題且被診斷有問題 TP，沒問題且被診斷沒有 TN\
錯誤：有問題且被診斷成沒問題 FN(偽陰)，沒問題且診斷成有問題 FP(偽陽)

在統計學上更喜歡用虛無假說(Null hypothesis) 和 對立假說(Alternative hypothesis) 表示
|   | 診斷有 | 診斷沒有 | 
| --- | --- | --- | 
| 有問題 | 1-α | β |
| 沒問題 | α | 1-β |

$$\ 正確率 Accuracy = \frac{TP+TN}{TP+TN+FP+FN} ，精確率 Precision = \frac{TP}{TP+FP} ， 召回率 recall = \frac{TP}{TP+FN} $$

在工業及醫療檢測情況中有問題的母數 (TP+FN) 是很少的，也就是不平衡的資料，使得正確率會非常高，畢竟沒問題的母數多，且只要能判斷正確即可達到很高的正確率與精確率，所以會更關心召回率，也就是***有問題且真的能被模型檢測出有問題***的結果比例是多高，所以模型會希望是 Precision 與 recall 都接近 1，就可以畫成以下圖

![img](https://ask.qcloudimg.com/http-save/7236395/k7u3m5kziu.png)
