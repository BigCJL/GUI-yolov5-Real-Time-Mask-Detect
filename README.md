BigCJL


使用yolov5训练的一个人脸口罩检测模型，并将模型导出为onnx文件，使用opencv的dnn模块读取模型并进行实时推理，推理速度与你的cpu性能有关。
提供了图形化的界面，可以选择图片、视频、实时检测三种功能。

![image](https://user-images.githubusercontent.com/79361803/133790512-1f8df5d1-e5cb-40c3-8f77-200d5503df41.png)


食用步骤：
1. 下载.onnx模型文件，这是我已经训练好的口罩检测模型，放到代码的相同路径下，当然你也可以在代码里自定义路径
链接：https://pan.baidu.com/s/1G9nN1_BF7XGulMOV1VtqGw 
提取码：bohd



2. 运行RealTime_Detect.py, 会调用电脑前置摄像头，并弹出窗口对当前视频实时检测，记得打开你的摄像头权限。

运行GUI_Detece.py，打开一个图形化界面，提供了图片、视频、实时摄像头三种检测功能。若选择视频或者图片检测，检测结果会自动保存到当前路径下。

博客：https://blog.csdn.net/m0_56155877/article/details/119280146?spm=1001.2014.3001.5501
