# BaiduDogCLS
百度狗分类

初赛 线上error: 0.1712
决赛 线上error: 0.1952


能work的方法:

* centerloss 稳定训练，提高0.1左右
* 集成模型(inception-v4、resnext50)两种模型，提升0.03左右
* pseudo-label，提升0.09左右，时间不够，不能多次迭代训练，否则这个提升还能更多
* 10-crop,单模型有1%左右浮动

试了没调成功的方法:

* crop（CAM可视化以及crop出狗，排除干扰）
* bilinear pooling
* xgboost & 提特征融合
* boosted bilinear pooling CNN（ECCV16')
* weighted softmax（增加hard类的权重）