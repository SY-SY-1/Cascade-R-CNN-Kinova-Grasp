# Cascade-R-CNN-Kinova-Grasp
基于Pytorch深度学习框架进行整体环境搭建，包括数据集制作，模型训练，模型测试，模型优化；基于kinova机器人搭建实际抓取环境；采用级联网络Cascade R-CNN提取特征。

一、针对机器人多物体抓取检测研究问题，选用Cascade R-CNN为基础网络框架，Cascade R-CNN是通用目标检测中表现较好的一种级联算法，其特点是速度快，检测精度高。

二、首先构建一个由三十二类对象组成的多目标抓取数据集( MOGD )。解决当前多物体抓取数据集较为缺乏的问题，并便于对多目标抓取检测模型进行评估。
    
三、由于平行夹持器的对称性，角度参数的取值不大于180°，我们的取向角θ被量化为R个部分( 每一部分相对于抓取矩形的中心进行划分 )。在我们的工作中，设R = 19。
    
四、多物体抓取数据集采用的标注方式与康奈尔抓取数据集基本一致，因此多物体检测与单物体检测方法相似，将抓取检测转化为目标定位加上角度分类问题，将各物体抓取框与水平方向的夹角转化成19个类别，包括一个背景类以提供负样本，采用Json格式存储训练和测试样本，对每一个标签都注明对应图像ID，抓取框的中心点坐标和宽高以及角度分类ID，这种存储格式符合COCO目标检测数据集规范。        
    检测图    
    ![Figure_21](https://user-images.githubusercontent.com/80105687/178427530-17239ecd-5e57-491a-b812-4c03f32a6501.png)
    ![Figure_22](https://user-images.githubusercontent.com/80105687/178427563-b3b0d0a8-197e-47b0-916b-e133d96ae49a.png)
    ![Figure_23](https://user-images.githubusercontent.com/80105687/178427574-f80dae13-1b1f-49e6-b9ce-7bda15b14085.png)

    抓取实物图    
    ![IMG_20210824_221425](https://user-images.githubusercontent.com/80105687/178427883-5f5306a5-7584-4e74-854a-81a48fa7ffab.jpg)
    ![IMG_20210824_221436](https://user-images.githubusercontent.com/80105687/178427917-4e40ecb1-2df8-4b2d-b00c-a2406880e156.jpg)
    ![IMG_20210824_221533](https://user-images.githubusercontent.com/80105687/178427941-f135846a-e942-4f9e-b2f9-f5d9d7649482.jpg)
