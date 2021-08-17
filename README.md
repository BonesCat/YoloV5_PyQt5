<<<<<<< HEAD
**更新2021.08.16**

**添加图片和视频保存功能：

1.图片和视频按照当前系统时间进行命名

2.各自检测结果存放入output文件夹

3.摄像头检测的默认设备序号更改为0，减少调试报错

温馨提示：

1.项目放置在全英文路径下，防止项目报错

2.默认使用cpu进行检测，自己可以在init中手动切换GPU（因为我的笔记本太老了）

3.当前的摄像头检测的存储有一点点问题，播放速度比较快，不知道是不是我用cpu检测，导致的帧率不匹配的问题（后面有时间在捣鼓捣鼓，我现在强制调慢了FPS😂）


## **一、项目简介**
使用PyQt5为YoloV5添加一个可视化检测界面，并实现简单的界面跳转，具体情况如下：

**博客与B站：**

博客地址：https://blog.csdn.net/wrh975373911/article/details/119322059?spm=1001.2014.3001.5501

B站视频：https://www.bilibili.com/video/BV1ZU4y1E7at

**特点：**
 1. UI界面与逻辑代码分离
 2. 支持自选定模型
 3. 同时输出检测结果与相应相关信息
 4. 支持图片，视频，摄像头检测
 5. 支持视频暂停与继续检测

**目的：**
 1. 熟悉QtDesign的使用
 2. 了解PyQt5基础控件与布局方法
 3. 了解界面跳转
 4. 了解信号与槽
 5. 熟悉视频在PyQt中的处理方法

**项目图片：**

![登录界面](https://img-blog.csdnimg.cn/541206b4f8324a2794978672f4b35b81.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dyaDk3NTM3MzkxMQ==,size_16,color_FFFFFF,t_70)
![注册界面](https://img-blog.csdnimg.cn/5ee90529650e41fb81065f17dcb40fc3.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dyaDk3NTM3MzkxMQ==,size_16,color_FFFFFF,t_70)

![检测界面](https://img-blog.csdnimg.cn/9161046e7c0744328152f0cdca8748d6.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dyaDk3NTM3MzkxMQ==,size_16,color_FFFFFF,t_70)


## **二、快速开始**
**环境与相关文件配置：**
 - 按照 ult-yolov5 中requirement的要求配置环境，自行安装PyQt5，注意都需要在一个evn环境中进行安装与配置
 - 下载或训练一个模型，将“.pt”文件放到weights文件夹，（权重文件可以自己选，程序默认打开weights文件夹）
 - 设置init中的opt

**两种程序使用方式：**

 - 直接运行detect_logical.py，进入检测界面
 - 运行main_logical.py，先登录，在进入检测界面

## **三、 参考与致谢**
 - 《PyQt5快速开发与实践》
 -  www.python3.vip
 - B站白月黑羽的PyQt教程 https://www.bilibili.com/video/BV1cJ411R7bP?from=search&seid=7706040462590056686
 - https://xugaoxiang.blog.csdn.net/article/details/118384430 从这个博主的博客中学到了很多知识，感觉博主，博主的代码框架也很好，也是本文代码是在其基础上进行学习和修改的
 - Github项目：YOLOv3GUI_Pytorch_PyQt5

## **四、 版权声明**
仅供交流学习使用，项目粗拙，勿商用，实际应用中出现的问题，个人不管哦~
=======

