# 预防伪造视频诈骗:基于PaddleGAN实现精准唇形合成

**注:本项目于AI Studio实现**

# 一、项目背景

现如今人工智能技术的不断发展，极大地促进了社会各方面生产力的发展，但是同时也会有不法分子使用人工智能技术来进行诈骗等违法行为。

而这种诈骗方式还比较新，因此还没有被很好地防范，因此本项目便**基于PaddleGAN实现唇部合成来达到伪造视频的效果**，希望能让公众对这种新型的诈骗手段有所警觉。

本教程是基于**PaddleGAN**实现的视频唇形同步模型**Wav2lip**, 它实现了人物口型与输入语音同步，俗称「对口型」。 

而Wav2lip不仅可以让静态图片说话，还可以将**视频进行唇形转换**，输出与目标语音相匹配的视频，而这也是给犯罪者可乘之机的地方，犯罪者可以利用含有一个人的视频和语音来**伪造出一个可以以假乱真的视频**，让被伪造者对犯罪者“言听计从”，想让他说什么就说什么，而这也是本项目想要尽力避免的。

本项目包含以下几个步骤：
* **Wav2lip原理讲解**
* **下载PaddleGAN代码**
* **唇形动作合成命令使用说明**
* **效果演示**

# Wav2lip模型原理
Wav2lip实现唇形与语音精准同步突破的关键在于，它采用了**唇形同步判别器**，以强制生成器持续产生准确而逼真的唇部运动。

此外，该研究通过在鉴别器中，使用**多个连续帧**而不是单个帧，并使用**视觉质量损失**（而不仅仅是对比损失）来考虑时间相关性，从而改善了视觉质量。

该wav2lip模型几乎是**万能**的，适用于任何人脸、任何语音、任何语言，对任意视频都能达到很高的准确率，**可以无缝地与原始视频融**合，还可以用于转换动画人脸，并且导入合成语音也是可行的

# 下载PaddleGAN代码


```python
# 切换到工作目录
%cd /home/aistudio/work
```

    /home/aistudio/work



```python
# 从gitee上下载PaddleGAN源码
!git clone https://gitee.com/PaddlePaddle/PaddleGAN
```

    fatal: destination path 'PaddleGAN' already exists and is not an empty directory.



```python
# 安装paddlepaddle
!pip install ppgan
```

```python
# 切换到PaddleGAN目录下
%cd /home/aistudio/work/PaddleGAN
# 安装相关需要的包
!pip install -r requirements.txt
# 切换到对应文件夹
%cd applications/tools
```


# 唇形动作合成命令使用说明

仿照本项目的步骤,你可以复现本项目的内容,只需要另外准备合适的视频和音频。

只需要在如下命令中修改**face参数**和**audio参数**分别为你的视频和音频的文件位置并运行程序，便可以在**outfile 参数**指定的目标位置生成结果的mp4文件。

参数具体说明如下：
* face:原始视频的位置,视频中的人物的唇形将根据音频进行唇形合成
* audio：驱动唇形合成的音频，视频中的人物将根据此音频进行唇形合成
* outfile: 处理完之后的结果文件的位置
* face_det_batch_size: 视频使用的batch_size,这里不能使用默认值,否则会超显存
* wav2lip_batch_size: 音频使用的batch_size,这里同样不能使用默认值,否则会超显存

本项目使用准备好的视频中说的话是"众所周知,视频是不能P的,但是时代变了,现在这段视频就是P出来的,我从来没有讲过这段话,在个人信息泄漏越来越严重的今天,你的形象与声音就可能像我一样被伪造,说一些你从来没有讲过的话."

而音频文件中的声音是“为了防止犯罪分子利用这种手段进行诈骗，我们有必要提高防范意识，提醒身边的亲朋好友这种新型诈骗的手段，让不法分子无机可乘。”


```python
!python wav2lip.py --face /home/aistudio/video.mp4 --audio /home/aistudio/audio.m4a --outfile /home/aistudio/output.mp4 --face_det_batch_size 8 --wav2lip_batch_size 64
```

# 效果演示

输入的原视频是这样子的:

![](https://media1.giphy.com/media/9z5gHZrPnq96SVleA7/giphy.gif?cid=790b76115f6f97aa90c3f8c1730295206a21091213ca55b9&rid=giphy.gif&ct=g)

而处理之后的结果是这样子的:

![](https://media3.giphy.com/media/NOWEiGV3TOso7PdsIb/giphy.gif?cid=790b76115ac4490eb80211f70f832792e062dc35caf2fac6&rid=giphy.gif&ct=g)

# 总结

从上述结果可以看到,虽然结果还没有完全达到以假乱真的地步,但是已经可以看到如果针对这种伪造视频技术而开发的AI应用在这个领域会有什么样的效果了.

而不完全精确的原因便是因为wav2lip使用了面部定位技术,而这种技术虽然可以加快算法的处理速度,但是会导致网络本身并没有识别人脸位置的功能,因此当人脸定位有偏差时,视频就会出现崩坏的现象.

但因为wav2lip本身并不是为了这个用途而设计的,因此效果上不够出彩是合理的,但是如果有犯罪分子专门开发和训练用于伪造视频用途的GAN网络,效果将会远比上面所展示的效果要骇人听闻.

希望大家能提高防范意识,同时提醒还不知道的亲朋好友,不要让犯罪分子用伪造的声音和视频来获取非法收益。
