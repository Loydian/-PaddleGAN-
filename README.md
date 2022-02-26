# 预防伪造视频诈骗:基于PaddleGAN实现精准唇形合成

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

    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Collecting ppgan
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/2f/32/6d5c52cc076ae31c0bdd47844a9dfb8923f9ea4aa19602d615b5f715918f/ppgan-2.1.0-py3-none-any.whl (394 kB)
         |████████████████████████████████| 394 kB 11.0 MB/s            
    [?25hCollecting imageio==2.9.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/6e/57/5d899fae74c1752f52869b613a8210a2480e1a69688e65df6cb26117d45d/imageio-2.9.0-py3-none-any.whl (3.3 MB)
         |████████████████████████████████| 3.3 MB 11.4 MB/s            
    [?25hCollecting natsort
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a9/76/0f624b7326f4458a249580c55e5654756084ec4572ce37a05f799b96bc24/natsort-8.1.0-py3-none-any.whl (37 kB)
    Collecting munch
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/cc/ab/85d8da5c9a45e072301beb37ad7f833cd344e04c817d97e0cc75681d248f/munch-2.5.0-py2.py3-none-any.whl (10 kB)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan) (4.1.1.26)
    Collecting numba==0.53.1
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/bb/73/d9c127eddbe3c105a33379d425b88f9dca249a6eddf39ce886494d49c3f9/numba-0.53.1-cp37-cp37m-manylinux2014_x86_64.whl (3.4 MB)
         |████████████████████████████████| 3.4 MB 8.8 MB/s            
    [?25hRequirement already satisfied: imageio-ffmpeg in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan) (0.3.0)
    Collecting librosa==0.8.1
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/54/19/a0e2bdc94bc0d1555e4f9bc4099a0751da83fa6e1e6157ec005564f8a98a/librosa-0.8.1-py3-none-any.whl (203 kB)
         |████████████████████████████████| 203 kB 12.9 MB/s            
    [?25hRequirement already satisfied: PyYAML>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan) (5.1.2)
    Requirement already satisfied: scipy>=1.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan) (1.6.3)
    Collecting scikit-image>=0.14.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d2/d9/d16d4cbb4840e0fb3bd329b49184d240b82b649e1bd579489394fbc85c81/scikit_image-0.19.2-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (13.5 MB)
         |████████████████████████████████| 13.5 MB 12.0 MB/s            
    [?25hRequirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan) (2.2.3)
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan) (4.27.0)
    Requirement already satisfied: easydict in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan) (1.9)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from imageio==2.9.0->ppgan) (1.19.5)
    Requirement already satisfied: pillow in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from imageio==2.9.0->ppgan) (8.2.0)
    Requirement already satisfied: joblib>=0.14 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->ppgan) (0.14.1)
    Requirement already satisfied: resampy>=0.2.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->ppgan) (0.2.2)
    Requirement already satisfied: decorator>=3.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->ppgan) (4.4.2)
    Requirement already satisfied: audioread>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->ppgan) (2.1.8)
    Collecting pooch>=1.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/8d/64/8e1bfeda3ba0f267b2d9a918e8ca51db8652d0e1a3412a5b3dbce85d90b6/pooch-1.6.0-py3-none-any.whl (56 kB)
         |████████████████████████████████| 56 kB 6.9 MB/s             
    [?25hRequirement already satisfied: soundfile>=0.10.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->ppgan) (0.10.3.post1)
    Requirement already satisfied: packaging>=20.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->ppgan) (21.3)
    Requirement already satisfied: scikit-learn!=0.19.0,>=0.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->ppgan) (0.24.2)
    Collecting llvmlite<0.37,>=0.36.0rc1
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/54/25/2b4015e2b0c3be2efa6870cf2cf2bd969dd0e5f937476fc13c102209df32/llvmlite-0.36.0-cp37-cp37m-manylinux2010_x86_64.whl (25.3 MB)
         |████████████████████████████████| 25.3 MB 14.8 MB/s            
    [?25hRequirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from numba==0.53.1->ppgan) (56.2.0)
    Collecting PyWavelets>=1.1.1
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/a1/9c/564511b6e1c4e1d835ed2d146670436036960d09339a8fa2921fe42dad08/PyWavelets-1.2.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (6.1 MB)
         |████████████████████████████████| 6.1 MB 8.5 MB/s            
    [?25hCollecting tifffile>=2019.7.26
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/d8/38/85ae5ed77598ca90558c17a2f79ddaba33173b31cf8d8f545d34d9134f0d/tifffile-2021.11.2-py3-none-any.whl (178 kB)
         |████████████████████████████████| 178 kB 6.1 MB/s            
    [?25hRequirement already satisfied: networkx>=2.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->ppgan) (2.4)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->ppgan) (2.8.2)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->ppgan) (0.10.0)
    Requirement already satisfied: six>=1.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->ppgan) (1.16.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->ppgan) (1.1.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->ppgan) (2019.3)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->ppgan) (3.0.7)
    Requirement already satisfied: requests>=2.19.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pooch>=1.0->librosa==0.8.1->ppgan) (2.24.0)
    Collecting appdirs>=1.3.0
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3b/00/2344469e2084fb287c2e0b57b72910309874c3245463acd6cf5e3db69324/appdirs-1.4.4-py2.py3-none-any.whl (9.6 kB)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn!=0.19.0,>=0.14.0->librosa==0.8.1->ppgan) (2.1.0)
    Requirement already satisfied: cffi>=1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from soundfile>=0.10.2->librosa==0.8.1->ppgan) (1.15.0)
    Requirement already satisfied: pycparser in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from cffi>=1.0->soundfile>=0.10.2->librosa==0.8.1->ppgan) (2.21)
    Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.19.0->pooch>=1.0->librosa==0.8.1->ppgan) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.19.0->pooch>=1.0->librosa==0.8.1->ppgan) (1.25.6)
    Requirement already satisfied: idna<3,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.19.0->pooch>=1.0->librosa==0.8.1->ppgan) (2.8)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.19.0->pooch>=1.0->librosa==0.8.1->ppgan) (2019.9.11)
    Installing collected packages: llvmlite, numba, appdirs, tifffile, PyWavelets, pooch, imageio, scikit-image, natsort, munch, librosa, ppgan
      Attempting uninstall: llvmlite
        Found existing installation: llvmlite 0.31.0
        Uninstalling llvmlite-0.31.0:
          Successfully uninstalled llvmlite-0.31.0
      Attempting uninstall: numba
        Found existing installation: numba 0.48.0
        Uninstalling numba-0.48.0:
          Successfully uninstalled numba-0.48.0
      Attempting uninstall: imageio
        Found existing installation: imageio 2.6.1
        Uninstalling imageio-2.6.1:
          Successfully uninstalled imageio-2.6.1
      Attempting uninstall: librosa
        Found existing installation: librosa 0.7.2
        Uninstalling librosa-0.7.2:
          Successfully uninstalled librosa-0.7.2
    Successfully installed PyWavelets-1.2.0 appdirs-1.4.4 imageio-2.9.0 librosa-0.8.1 llvmlite-0.36.0 munch-2.5.0 natsort-8.1.0 numba-0.53.1 pooch-1.6.0 ppgan-2.1.0 scikit-image-0.19.2 tifffile-2021.11.2
    [33mWARNING: You are using pip version 21.3.1; however, version 22.0.3 is available.
    You should consider upgrading via the '/opt/conda/envs/python35-paddle120-env/bin/python -m pip install --upgrade pip' command.[0m



```python
# 切换到PaddleGAN目录下
%cd /home/aistudio/work/PaddleGAN
# 安装相关需要的包
!pip install -r requirements.txt
# 切换到对应文件夹
%cd applications/tools
```

    /home/aistudio/work/PaddleGAN
    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 1)) (4.27.0)
    Requirement already satisfied: PyYAML>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 2)) (5.1.2)
    Requirement already satisfied: scikit-image>=0.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 3)) (0.19.2)
    Requirement already satisfied: scipy>=1.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 4)) (1.6.3)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 5)) (4.1.1.26)
    Requirement already satisfied: imageio==2.9.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 6)) (2.9.0)
    Requirement already satisfied: imageio-ffmpeg in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 7)) (0.3.0)
    Requirement already satisfied: librosa==0.8.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 8)) (0.8.1)
    Requirement already satisfied: numba==0.53.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 9)) (0.53.1)
    Requirement already satisfied: easydict in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 10)) (1.9)
    Requirement already satisfied: munch in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 11)) (2.5.0)
    Requirement already satisfied: natsort in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 12)) (8.1.0)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from imageio==2.9.0->-r requirements.txt (line 6)) (1.19.5)
    Requirement already satisfied: pillow in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from imageio==2.9.0->-r requirements.txt (line 6)) (8.2.0)
    Requirement already satisfied: resampy>=0.2.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (0.2.2)
    Requirement already satisfied: audioread>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (2.1.8)
    Requirement already satisfied: soundfile>=0.10.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (0.10.3.post1)
    Requirement already satisfied: pooch>=1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (1.6.0)
    Requirement already satisfied: packaging>=20.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (21.3)
    Requirement already satisfied: decorator>=3.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (4.4.2)
    Requirement already satisfied: joblib>=0.14 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (0.14.1)
    Requirement already satisfied: scikit-learn!=0.19.0,>=0.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.8.1->-r requirements.txt (line 8)) (0.24.2)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from numba==0.53.1->-r requirements.txt (line 9)) (56.2.0)
    Requirement already satisfied: llvmlite<0.37,>=0.36.0rc1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from numba==0.53.1->-r requirements.txt (line 9)) (0.36.0)
    Requirement already satisfied: networkx>=2.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->-r requirements.txt (line 3)) (2.4)
    Requirement already satisfied: tifffile>=2019.7.26 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->-r requirements.txt (line 3)) (2021.11.2)
    Requirement already satisfied: PyWavelets>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->-r requirements.txt (line 3)) (1.2.0)
    Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from munch->-r requirements.txt (line 11)) (1.16.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from packaging>=20.0->librosa==0.8.1->-r requirements.txt (line 8)) (3.0.7)
    Requirement already satisfied: appdirs>=1.3.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pooch>=1.0->librosa==0.8.1->-r requirements.txt (line 8)) (1.4.4)
    Requirement already satisfied: requests>=2.19.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pooch>=1.0->librosa==0.8.1->-r requirements.txt (line 8)) (2.24.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn!=0.19.0,>=0.14.0->librosa==0.8.1->-r requirements.txt (line 8)) (2.1.0)
    Requirement already satisfied: cffi>=1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from soundfile>=0.10.2->librosa==0.8.1->-r requirements.txt (line 8)) (1.15.0)
    Requirement already satisfied: pycparser in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from cffi>=1.0->soundfile>=0.10.2->librosa==0.8.1->-r requirements.txt (line 8)) (2.21)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.19.0->pooch>=1.0->librosa==0.8.1->-r requirements.txt (line 8)) (2019.9.11)
    Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.19.0->pooch>=1.0->librosa==0.8.1->-r requirements.txt (line 8)) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.19.0->pooch>=1.0->librosa==0.8.1->-r requirements.txt (line 8)) (2.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.19.0->pooch>=1.0->librosa==0.8.1->-r requirements.txt (line 8)) (1.25.6)
    [33mWARNING: You are using pip version 21.3.1; however, version 22.0.3 is available.
    You should consider upgrading via the '/opt/conda/envs/python35-paddle120-env/bin/python -m pip install --upgrade pip' command.[0m
    /home/aistudio/work/PaddleGAN/applications/tools


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

    Reading video frames...
    Number of frames available for inference: 473
    Extracting raw audio...
    ffmpeg version 2.8.15-0ubuntu0.16.04.1 Copyright (c) 2000-2018 the FFmpeg developers
      built with gcc 5.4.0 (Ubuntu 5.4.0-6ubuntu1~16.04.10) 20160609
      configuration: --prefix=/usr --extra-version=0ubuntu0.16.04.1 --build-suffix=-ffmpeg --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --cc=cc --cxx=g++ --enable-gpl --enable-shared --disable-stripping --disable-decoder=libopenjpeg --disable-decoder=libschroedinger --enable-avresample --enable-avisynth --enable-gnutls --enable-ladspa --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libmodplug --enable-libmp3lame --enable-libopenjpeg --enable-libopus --enable-libpulse --enable-librtmp --enable-libschroedinger --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvorbis --enable-libvpx --enable-libwavpack --enable-libwebp --enable-libx265 --enable-libxvid --enable-libzvbi --enable-openal --enable-opengl --enable-x11grab --enable-libdc1394 --enable-libiec61883 --enable-libzmq --enable-frei0r --enable-libx264 --enable-libopencv
      libavutil      54. 31.100 / 54. 31.100
      libavcodec     56. 60.100 / 56. 60.100
      libavformat    56. 40.101 / 56. 40.101
      libavdevice    56.  4.100 / 56.  4.100
      libavfilter     5. 40.101 /  5. 40.101
      libavresample   2.  1.  0 /  2.  1.  0
      libswscale      3.  1.101 /  3.  1.101
      libswresample   1.  2.101 /  1.  2.101
      libpostproc    53.  3.100 / 53.  3.100
    Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '/home/aistudio/audio.m4a':
      Metadata:
        major_brand     : mp42
        minor_version   : 0
        compatible_brands: isommp42
        creation_time   : 2022-02-19 14:13:20
      Duration: 00:00:13.40, start: 0.000000, bitrate: 152 kb/s
        Stream #0:0(eng): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 148 kb/s (default)
        Metadata:
          creation_time   : 2022-02-19 14:13:20
          handler_name    : SoundHandle
    Output #0, wav, to 'temp/temp.wav':
      Metadata:
        major_brand     : mp42
        minor_version   : 0
        compatible_brands: isommp42
        ISFT            : Lavf56.40.101
        Stream #0:0(eng): Audio: pcm_s16le ([1][0][0][0] / 0x0001), 48000 Hz, stereo, s16, 1536 kb/s (default)
        Metadata:
          creation_time   : 2022-02-19 14:13:20
          handler_name    : SoundHandle
          encoder         : Lavc56.60.100 pcm_s16le
    Stream mapping:
      Stream #0:0 -> #0:0 (aac (native) -> pcm_s16le (native))
    Press [q] to stop, [?] for help
    size=    2512kB time=00:00:13.39 bitrate=1536.0kbits/s    
    video:0kB audio:2512kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.003032%
    Length of mel chunks: 397
    W0219 22:48:18.216428  2498 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
    W0219 22:48:18.220994  2498 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    Model loaded
      0%|                                                     | 0/7 [00:00<?, ?it/s]
      0%|                                                    | 0/50 [00:00<?, ?it/s][A
      2%|▉                                           | 1/50 [00:02<02:11,  2.68s/it][A
      4%|█▊                                          | 2/50 [00:05<02:07,  2.65s/it][A
      6%|██▋                                         | 3/50 [00:07<02:02,  2.61s/it][A
      8%|███▌                                        | 4/50 [00:10<01:57,  2.56s/it][A
     10%|████▍                                       | 5/50 [00:12<01:53,  2.51s/it][A
     12%|█████▎                                      | 6/50 [00:15<01:49,  2.48s/it][A
     14%|██████▏                                     | 7/50 [00:17<01:44,  2.43s/it][A
     16%|███████                                     | 8/50 [00:19<01:43,  2.47s/it][A
     18%|███████▉                                    | 9/50 [00:22<01:41,  2.47s/it][A
     20%|████████▌                                  | 10/50 [00:24<01:39,  2.49s/it][A
     22%|█████████▍                                 | 11/50 [00:27<01:40,  2.58s/it][A
     24%|██████████▎                                | 12/50 [00:30<01:42,  2.70s/it][A
     26%|███████████▏                               | 13/50 [00:33<01:41,  2.74s/it][A
     28%|████████████                               | 14/50 [00:35<01:35,  2.65s/it][A
     30%|████████████▉                              | 15/50 [00:38<01:29,  2.55s/it][A
     32%|█████████████▊                             | 16/50 [00:40<01:26,  2.55s/it][A
     34%|██████████████▌                            | 17/50 [00:43<01:22,  2.49s/it][A
     36%|███████████████▍                           | 18/50 [00:45<01:17,  2.41s/it][A
     38%|████████████████▎                          | 19/50 [00:47<01:14,  2.39s/it][A
     40%|█████████████████▏                         | 20/50 [00:50<01:11,  2.37s/it][A
     42%|██████████████████                         | 21/50 [00:52<01:08,  2.35s/it][A
     44%|██████████████████▉                        | 22/50 [00:54<01:04,  2.31s/it][A
     46%|███████████████████▊                       | 23/50 [00:56<01:02,  2.31s/it][A
     48%|████████████████████▋                      | 24/50 [00:59<01:00,  2.32s/it][A
     50%|█████████████████████▌                     | 25/50 [01:01<00:59,  2.38s/it][A
     52%|██████████████████████▎                    | 26/50 [01:04<00:57,  2.40s/it][A
     54%|███████████████████████▏                   | 27/50 [01:06<00:55,  2.40s/it][A
     56%|████████████████████████                   | 28/50 [01:09<00:52,  2.41s/it][A
     58%|████████████████████████▉                  | 29/50 [01:11<00:50,  2.40s/it][A
     60%|█████████████████████████▊                 | 30/50 [01:13<00:48,  2.42s/it][A
     62%|██████████████████████████▋                | 31/50 [01:16<00:46,  2.43s/it][A
     64%|███████████████████████████▌               | 32/50 [01:18<00:43,  2.43s/it][A
     66%|████████████████████████████▍              | 33/50 [01:21<00:41,  2.43s/it][A
     68%|█████████████████████████████▏             | 34/50 [01:23<00:39,  2.50s/it][A
     70%|██████████████████████████████             | 35/50 [01:26<00:38,  2.54s/it][A
     72%|██████████████████████████████▉            | 36/50 [01:29<00:36,  2.61s/it][A
     74%|███████████████████████████████▊           | 37/50 [01:31<00:33,  2.60s/it][A
     76%|████████████████████████████████▋          | 38/50 [01:34<00:31,  2.62s/it][A
     78%|█████████████████████████████████▌         | 39/50 [01:37<00:29,  2.70s/it][A
     80%|██████████████████████████████████▍        | 40/50 [01:40<00:27,  2.77s/it][A
     82%|███████████████████████████████████▎       | 41/50 [01:43<00:24,  2.75s/it][A
     84%|████████████████████████████████████       | 42/50 [01:45<00:22,  2.75s/it][A
     86%|████████████████████████████████████▉      | 43/50 [01:48<00:18,  2.67s/it][A
     88%|█████████████████████████████████████▊     | 44/50 [01:51<00:16,  2.73s/it][A
     90%|██████████████████████████████████████▋    | 45/50 [01:53<00:13,  2.70s/it][A
     92%|███████████████████████████████████████▌   | 46/50 [01:56<00:10,  2.67s/it][A
     94%|████████████████████████████████████████▍  | 47/50 [01:59<00:08,  2.76s/it][A
     96%|█████████████████████████████████████████▎ | 48/50 [02:01<00:05,  2.72s/it][A
     98%|██████████████████████████████████████████▏| 49/50 [02:04<00:02,  2.64s/it][A
    100%|███████████████████████████████████████████| 50/50 [02:05<00:00,  2.32s/it][A
    100%|█████████████████████████████████████████████| 7/7 [02:13<00:00, 15.78s/it]
    ffmpeg version 2.8.15-0ubuntu0.16.04.1 Copyright (c) 2000-2018 the FFmpeg developers
      built with gcc 5.4.0 (Ubuntu 5.4.0-6ubuntu1~16.04.10) 20160609
      configuration: --prefix=/usr --extra-version=0ubuntu0.16.04.1 --build-suffix=-ffmpeg --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --cc=cc --cxx=g++ --enable-gpl --enable-shared --disable-stripping --disable-decoder=libopenjpeg --disable-decoder=libschroedinger --enable-avresample --enable-avisynth --enable-gnutls --enable-ladspa --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libmodplug --enable-libmp3lame --enable-libopenjpeg --enable-libopus --enable-libpulse --enable-librtmp --enable-libschroedinger --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvorbis --enable-libvpx --enable-libwavpack --enable-libwebp --enable-libx265 --enable-libxvid --enable-libzvbi --enable-openal --enable-opengl --enable-x11grab --enable-libdc1394 --enable-libiec61883 --enable-libzmq --enable-frei0r --enable-libx264 --enable-libopencv
      libavutil      54. 31.100 / 54. 31.100
      libavcodec     56. 60.100 / 56. 60.100
      libavformat    56. 40.101 / 56. 40.101
      libavdevice    56.  4.100 / 56.  4.100
      libavfilter     5. 40.101 /  5. 40.101
      libavresample   2.  1.  0 /  2.  1.  0
      libswscale      3.  1.101 /  3.  1.101
      libswresample   1.  2.101 /  1.  2.101
      libpostproc    53.  3.100 / 53.  3.100
    [0;33mGuessed Channel Layout for  Input Stream #0.0 : stereo
    [0mInput #0, wav, from 'temp/temp.wav':
      Metadata:
        encoder         : Lavf56.40.101
      Duration: 00:00:13.40, bitrate: 1536 kb/s
        Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 48000 Hz, 2 channels, s16, 1536 kb/s
    Input #1, avi, from 'temp/result.avi':
      Metadata:
        encoder         : Lavf58.31.101
      Duration: 00:00:13.25, start: 0.000000, bitrate: 7494 kb/s
        Stream #1:0: Video: mpeg4 (Simple Profile) (DIVX / 0x58564944), yuv420p, 1920x1080 [SAR 1:1 DAR 16:9], 7500 kb/s, 29.96 fps, 29.97 tbr, 29.96 tbn, 7491 tbc
    [1;36m[libx264 @ 0x1e84ee0] [0m[0;33m-qscale is ignored, -crf is recommended.
    [0m[1;36m[libx264 @ 0x1e84ee0] [0musing SAR=1/1
    [1;36m[libx264 @ 0x1e84ee0] [0musing cpu capabilities: MMX2 SSE2Fast SSSE3 SSE4.2 AVX FMA3 AVX2 LZCNT BMI2
    [1;36m[libx264 @ 0x1e84ee0] [0mprofile High, level 4.0
    [1;36m[libx264 @ 0x1e84ee0] [0m264 - core 148 r2643 5c65704 - H.264/MPEG-4 AVC codec - Copyleft 2003-2015 - http://www.videolan.org/x264.html - options: cabac=1 ref=3 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 threads=34 lookahead_threads=5 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=2 keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=crf mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
    Output #0, mp4, to '/home/aistudio/output.mp4':
      Metadata:
        encoder         : Lavf56.40.101
        Stream #0:0: Video: h264 (libx264) ([33][0][0][0] / 0x0021), yuv420p, 1920x1080 [SAR 1:1 DAR 16:9], q=-1--1, 29.97 fps, 30k tbn, 29.97 tbc
        Metadata:
          encoder         : Lavc56.60.100 libx264
        Stream #0:1: Audio: aac ([64][0][0][0] / 0x0040), 48000 Hz, stereo, fltp, 128 kb/s
        Metadata:
          encoder         : Lavc56.60.100 aac
    Stream mapping:
      Stream #1:0 -> #0:0 (mpeg4 (native) -> h264 (libx264))
      Stream #0:0 -> #0:1 (pcm_s16le (native) -> aac (native))
    Press [q] to stop, [?] for help
    frame=  397 fps=9.5 q=-1.0 Lsize=    5954kB time=00:00:13.39 bitrate=3640.9kbits/s    
    video:5724kB audio:216kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.243896%
    [1;36m[libx264 @ 0x1e84ee0] [0mframe I:2     Avg QP:19.10  size: 55733
    [1;36m[libx264 @ 0x1e84ee0] [0mframe P:249   Avg QP:21.49  size: 19187
    [1;36m[libx264 @ 0x1e84ee0] [0mframe B:146   Avg QP:24.19  size:  6654
    [1;36m[libx264 @ 0x1e84ee0] [0mconsecutive B-frames: 45.6% 15.6%  1.5% 37.3%
    [1;36m[libx264 @ 0x1e84ee0] [0mmb I  I16..4: 34.5% 62.5%  3.0%
    [1;36m[libx264 @ 0x1e84ee0] [0mmb P  I16..4:  2.8%  9.2%  0.0%  P16..4: 38.9%  5.8%  3.0%  0.0%  0.0%    skip:40.3%
    [1;36m[libx264 @ 0x1e84ee0] [0mmb B  I16..4:  0.5%  1.8%  0.0%  B16..8: 34.4%  1.5%  0.1%  direct: 1.3%  skip:60.3%  L0:52.2% L1:45.3% BI: 2.5%
    [1;36m[libx264 @ 0x1e84ee0] [0m8x8 transform intra:76.0% inter:85.2%
    [1;36m[libx264 @ 0x1e84ee0] [0mcoded y,uvDC,uvAC intra: 31.7% 34.8% 0.9% inter: 8.0% 8.1% 0.0%
    [1;36m[libx264 @ 0x1e84ee0] [0mi16 v,h,dc,p: 40% 24% 23% 13%
    [1;36m[libx264 @ 0x1e84ee0] [0mi8 v,h,dc,ddl,ddr,vr,hd,vl,hu: 25% 19% 48%  2%  1%  1%  1%  1%  1%
    [1;36m[libx264 @ 0x1e84ee0] [0mi4 v,h,dc,ddl,ddr,vr,hd,vl,hu: 26% 21% 13%  5%  9%  6% 10%  4%  4%
    [1;36m[libx264 @ 0x1e84ee0] [0mi8c dc,h,v,p: 56% 22% 21%  2%
    [1;36m[libx264 @ 0x1e84ee0] [0mWeighted P-Frames: Y:0.0% UV:0.0%
    [1;36m[libx264 @ 0x1e84ee0] [0mref P L0: 69.9%  7.5% 16.5%  6.1%
    [1;36m[libx264 @ 0x1e84ee0] [0mref B L0: 84.3% 13.1%  2.6%
    [1;36m[libx264 @ 0x1e84ee0] [0mref B L1: 95.9%  4.1%
    [1;36m[libx264 @ 0x1e84ee0] [0mkb/s:3539.30
    [0m

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
