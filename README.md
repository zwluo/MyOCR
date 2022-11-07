# MyOCR
深度学习文字识别

本项目只是做一个整合，不是原创，原先的代码分散在三个项目里面，且结构复杂

项目使用的模型

VGG-BiLSTM-CTC

## 执行步骤

1.generate_char.py，搜索txt可以替换字符字典

2.convert.py

3.create_lmdb_dataset.py

4.替换lmdb文件

5.替换trian.py里面的中文表character

5.开始训练，train.py 搜索RMSprop可以切换优化器

6.替换demo.py里面的character参数，和train.py里面的保持一致

7.测试模型，demo.py

## 从零开始参考教程
https://zhuanlan.zhihu.com/p/400270506

## 各种算法介绍
https://zhuanlan.zhihu.com/p/356842725?utm_id=0
https://www.cnblogs.com/skyfsm/category/1123384.html

## 参考项目

样本生成：https://github.com/AstarLight/CPS-OCR-Engine https://github.com/Belval/TextRecognitionDataGenerator

数据转换：https://github.com/DaveLogs/TRDG2DTRB https://github.com/clovaai/deep-text-recognition-benchmark

字符训练和识别：https://github.com/clovaai/deep-text-recognition-benchmark

## 说明
运行参数在代码文件的最上方

字体存放于fonts/cn目录，免费字体下载https://www.fonts.net.cn/commercial-free/fonts-zh/tag-fangzheng-1.html?previewText=%E6%96%B9%E6%AD%A3%E4%B9%A6%E5%AE%8B

文字样本存放于fonts/cn.txt文件

这个项目把文本识别模块化（特征提取-序列特征提取-特征转换-预测），使每一个模块可以单独优化，从而量化不同模块的贡献。

使用深度学习进行汉字识别，需要大量的样本用于学习，本质上和模板匹配没有区别，最好还是

## 简化代码

train.py和demo.py里面的--output_channel从512修改为256，训练时间可以缩短一半
feature_extraction.py里面的第22行和第29行的内容nn.MaxPool2d((2, 1), (2, 1)) 被修改为 nn.MaxPool2d(2, stride=2)，原先的是用于区分i和l字符，汉字用不到，就改了，也能快一点

## 运行环境

windows11，cpu，16G内存，PyCharm

python版本：3.9.2

## 当前系统安装包列表

absl-py                      1.0.0
addict                       2.4.0
arabic-reshaper              2.1.4
astunparse                   1.6.3
attrdict                     2.0.1
attrs                        20.3.0
Automat                      20.2.0
Babel                        2.10.3
basicsr                      1.3.4.9
bce-python-sdk               0.8.74
beautifulsoup4               4.11.1
boto3                        1.17.27
botocore                     1.20.27
bs4                          0.0.1
cachetools                   4.2.4
certifi                      2020.12.5
cffi                         1.14.5
chardet                      4.0.0
click                        8.1.3
colorama                     0.4.4
common                       0.1.2
constantly                   15.1.0
cryptography                 3.4.6
cssselect                    1.1.0
cssutils                     2.6.0
cycler                       0.11.0
Cython                       0.29.32
data                         0.4
decorator                    4.4.2
diffimg                      0.3.0
dual                         0.0.10
dynamo3                      0.4.10
easyocr                      1.6.2
et-xmlfile                   1.1.0
facexlib                     0.2.1.1
filterpy                     1.4.5
fire                         0.4.0
Flask                        2.2.2
Flask-Babel                  2.0.0
flatbuffers                  22.9.24
flywheel                     0.5.4
fonttools                    4.28.4
funcsigs                     1.0.2
future                       0.18.2
gast                         0.4.0
gfpgan                       0.2.4          d:\workspace\gfpgan-master
google-auth                  2.3.3
google-auth-oauthlib         0.4.6
google-pasta                 0.2.0
graphviz                     0.16
grpcio                       1.42.0
h5py                         3.7.0
hyperlink                    21.0.0
idna                         2.10
imageio                      2.13.3
imgaug                       0.4.0
importlib-metadata           4.8.2
incremental                  21.3.0
itemadapter                  0.2.0
itemloaders                  1.0.4
itsdangerous                 2.1.2
jarowinkler                  1.2.1
jieba                        0.42.1
Jinja2                       3.1.2
jmespath                     0.10.0
joblib                       1.2.0
keras                        2.10.0
Keras-Preprocessing          1.1.2
kiwisolver                   1.3.2
libclang                     14.0.6
llvmlite                     0.37.0
lmdb                         1.2.1
lxml                         4.6.2
Markdown                     3.3.6
MarkupSafe                   2.1.1
matplotlib                   3.5.1
natsort                      8.2.0
networkx                     2.6.3
ninja                        1.10.2.4
nltk                         3.7
numba                        0.54.1
numpy                        1.20.1
oauthlib                     3.1.1
objgraph                     3.5.0
opencv-contrib-python        4.6.0.66
opencv-python                4.5.4.60
opencv-python-headless       4.5.4.60
openpyxl                     3.0.10
opt-einsum                   3.3.0
packaging                    21.3
paddleocr                    2.6.0.1
paddlepaddle-tiny            1.6.1
pandas                       1.4.2
parsel                       1.6.0
Pillow                       9.2.0
pip                          22.3
premailer                    3.10.0
Protego                      0.1.16
protobuf                     3.15.6
pyasn1                       0.4.8
pyasn1-modules               0.2.8
pyclipper                    1.3.0.post3
pycparser                    2.20
pycryptodome                 3.15.0
PyDispatcher                 2.0.5
PyNLPIR                      0.6.0
pyOpenSSL                    20.0.1
pyparsing                    3.0.6
python-bidi                  0.4.2
python-dateutil              2.8.1
pytz                         2022.1
PyWavelets                   1.2.0
PyYAML                       5.4.1
queuelib                     1.5.0
rapidfuzz                    2.8.0
realesrgan                   0.2.3.0
regex                        2022.9.13
requests                     2.25.1
requests-oauthlib            1.3.0
rsa                          4.8
s3transfer                   0.3.4
scikit-image                 0.19.1
scipy                        1.7.3
Scrapy                       2.4.1
service-identity             18.1.0
setuptools                   49.2.1
Shapely                      1.8.4
six                          1.15.0
soupsieve                    2.3.2.post1
tb-nightly                   2.8.0a20211215
tensorboard                  2.10.1
tensorboard-data-server      0.6.1
tensorboard-plugin-wit       1.8.0
tensorflow                   2.10.0
tensorflow-estimator         2.10.0
tensorflow-io-gcs-filesystem 0.27.0
termcolor                    2.0.1
tf-slim                      1.1.0
tifffile                     2021.11.2
tight                        0.1.0
torch                        1.10.1
torchvision                  0.11.2
tqdm                         4.62.3
Twisted                      21.2.0
twisted-iocpsupport          1.0.1
typing_extensions            4.0.1
urllib3                      1.26.3
visualdl                     2.4.0
w3lib                        1.22.0
Werkzeug                     2.2.2
wheel                        0.37.0
wikipedia                    1.4.0
wrapt                        1.14.1
yapf                         0.31.0
zipp                         3.6.0
zope.interface               5.2.0
