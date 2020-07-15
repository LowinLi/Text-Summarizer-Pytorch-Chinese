# Text-Summarizer-Pytorch-Chinese
![Python application](https://github.com/LowinLi/Text-Summarizer-Pytorch-Chinese/workflows/Python%20application/badge.svg)
+ 提供一款中文版生成式摘要服务。
+ 提供从数据到训练到部署，完整流程参考。
## 初衷
由于工作需要，在开源社区寻找汉语生成摘要模型时，几乎找不到可用的开源项目。

本项目在英文生成式摘要开源项目[Text-Summarizer-Pytorch](https://github.com/rohithreddy024/Text-Summarizer-Pytorch)基础上（指针生成网络），结合jieba分词，在数据集[LCSTS](http://icrc.hitsz.edu.cn/Article/show/139.html)上跑通一遍训练流程，中间自然踩过了很多坑，完整代码在这里开源出来供大家参考。

这里包括下载已经训练好的模型，部署服务，也包括借鉴代码完整跑一边训练流程，做为baseline使用。
## 效果
测试集指标：
|  集合   | 验证集  | 测试集|
|  ----  | ----   |----  |
| ROUGE-1  | 0.3553 |0.3396 |
| ROUGE-2  | 0.1843 |0.1668 |
| ROUGE-L  | 0.3481 |0.3320 |

该模型没有经过细致优化，只是完整跑了一遍流程，仅供参考。
[case请移步readme最下方](#摘要效果示例)

## 搭建服务

+ 已训练好模型：
链接: https://pan.baidu.com/s/1NKMIAsaE8H7GiCpP7Jovig 提取码: d7pr
+ 对应字典：
链接: https://pan.baidu.com/s/1A3vzYYYenu7vfNQgRX9NHA 提取码: 8ti6

把下载的两份文件放在根目录下。
+ 部署：
```bash
sudo docker-compose up
```
+ 测试：
```bash
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"text":"e公司讯，龙力退（002604）7月14日晚间公告，公司股票已被深交所决定终止上市，并在退市整理期交易30个交易日，最后交易日为7月14日，将在2020年7月15日被摘牌。公司股票终止上市后，将进
入股转系统进行股份转让。"' http://localhost:5000/abstract
```

## 训练模型

#### 下载PreLCSTS数据集
链接: https://pan.baidu.com/s/172CvApckpZu602hr6nwgAA 提取码: wb3s
1. 下载好的文件夹放在根目录下
2. 预处理数据
```bash
pip install requirements.txt
python make_data_files.py
```
3. 开始训练
```bash
sh train.sh
```
4. 评测验证集
```bash
sh eval.sh
```
5. 选出效果最好的模型,改shell脚本，进行再训练
```bash
sh train_lr.sh
```
6. 选出效果最好的模型，改shell脚本，进行测试
```bash
sh test.sh
```

## 训练中间过程
#### 训练时损失函数降低
```
2020-07-04 12:58:38,434 - data_util.log - INFO - Bucket queue size: 0, Input queue size: 0
2020-07-04 12:59:38,499 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 65600
2020-07-04 13:00:30,526 - data_util.log - INFO - iter:50  mle_loss:6.481  reward:0.0000
2020-07-04 13:00:38,552 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 300000
2020-07-04 13:01:07,877 - data_util.log - INFO - iter:100  mle_loss:6.031  reward:0.0000
2020-07-04 13:01:38,612 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 300000
2020-07-04 13:01:46,061 - data_util.log - INFO - iter:150  mle_loss:5.883  reward:0.0000
2020-07-04 13:02:24,137 - data_util.log - INFO - iter:200  mle_loss:5.790  reward:0.0000
2020-07-04 13:02:38,620 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 300000
2020-07-04 13:03:03,381 - data_util.log - INFO - iter:250  mle_loss:5.740  reward:0.0000
2020-07-04 13:03:38,633 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 300000
2020-07-04 13:03:42,141 - data_util.log - INFO - iter:300  mle_loss:5.690  reward:0.0000
2020-07-04 13:04:20,430 - data_util.log - INFO - iter:350  mle_loss:5.623  reward:0.0000
2020-07-04 13:04:38,692 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 300000
2020-07-04 13:04:59,155 - data_util.log - INFO - iter:400  mle_loss:5.592  reward:0.0000
2020-07-04 13:05:37,330 - data_util.log - INFO - iter:450  mle_loss:5.531  reward:0.0000
2020-07-04 13:05:38,752 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 300000
2020-07-04 13:06:16,069 - data_util.log - INFO - iter:500  mle_loss:5.473  reward:0.0000
2020-07-04 13:06:38,812 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 300000
2020-07-04 13:06:55,706 - data_util.log - INFO - iter:550  mle_loss:5.459  reward:0.0000
2020-07-04 13:07:33,658 - data_util.log - INFO - iter:600  mle_loss:5.366  reward:0.0000
2020-07-04 13:07:38,873 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 300000
...
2020-07-06 09:24:01,484 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 299732
2020-07-06 09:24:04,571 - data_util.log - INFO - iter:206800  mle_loss:2.631  reward:0.0000
2020-07-06 09:24:42,961 - data_util.log - INFO - iter:206850  mle_loss:2.639  reward:0.0000
2020-07-06 09:25:01,511 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 300000
2020-07-06 09:25:20,772 - data_util.log - INFO - iter:206900  mle_loss:2.653  reward:0.0000
2020-07-06 09:26:00,946 - data_util.log - INFO - iter:206950  mle_loss:2.657  reward:0.0000
2020-07-06 09:26:01,571 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 300000
2020-07-06 09:26:38,844 - data_util.log - INFO - iter:207000  mle_loss:2.666  reward:0.0000
2020-07-06 09:27:01,600 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 300000
2020-07-06 09:27:16,921 - data_util.log - INFO - iter:207050  mle_loss:2.634  reward:0.0000
2020-07-06 09:27:54,971 - data_util.log - INFO - iter:207100  mle_loss:2.658  reward:0.0000
2020-07-06 09:28:01,661 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 300000
2020-07-06 09:28:32,825 - data_util.log - INFO - iter:207150  mle_loss:2.620  reward:0.0000
2020-07-06 09:29:01,721 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 300000
2020-07-06 09:29:10,510 - data_util.log - INFO - iter:207200  mle_loss:2.665  reward:0.0000
2020-07-06 09:29:50,878 - data_util.log - INFO - iter:207250  mle_loss:2.656  reward:0.0000
2020-07-06 09:30:01,782 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 300000
2020-07-06 09:30:29,142 - data_util.log - INFO - iter:207300  mle_loss:2.662  reward:0.0000
```
#### 第一步验证集验证（0200000.tar模型最佳）
```
2020-07-13 14:12:15,025 - data_util.log - INFO - 0005000.tar rouge_1:0.2338 rouge_2:0.0837 rouge_l:0.2338
2020-07-13 14:12:15,060 - data_util.log - INFO - 
2020-07-13 14:12:21,874 - data_util.log - INFO - 0010000.tar rouge_1:0.2782 rouge_2:0.1240 rouge_l:0.2699
2020-07-13 14:12:21,908 - data_util.log - INFO - 
2020-07-13 14:12:28,677 - data_util.log - INFO - 0015000.tar rouge_1:0.2833 rouge_2:0.1211 rouge_l:0.2751
2020-07-13 14:12:28,712 - data_util.log - INFO - 
2020-07-13 14:12:35,460 - data_util.log - INFO - 0020000.tar rouge_1:0.3009 rouge_2:0.1351 rouge_l:0.2898
2020-07-13 14:12:35,496 - data_util.log - INFO - 
2020-07-13 14:12:42,312 - data_util.log - INFO - 0025000.tar rouge_1:0.3121 rouge_2:0.1397 rouge_l:0.3074
2020-07-13 14:12:42,348 - data_util.log - INFO - 
2020-07-13 14:12:49,107 - data_util.log - INFO - 0030000.tar rouge_1:0.2947 rouge_2:0.1264 rouge_l:0.2898
2020-07-13 14:12:49,142 - data_util.log - INFO - 
2020-07-13 14:12:56,030 - data_util.log - INFO - 0035000.tar rouge_1:0.2959 rouge_2:0.1317 rouge_l:0.2870
2020-07-13 14:12:56,067 - data_util.log - INFO - 
2020-07-13 14:13:02,853 - data_util.log - INFO - 0040000.tar rouge_1:0.3200 rouge_2:0.1388 rouge_l:0.3082
2020-07-13 14:13:02,887 - data_util.log - INFO - 
2020-07-13 14:13:09,660 - data_util.log - INFO - 0045000.tar rouge_1:0.2928 rouge_2:0.1212 rouge_l:0.2851
2020-07-13 14:13:09,695 - data_util.log - INFO - 
2020-07-13 14:13:16,485 - data_util.log - INFO - 0050000.tar rouge_1:0.2910 rouge_2:0.1332 rouge_l:0.2883
2020-07-13 14:13:16,519 - data_util.log - INFO - 
2020-07-13 14:13:23,372 - data_util.log - INFO - 0055000.tar rouge_1:0.3003 rouge_2:0.1341 rouge_l:0.2917
2020-07-13 14:13:23,406 - data_util.log - INFO - 
2020-07-13 14:13:30,218 - data_util.log - INFO - 0060000.tar rouge_1:0.3154 rouge_2:0.1498 rouge_l:0.3087
2020-07-13 14:13:30,253 - data_util.log - INFO - 
2020-07-13 14:13:37,074 - data_util.log - INFO - 0065000.tar rouge_1:0.3137 rouge_2:0.1324 rouge_l:0.3060
2020-07-13 14:13:37,108 - data_util.log - INFO - 
2020-07-13 14:13:44,033 - data_util.log - INFO - 0070000.tar rouge_1:0.3128 rouge_2:0.1631 rouge_l:0.3055
2020-07-13 14:13:44,067 - data_util.log - INFO - 
2020-07-13 14:13:50,852 - data_util.log - INFO - 0075000.tar rouge_1:0.3072 rouge_2:0.1439 rouge_l:0.3047
2020-07-13 14:13:50,886 - data_util.log - INFO - 
2020-07-13 14:13:57,661 - data_util.log - INFO - 0080000.tar rouge_1:0.3141 rouge_2:0.1339 rouge_l:0.3038
2020-07-13 14:13:57,696 - data_util.log - INFO - 
2020-07-13 14:14:04,535 - data_util.log - INFO - 0085000.tar rouge_1:0.3093 rouge_2:0.1337 rouge_l:0.3049
2020-07-13 14:14:04,570 - data_util.log - INFO - 
2020-07-13 14:14:11,412 - data_util.log - INFO - 0090000.tar rouge_1:0.3119 rouge_2:0.1434 rouge_l:0.3058
2020-07-13 14:14:11,447 - data_util.log - INFO - 
2020-07-13 14:14:18,227 - data_util.log - INFO - 0095000.tar rouge_1:0.3109 rouge_2:0.1490 rouge_l:0.3067
2020-07-13 14:14:18,262 - data_util.log - INFO - 
2020-07-13 14:14:25,106 - data_util.log - INFO - 0100000.tar rouge_1:0.3217 rouge_2:0.1566 rouge_l:0.3159
2020-07-13 14:14:25,140 - data_util.log - INFO - 
2020-07-13 14:14:32,051 - data_util.log - INFO - 0105000.tar rouge_1:0.3349 rouge_2:0.1560 rouge_l:0.3277
2020-07-13 14:14:32,086 - data_util.log - INFO - 
2020-07-13 14:14:38,902 - data_util.log - INFO - 0110000.tar rouge_1:0.3251 rouge_2:0.1640 rouge_l:0.3179
2020-07-13 14:14:38,936 - data_util.log - INFO - 
2020-07-13 14:14:45,719 - data_util.log - INFO - 0115000.tar rouge_1:0.2923 rouge_2:0.1306 rouge_l:0.2940
2020-07-13 14:14:45,753 - data_util.log - INFO - 
2020-07-13 14:14:52,565 - data_util.log - INFO - 0120000.tar rouge_1:0.3208 rouge_2:0.1511 rouge_l:0.3173
2020-07-13 14:14:52,600 - data_util.log - INFO - 
2020-07-13 14:14:59,384 - data_util.log - INFO - 0125000.tar rouge_1:0.3210 rouge_2:0.1564 rouge_l:0.3176
2020-07-13 14:14:59,418 - data_util.log - INFO - 
2020-07-13 14:15:06,218 - data_util.log - INFO - 0130000.tar rouge_1:0.3092 rouge_2:0.1534 rouge_l:0.3034
2020-07-13 14:15:06,252 - data_util.log - INFO - 
2020-07-13 14:15:13,101 - data_util.log - INFO - 0135000.tar rouge_1:0.3069 rouge_2:0.1536 rouge_l:0.3060
2020-07-13 14:15:13,135 - data_util.log - INFO - 
2020-07-13 14:15:19,985 - data_util.log - INFO - 0140000.tar rouge_1:0.3436 rouge_2:0.1842 rouge_l:0.3392
2020-07-13 14:15:20,019 - data_util.log - INFO - 
2020-07-13 14:15:26,786 - data_util.log - INFO - 0145000.tar rouge_1:0.3205 rouge_2:0.1748 rouge_l:0.3191
2020-07-13 14:15:26,821 - data_util.log - INFO - 
2020-07-13 14:15:33,613 - data_util.log - INFO - 0150000.tar rouge_1:0.3253 rouge_2:0.1786 rouge_l:0.3198
2020-07-13 14:15:33,648 - data_util.log - INFO - 
2020-07-13 14:15:40,436 - data_util.log - INFO - 0155000.tar rouge_1:0.3282 rouge_2:0.1720 rouge_l:0.3218
2020-07-13 14:15:40,469 - data_util.log - INFO - 
2020-07-13 14:15:47,228 - data_util.log - INFO - 0160000.tar rouge_1:0.3243 rouge_2:0.1690 rouge_l:0.3204
2020-07-13 14:15:47,262 - data_util.log - INFO - 
2020-07-13 14:15:54,078 - data_util.log - INFO - 0165000.tar rouge_1:0.3223 rouge_2:0.1592 rouge_l:0.3160
2020-07-13 14:15:54,114 - data_util.log - INFO - 
2020-07-13 14:16:00,915 - data_util.log - INFO - 0170000.tar rouge_1:0.3393 rouge_2:0.1788 rouge_l:0.3329
2020-07-13 14:16:00,950 - data_util.log - INFO - 
2020-07-13 14:16:07,790 - data_util.log - INFO - 0175000.tar rouge_1:0.3156 rouge_2:0.1729 rouge_l:0.3139
2020-07-13 14:16:07,825 - data_util.log - INFO - 
2020-07-13 14:16:14,601 - data_util.log - INFO - 0180000.tar rouge_1:0.3321 rouge_2:0.1717 rouge_l:0.3285
2020-07-13 14:16:14,635 - data_util.log - INFO - 
2020-07-13 14:16:21,411 - data_util.log - INFO - 0185000.tar rouge_1:0.3328 rouge_2:0.1812 rouge_l:0.3338
2020-07-13 14:16:21,444 - data_util.log - INFO - 
2020-07-13 14:16:28,230 - data_util.log - INFO - 0190000.tar rouge_1:0.3313 rouge_2:0.1610 rouge_l:0.3288
2020-07-13 14:16:28,264 - data_util.log - INFO - 
2020-07-13 14:16:35,013 - data_util.log - INFO - 0195000.tar rouge_1:0.3418 rouge_2:0.1818 rouge_l:0.3397
2020-07-13 14:16:35,047 - data_util.log - INFO - 
2020-07-13 14:16:41,848 - data_util.log - INFO - 0200000.tar rouge_1:0.3553 rouge_2:0.1843 rouge_l:0.3481
2020-07-13 14:16:41,883 - data_util.log - INFO - 
2020-07-13 14:16:48,698 - data_util.log - INFO - 0205000.tar rouge_1:0.3442 rouge_2:0.1815 rouge_l:0.3393
```
#### 选择0200000.tar再次训练
```
2020-07-13 14:31:09,581 - data_util.log - INFO - iter:200050  mle_loss:2.367  reward:0.3033
2020-07-13 14:31:36,368 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 14:31:41,245 - data_util.log - INFO - iter:200100  mle_loss:2.393  reward:0.3069
2020-07-13 14:32:12,697 - data_util.log - INFO - iter:200150  mle_loss:2.526  reward:0.2882
2020-07-13 14:32:36,427 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 14:32:43,759 - data_util.log - INFO - iter:200200  mle_loss:2.510  reward:0.3058
2020-07-13 14:33:15,042 - data_util.log - INFO - iter:200250  mle_loss:2.330  reward:0.3175
2020-07-13 14:33:36,487 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 14:33:46,385 - data_util.log - INFO - iter:200300  mle_loss:2.339  reward:0.3113
2020-07-13 14:34:17,783 - data_util.log - INFO - iter:200350  mle_loss:2.379  reward:0.3184
2020-07-13 14:34:36,547 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 14:34:49,007 - data_util.log - INFO - iter:200400  mle_loss:2.374  reward:0.3214
2020-07-13 14:35:20,016 - data_util.log - INFO - iter:200450  mle_loss:2.377  reward:0.3169
2020-07-13 14:35:36,607 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 14:35:51,009 - data_util.log - INFO - iter:200500  mle_loss:2.535  reward:0.3105
2020-07-13 14:36:22,214 - data_util.log - INFO - iter:200550  mle_loss:2.513  reward:0.3016
2020-07-13 14:36:36,667 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 14:36:52,763 - data_util.log - INFO - iter:200600  mle_loss:2.346  reward:0.3243
2020-07-13 14:37:23,430 - data_util.log - INFO - iter:200650  mle_loss:2.387  reward:0.3087
2020-07-13 14:37:36,727 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 14:37:54,837 - data_util.log - INFO - iter:200700  mle_loss:2.519  reward:0.3004
2020-07-13 14:38:25,886 - data_util.log - INFO - iter:200750  mle_loss:2.370  reward:0.3152
2020-07-13 14:38:36,787 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 14:38:57,339 - data_util.log - INFO - iter:200800  mle_loss:2.863  reward:0.2670
2020-07-13 14:39:29,184 - data_util.log - INFO - iter:200850  mle_loss:2.580  reward:0.2928
2020-07-13 14:39:36,830 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
...
2020-07-13 17:09:45,033 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 17:10:15,191 - data_util.log - INFO - iter:215250  mle_loss:2.443  reward:0.3088
2020-07-13 17:10:45,091 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 17:10:47,063 - data_util.log - INFO - iter:215300  mle_loss:2.512  reward:0.2928
2020-07-13 17:11:18,852 - data_util.log - INFO - iter:215350  mle_loss:2.186  reward:0.3183
2020-07-13 17:11:45,151 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 17:11:50,751 - data_util.log - INFO - iter:215400  mle_loss:2.303  reward:0.3289
2020-07-13 17:12:22,880 - data_util.log - INFO - iter:215450  mle_loss:2.335  reward:0.3426
2020-07-13 17:12:45,211 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 17:12:55,016 - data_util.log - INFO - iter:215500  mle_loss:2.339  reward:0.3290
2020-07-13 17:13:26,842 - data_util.log - INFO - iter:215550  mle_loss:2.379  reward:0.3230
2020-07-13 17:13:45,271 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 17:13:58,166 - data_util.log - INFO - iter:215600  mle_loss:2.474  reward:0.3069
2020-07-13 17:14:29,588 - data_util.log - INFO - iter:215650  mle_loss:2.357  reward:0.3247
2020-07-13 17:14:45,331 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 17:15:01,584 - data_util.log - INFO - iter:215700  mle_loss:2.505  reward:0.3078
2020-07-13 17:15:33,432 - data_util.log - INFO - iter:215750  mle_loss:2.261  reward:0.3262
2020-07-13 17:15:45,391 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 17:16:05,635 - data_util.log - INFO - iter:215800  mle_loss:2.480  reward:0.3156
2020-07-13 17:16:37,601 - data_util.log - INFO - iter:215850  mle_loss:2.395  reward:0.3326
2020-07-13 17:16:45,452 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
2020-07-13 17:17:09,609 - data_util.log - INFO - iter:215900  mle_loss:2.487  reward:0.3044
2020-07-13 17:17:41,977 - data_util.log - INFO - iter:215950  mle_loss:2.399  reward:0.3388
2020-07-13 17:17:45,511 - data_util.log - INFO - Bucket queue size: 1000, Input queue size: 10000
```

#### 第二次用验证集验证
```
2020-07-13 17:19:01,530 - data_util.log - INFO - 0200000.tar rouge_1:0.3553 rouge_2:0.1843 rouge_l:0.3481
2020-07-13 17:19:01,564 - data_util.log - INFO - 
2020-07-13 17:19:08,650 - data_util.log - INFO - 0205000.tar rouge_1:0.3449 rouge_2:0.1748 rouge_l:0.3376
2020-07-13 17:19:08,684 - data_util.log - INFO - 
2020-07-13 17:19:15,766 - data_util.log - INFO - 0210000.tar rouge_1:0.3335 rouge_2:0.1635 rouge_l:0.3312
2020-07-13 17:19:15,800 - data_util.log - INFO - 
2020-07-13 17:19:22,869 - data_util.log - INFO - 0215000.tar rouge_1:0.3486 rouge_2:0.1842 rouge_l:0.3467
```

#### 测试
```
2020-07-13 17:20:01,738 - data_util.log - INFO - 0200000.tar rouge_1:0.3396 rouge_2:0.1668 rouge_l:0.3320
```

#### 摘要效果示例
article:文本；ref：参考摘要；dec：模型摘要
```
article: 用于 众筹 的 是 一套 100 平方米 市场价 约 为 90 万 的 三室 住房 只要 投资 1000 元 以上 就 可以 成为 投资者 筹到 54 万元 总额 截止 相当于 总价 的 6 折 成交 金额 超出 54 万 的 部分 将 作为 投资收益 分给 未能 拍 得 房屋 的 其他人
ref: 万科 推出 首个 房产 众筹 预期 年化 收益率 不 低于 40%
dec: 万科 推出 首个 房产 众筹 平台

article: 扩大内需 的 难点 和 重点 在 消费 潜力 也 在 消费 扩大 居民消费 要 在 提高 消费 能力 稳定 消费 预期 增强 消费 意愿 改善 消费 环境 上 下功夫 对此 代表 委员 纷纷表示 应该 健全 医疗 养老 等 保障体系 让 消费者 愿意 消费 并且 敢于 消费 人民日报
ref: 代表 委员 热议 扩 内需 让 百姓 有钱 花敢 花钱
dec: 让 消费者 有钱 敢于 消费

article: 百盛 青岛 啤酒 城 项目 金狮 广场 已现 雏形 外墙 醒目 的 大字 预示 着 这里 将来 的 商业 繁荣 百盛 14 亿 收购 青岛 购物中心 相关 消息 称 百盛 未来 新开 门店 将 以店 中店 和 购物中心 为主 不再 开设 单体 百货 外资 第一 百货 品牌 百盛 也 开始 走 自我 转型 的 道路
ref: 青岛 百盛 14 亿 收购 青岛 购物中心
dec: 青岛 啤酒 城 项目 更名 青岛 购物中心

article: 正 快速 老龄化 的 中国 将 拥有 世界 上 最大 的 老龄 产业 市场 到 2050 年 中国 老年人 口 的 消费 潜力 将 增长 到 106 万亿元 左右 GDP 占 比 将 增长 到 33% 老龄 金融业 和 老龄 房地产业 将 是 增长 的 两大 亮点 今天上午 全国 老龄 工作 办 发布 了 最新 的 中国 老龄 产业 发展 报告
ref: 老龄 产业 亟待 深耕 的 市场
dec: 中国 老龄 产业 发展 报告

article: 游戏 数据分析 中 我们 发现 某项 物品 最近 销售 数据 在 下滑 我们 可能 就 会 下结论 这个 物品 受欢迎 程度 在 下降 但 这个 结论 是 不 准确 的 必须 结合 着 其他 的 数据 一块 看 例如 DAU 如果 DAU 在 下降 那么 该 物品 的 销售 随之 下降 是 正常 的
ref: 几个 很 有 启发性 的 关于 数据 会 说谎 的 真实 例子
dec: 游戏 中 的 销售 数据 在 哪里

article: 在 国外 餐饮 O2O 领域 的 投融资 事件 也 很多 一 Google 收购 软件 公司 Appetas 对抗 Yelp 留下 员工 关闭 网站 二 TripAdvisor 掀起 收购 浪潮 最近 一次 收购 餐饮 预订 网站 LaFourchette 三 Concur 投资 Lastminute 餐厅 预订 网站 Table8 @ 李小双 亿欧网
ref: 国外 巨头 在 餐饮 O2O 领域 的 投融资 事件
dec: 国外 餐饮 O2O 领域 的 投融资 事件

article: 国务院 大部 制 机构 改革方案 10 日 公布 铁道部 并入 交通部 政企分开 建 铁路 总公司 被 媒体 称为 末任 铁道部 部长 的 盛 光祖 表示 铁道部 虽然 将 被 撤销 但 铁路职工 不 存在 安置 问题 也 不会 裁员 铁路 票价 一直 偏低 今后 要 按照 市场规律 按照 企业化 经营 的 模式 来定 票价
ref: 别 了 64 岁 的 铁道部
dec: 别 了 糊涂账

article: 新浪 微博 已有 三岁 俗话说 三岁 看 老 但 从 发展 轨迹 来看 其 未来 并 不 乐观 活跃 用户 少 僵尸 粉 泛滥 内容 趋向 同质化 这些 问题 都 是 源于 新浪 微博 的 媒体 本质 不能 有效 地 加强 普通用户 之间 的 互动 不是 成功 就是 死去 未来 到底 会 怎样 呢
ref: 处于 疯狂 边缘 的 野兽 新浪 微博 繁华 下 的 危机
dec: 专栏 新浪 微博 的 未来

article: 继 寒假 兼职 飞机票 购 退款 热门 综艺 类节目 中奖 等 电话 诈骗 后 喂 我 是 你 领导 骗局 最近 疯狂 来袭 骗子 在 温州 龙湾 转悠 害 得 不少 人 被 骗钱 市民 接到 此类 诈骗 电话 时要 谨慎 与 当事人 当面 核实 以免 上当受骗
ref: 我 是 你 领导 骗局 很 疯狂
dec: 我 是 你 领导 极具 诈骗 电话

article: 记者 采访 获悉 阿里 集团 电商 资产 将 于 年内 上市 这 将 是 陆兆禧 出任 阿里 集团 CEO 所 面临 的 的 首要任务 届时 阿里巴巴 将 可能 成为 全球 第三 大 市值 公司 估值 为 腾讯 与 百度 市值 之 和 陆兆禧 2011 年 出任 阿里 B2BCEO 时 曾 成功 完成 香港 退市 的 任务
ref: 陆兆禧 首务 阿里 电商 资产 年内 上市
dec: 阿里 集团 电商 资产 年内 上市

article: 河南 出入境 检验 检疫局 12 日 对外 通报 称 该局 从 来自 日本 的 邮包 中 截获 日本产 注射用 人体 胎盘 提取液 750 支 邮包 上 收件人 为 某 整形医院 人体 胎盘 提取液 属 生物制品 有 传播 艾滋病 乙肝 丙肝 等 传染病 的 可能 我国 明令禁止 邮寄 进境
ref: 河南 截获 日本 违禁 人体 胎盘 提取液 有 染艾滋 风险
dec: 河南 查获 首 起 人体 胎盘 提取液
```