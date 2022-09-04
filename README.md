# chinese_information_extraction
中文信息抽取，包含实体抽取、关系抽取、事件抽取。<br>

数据、预训练模型、训练好的模型下载地址：<br>

链接：https://pan.baidu.com/s/1TdJOF7vjLw4caE1SkZEZGA?pwd=gdpq <br>
提取码：gdpq<br>

说明：主要是将苏剑林的关于信息抽取的一些代码进行跑通并做了一些规划，具体实现可以去看一下他的文章，这里贴出地址：

[GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://spaces.ac.cn/archives/8373)

[GPLinker：基于GlobalPointer的实体关系联合抽取](https://spaces.ac.cn/archives/8888)

[GPLinker：基于GlobalPointer的事件联合抽取 ](https://kexue.fm/archives/8926)

每一个代码里面都有训练和验证，并新增了预测功能，只需要修改do_train和do_predict即可。执行代码：

```python
python xxx_ner/re/ee.py
```

# 依赖
```
pip install keras==2.2.4 
pip install bert4keras==0.10.6 
pip install tensorflow-gpu==1.14.0 
pip install h5py==2.10.0 
```

# 实体抽取
主代码在ner下，数据在data/ner/china-people-daily-ner-corpus/下。ner的结果没有保存，自行预测保存即可。

### bert-crf

```
maxlen = 256  # 句子最大长度
epochs = 10  # 训练的epoch
batch_size = 32  # batchsize大小
learning_rate = 2e-5  # 学习率
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率，crf层的学习率要设置比bert层的大一些

test_f1：0.94876
valid_f1：0.95438
```

## globalpointer

```
maxlen = 256
epochs = 10
batch_size = 32
learning_rate = 2e-5

test_f1: 0.95473
valid_f1: 0.96122
```

# 关系抽取
这里需要安装最新版的keras4bert==0.11.3。这里我只训练了一个epoch，因此将优化器改为了Adam。主代码在re下，数据在data/re/duie/下。

## casrel

```
maxlen = 128
batch_size = 64
epochs = 1
learning_rate = 1e-5

f1: 0.77742, precision: 0.79807, recall: 0.75781, best f1: 0.77742
```

```
{
    "text": "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部",
    "spo_list": [
        [
            "查尔斯·阿兰基斯",
            "出生地",
            "圣地亚哥"
        ],
        [
            "查尔斯·阿兰基斯",
            "出生日期",
            "1989年4月17日"
        ]
    ],
    "spo_list_pred": [
        [
            "查尔斯·阿兰基斯",
            "出生地",
            "智利圣地亚哥"
        ],
        [
            "查尔斯·阿兰基斯",
            "出生日期",
            "1989年4月17日"
        ]
    ],
    "new": [
        [
            "查尔斯·阿兰基斯",
            "出生地",
            "智利圣地亚哥"
        ]
    ],
    "lack": [
        [
            "查尔斯·阿兰基斯",
            "出生地",
            "圣地亚哥"
        ]
    ]
}
```

## globalpointer

```
maxlen = 128
batch_size = 32
epochs = 1
learning_rate = 1e-5

f1: 0.77831, precision: 0.82360, recall: 0.73774, best f1: 0.77831
```

```
{
    "text": "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部",
    "spo_list": [
        [
            "查尔斯·阿兰基斯",
            "出生日期",
            "1989年4月17日"
        ],
        [
            "查尔斯·阿兰基斯",
            "出生地",
            "圣地亚哥"
        ]
    ],
    "spo_list_pred": [
        [
            "查尔斯·阿兰基斯",
            "出生日期",
            "1989年4月17日"
        ],
        [
            "查尔斯·阿兰基斯",
            "国籍",
            "智利"
        ],
        [
            "查尔斯·阿兰基斯",
            "出生地",
            "智利圣地亚哥"
        ]
    ],
    "new": [
        [
            "查尔斯·阿兰基斯",
            "国籍",
            "智利"
        ],
        [
            "查尔斯·阿兰基斯",
            "出生地",
            "智利圣地亚哥"
        ]
    ],
    "lack": [
        [
            "查尔斯·阿兰基斯",
            "出生地",
            "圣地亚哥"
        ]
    ]
}
```

# 事件抽取

这里我只运行了13个epoch。主代码在ee下，数据在data/re/duee/下。

```
[event level] f1: 0.36364, precision: 0.34563, recall: 0.38362, best f1: 0.37487
[argument level] f1: 0.69103, precision: 0.71955, recall: 0.66468, best f1: 0.69103
```

由于之前使用的是谷歌的colab，断断续续的，这里只显示epoch=5的结果：

```
{"text": "振华三部曲的《暗恋橘生淮南》终于定档了，洛枳爱盛淮南谁也不知道，洛枳爱盛淮南其实全世界都知道。", "id": "a7c74f75eb8986377096b4dc62db217d", "event_list": [{"event_type": "产品行为-上映", "arguments": [{"role": "上映影视", "argument": "振华三部曲的《暗恋橘生淮南》"}]}]}
{"text": "腾讯收购《全境封锁》瑞典工作室 欲开发另类游戏大IP", "id": "1bf5de39669122e4458ed6db2cddc0c4", "event_list": [{"event_type": "财经/交易-出售/收购", "arguments": []}]}
{"text": "6月22日，山外杯第四届全国体育院校篮球联赛（SCBA）在日照市山东外国语职业技术大学拉开战幕。", "id": "b98df49b32e4e9924c23bb0cd0c1e83d", "event_list": []}
{"text": "e公司讯，工信部装备工业司发布2019年智能网联汽车标准化工作要点。", "id": "b73704c1d86084ef14d942168b310b1c", "event_list": [{"event_type": "产品行为-发布", "arguments": [{"role": "发布产品", "argument": "2019年智能网联汽车标准化工作要点"}, {"role": "发布方", "argument": "工信部装备工业司"}]}]}
{"text": "新京报讯  5月7日，台湾歌手陈绮贞在社交网络上宣布，已于两年前与交往18年的男友、音乐人钟成虎分手。", "id": "9f7f677595a7f19ca16304a3d85ae94f", "event_list": [{"event_type": "人生-分手", "arguments": []}]}
{"text": "国际金价短期回调 后市银价有望出现较大涨幅", "id": "4d1f964593cd077f9171c09512974e8c", "event_list": []}
{"text": "央视名嘴韩乔生在赛前为中国男篮加油，期待球队展现英雄本色，输球后的韩乔生也相当无奈，他用3个“没有”来点评中国男篮，没有投手、没有经验、没有体力，实在太扎心。", "id": "6e62429b5f2e65c9f6a0052d6d1fa20d", "event_list": [{"event_type": "竞赛行为-胜负", "arguments": [{"role": "败者", "argument": "中国男篮"}]}]}
{"text": "8月31日，第四届两岸关系天府论坛在四川眉山市举行。", "id": "af0185e53b0512bcb5909a20bf3ce1c5", "event_list": []}
{"text": "6月10日基金异动：申万菱信中证申万电子行业投资指数分级B较前一交易日上涨6.6837%", "id": "1461cea84fc66aa3b0ac48b2d3675720", "event_list": []}
{"text": "期间，肖某及其父母向被告黄某支付了彩礼14万元。", "id": "e1eb8215bb981028632296a6d95ebf6f", "event_list": []}

```

# 补充

Q：怎么训练自己的数据集呢？<br>

A：只需要将数据的格式转换为data下面相关数据的格式就可以了。<br>

Q：我想了解下算法背后的原理？

A：直接去看苏剑林的博客，然后跟着代码一步步看下去，做注释。
