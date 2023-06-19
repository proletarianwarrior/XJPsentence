# 习近平外交语录分析

作者：能动B2104 杨牧天

## 项目架构

```python
XJPsentence # 项目名称
│  main.py # 主程序
│  test.py # 测试文件
│
├─configuration # 配置文件目录
│      config.yml # 配置文件
│
├─crawler # 爬虫目录
│  │  Crawler.py # 爬虫文件
│  │  __init__.py
│  
│
└─NLP # 自然语言处理目录
    │  DataProcessing.py # 数据深化处理文件
    │  NLP.py # 数据初步处理文件
    │  OPENAI_KEY.env # openai环境配置文件
    │  __init__.py
    │  STSONG.TTF # 词云图字体文件
    │
    └─original_data # 数据保存目录
```

## 项目过程

### 爬取数据

[`config.yml`](.\XJPsentence\configuration\config.yml)

```yaml
# Crawler resource configuration

data_url: "http://jhsjk.people.cn/result?"
json_url: "http://jhsjk.people.cn/testnew/result?"
article_url: "http://jhsjk.people.cn/article/{id}"

params:
  keywords: '' # str ''
  isFuzzy: 0 # int 0
  searchArea: 0 # int 0
  year: 0 # int 0
  form: 0 # int 0
  type: 108 # int 108(外交) 102(政治)
  page: 1 # int
  origin: '全部' # str '全部'
  source: 2 # int 2


# Crawler parameter setting
crawl_page: 0
```

> 数据来源网站为http://jhsjk.people.cn/result，该网站为习近平系列重要讲话数据库，网站首页如下：
>
> ![](D:\Users\DanYang\Desktop\习近平外交语录分析\image\website.png)
>
> 设置params参数，通过Ajax数据爬取的方式，得到了json格式的原始数据，并将其保存在MongoDB数据库中
>
> 如图为使用MongoDB数据库可视化软件MongoDBCompass展示的原始数据：
>
> ![](D:\Users\DanYang\Desktop\习近平外交语录分析\image\MongoDB.png)
>
> 共爬取886条数据，其中包含讲话内容的题目，时间，发布内容的媒体机构，讲话的具体内容

### 基于chatGPT与text2vec的自然语言处理

注：仅展示关键代码，完整代码请见[`NLP.py`](.\XJPsentence\NLP\NLP.py)，[`DataProcessing.py`](.\XJPsentence\NLP\DataProcessing.py)

**经过观察可发现，数据具有以下缺陷：**

+ 数据量严重不足。不同年份数据量不同，多则300多条，少则20条，由于我们需要获取不同年份的主题，因此模型训练数据量严重不足。
+ 内容噪声较多。由于是新闻报道，其中内容并不都是习近平的讲话部分。
+ 编码问题。由于网站书写格式的原因，出现了不可识别的编码，如\u3000等

**综合分析考虑如下方案：**

1. 使用自然语言处理主题提取的相关机器学习算法，如LDA模型

2. 使用预训练好的模型进行主题提取，在Huggingface.co上有诸多开源模型可供选择
3. 使用openai提供的接口进行简单的提示词搭建，利用大语言模型进行主题提取

**最终我选择了第三个方案并取得了较好的效果，理由如下：**

1. 大规模预训练：GPT是在大规模文本数据上进行了广泛的预训练，具备了广泛的语言知识和理解能力。这使得它能够更好地处理各种类型的问题，即使在小样本数据的情况下也能够提供有意义的回答。
2. 迁移学习：GPT的预训练模型经过大规模的数据训练，学到了丰富的语言表示和语法知识。这种迁移学习的能力使得模型能够更好地泛化到新的任务和领域，包括小样本数据的情况。
3. 多领域知识：GPT的训练数据来自于各种领域的文本，涵盖了广泛的主题和领域知识。这使得模型具备了跨领域的理解能力，即使在小样本数据的情况下，也能够提供相关和准确的回答。

**实施过程如下：**

#### 环境变量及接口的搭建

[`OPENAI_KEY.env`](.\XJPsentence\NLP\OPENAI_KEY.env)

```python
API_KEY="sk-rQtTfYF6vsPuYJFky5K2T3BlbkFJthX2BOedIsraeX******" #后六位密匙未显示
```

> 将openai的api_key设置为环境变量，使用如下命令读取：
>
> ```python
> load_dotenv('OPENAI_KEY.env')
> api_key = os.getenv('API_KEY')
> ```
>
> 搭建接口：
>
> ```python
> self.llm = ChatOpenAI(
>             temperature=temperature,
>             openai_api_key=api_key,
>             model_name=model
>         )
> ```

#### 数据清洗



#### 提示词的编写

```python
self.prompt = PromptTemplate(
            input_variables=["describe"],
            template="""
            这是一篇有关习近平外交的新闻报道：{describe}。
            请为我总结最多10个习近平讲话中的关键词(具体几个合适你自己判断)并给出关键词在文章中的权重值(范围为0到1)，输出格式为：关键词+英文逗号+权重。中间以英文分号进行分隔，不要用换行符分隔
            注意你只需给出关键词及权重，其余的什么也不要说
            """
        )
```

> 这里需要注意几点：
>
> 1. 需要强调文本内容的性质
> 2. 需要强调输出格式（为了方便后续数据处理）
> 3. 提示词不应太长（会占用token）

最终得到输出样例如下 ：

```python
习近平外交,0.8;中叙关系,0.7;抗击新冠肺炎疫情,0.5;一带一路,0.4;国家主权领土完整,0.4;国际公平正义,0.3;外部势力干涉,0.3;人权,0.2;台湾,0.1;南海,0.1
```

#### 计算token并舍弃部分数据

```python
def _cal_token(message, model, prompt):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt.format_prompt(describe=message).to_string()))
```

> 使用以上函数计算文章内容所消耗的token数，舍弃token数大于4097的文章
>
> **经计算最终数据量为：823条**

#### 数据的进一步处理

**经过观察可以发现，同一年份不同文章的关键词存在语义相近的现象，针对这种现象，我使用`text2vec`库进行语义相似度的比较，把达到一定阙值的关键词合并，对应权值相加**

以2013年数据为例，筛查结果如下：

```python
亚太经合组织, 亚太一体化
合作, 合作共赢
地区和平, 和平
全球经济治理, 全球治理
```

最终结果如下：

```python
defaultdict(<class 'float'>, {'习近平外交': 2.8, '亚太经合组织': 1.5, '经济全球化': 0.6, '宏观经济政策协调': 0.7, '改革创新': 0.5, '多边贸易体制': 0.6, '开放式发展': 0.5, '贸易保护主义': 0.4, '经济金融稳定': 0.6, '上海合作组织': 1.5, '合作': 2.3, '安全稳定': 0.6, '贸易投资': 0.5, '能源': 0.4, '人文交流': 0.5, '叙利亚问题': 0.2, '阿富汗': 0.2, '金融合作': 0.2, '传统医学': 0.1, '司法部长会议': 0.6, '法治建设': 0.5, '成员国合作': 0.4, '地区和平': 0.8, '国际社会': 0.2, '科学立法': 0.1, '社会主义法治': 0.1, '小康社会': 0.1, '世界经济增长': 0.8, '开放型世界经济': 0.7, '经济增长质量': 0.6, '二十国集团': 0.8, '全球经济治理': 1.0, '改革': 0.6, '市场体系建设': 0.5, '贸易争端': 0.4, '中非友谊': 0.3, '发展': 0.8, '非洲': 0.2, '中刚关系': 0.2, '友好合作': 0.2, '国际关系': 0.6, '民间交往': 0.2, '金砖国家': 0.8, '经济': 0.4, '开放': 0.3, '合理关切': 0.3, '民主化': 0.3})
```

**至此，针对每一年的数据都进行上述操作即可根据关键词及对应权值绘制出词云图**

### 词云图的绘制

使用如下代码绘制词云图：

```python
def plot_wordcloud(data):
    data = Counter(data)
    del data['习近平外交']
    del data['外交部发言人']
    del data['国家主席']
    del data['习近平']
    max_num = 50 if len(data) > 50 else len(data)
    data = {list(data.keys())[i]: list(data.values())[i] for i in range(max_num)}
    fig = plt.figure(dpi=5000)
    wordcloud = WordCloud(font_path='STSONG.TTF', background_color='white', max_words=1000, width=1000, height=500)
    wordcloud.generate_from_frequencies(data)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
```

> **其中去除了几项与我们想获得的数据无关的关键词**。

## 项目结果

> 2013

![](D:\Users\DanYang\Desktop\习近平外交语录分析\image\2013.png)

> 2014

![](D:\Users\DanYang\Desktop\习近平外交语录分析\image\2014.png)

> 2015

![](D:\Users\DanYang\Desktop\习近平外交语录分析\image\2015.png)

> 2016

![](D:\Users\DanYang\Desktop\习近平外交语录分析\image\2016.png)

> 2017

![](D:\Users\DanYang\Desktop\习近平外交语录分析\image\2017.png)

> 2018

![](D:\Users\DanYang\Desktop\习近平外交语录分析\image\2018.png)

> 2019

![](D:\Users\DanYang\Desktop\习近平外交语录分析\image\2019.png)

> 2020

![](D:\Users\DanYang\Desktop\习近平外交语录分析\image\2020.png)

> 2021

![](D:\Users\DanYang\Desktop\习近平外交语录分析\image\2021.png)

> 2022

![](D:\Users\DanYang\Desktop\习近平外交语录分析\image\2022.png)

> 2023

![](D:\Users\DanYang\Desktop\习近平外交语录分析\image\2023.png)

