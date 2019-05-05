# 实验一： 网页文本的预处理

实验分为两个部分

## 任务一：网页的抓取和正文提取

爬取百度百科站点的1000个网页，提取网页标题和网页正文（这里选取每个词条的简介作为正文），以及网页中的照片并保存附件到本地。

爬取下来的数据保存为 json 格式，如下：

```json
{
    “url”: “http://today.hit.edu.cn/article/2019/03/25/65084”,
    “title”: “计算机学院召开第 3 次科创俱乐部主席联席会”,
    “paragraph”: "text paragraph"
    “file_name”: [file_1, file_2, … file_n]
}
```

程序文件`craw.py`

数据路径`data/data.json`	

附件路径`data/attachment`

## 任务二：分词处理、去停用词处理

提取的网页文本进行分词和去停用词处理，并将结果保存

分词工具使用pyltp，对任务一得到的标题和正文进行分词，并保存为json格式，如下：

```json
{
    “url”: “http://today.hit.edu.cn/article/2019/03/25/65084”,
    “segmented_title”: [“计算机学院”, “召开”, “第 3 次”, “科创俱乐部”, “主席”, “联席会”],
    “segmented_paragraph”:[segmented text paragraph 1],
    “file_name”: [file_1, file_2, … file_n]
}
```

程序文件`segment.py`

数据路径`data/preprocessed.json`	

停用词表`data/stop_words.txt`