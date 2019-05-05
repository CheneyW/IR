#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/4/29
"""

from bs4 import BeautifulSoup
from urllib import parse
from urllib import request
import os
import json
import re
import requests

DATA_PATH = os.path.join('data', 'data.json')  # 数据路径
FILE_PATH = os.path.join('data', 'attachment')  # 附件路径

text_len_threshold = 100  # 最小文本长度

if not os.path.exists(FILE_PATH):
    os.mkdir(FILE_PATH)
if os.path.exists(DATA_PATH):
    os.remove(DATA_PATH)


# url管理器
class UrlManager(object):
    def __init__(self):
        self.new_urls = set()  # 待访问的url
        self.urls = set()  # 已经加入的url

    def add_url(self, url):
        # 防止重复爬取
        if url not in self.urls:
            self.new_urls.add(url)
            self.urls.add(url)

    def add_urls(self, urls):
        for url in urls:
            self.add_url(url)

    def not_empty(self):
        return len(self.new_urls) != 0

    def get_url(self):
        new_url = self.new_urls.pop()
        return new_url


class Crawler(object):
    def __init__(self):
        self.url_manager = UrlManager()

    # 爬取
    def crawl(self, root_url, page_num=1000):
        count = 1  # record the current number url
        self.url_manager.add_url(root_url)
        while self.url_manager.not_empty():
            try:
                new_url = self.url_manager.get_url()
                html_cont = self.download(new_url)
                urls, data, imgs = self.parse(new_url, html_cont)
                self.url_manager.add_urls(urls)
                # 过滤过短的文本
                if len(data['paragraphs']) < text_len_threshold:
                    continue
                self.save(data, imgs)
                print('crawl %4d: %s \t%s' % (count, data['title'], new_url))
                if count == page_num:
                    break
                count += 1
            except Exception as e:
                print(e)

    # 请求下载页面
    @staticmethod
    def download(url):
        response = request.urlopen(url)
        if response.getcode() != 200:  # 判断是否请求成功
            return None
        return response.read()

    # 解析网页
    @staticmethod
    def parse(page_url, html_cont):
        def get_urls(page_url, soup):  # 获取url
            urls = set()
            for x in soup.find_all('a', href=re.compile(r"/item/")):
                urls.add(parse.urljoin(page_url, x['href']))  # 将基地址与一个相对地址形成一个绝对地址
            return urls

        def get_data(page_url, soup):  # 获取数据
            data = dict()
            data['url'] = page_url
            data['title'] = soup.find('dd', class_="lemmaWgt-lemmaTitle-title").find("h1").get_text()
            paragraphs = soup.find('div', class_="lemma-summary").get_text()
            data['paragraphs'] = re.sub(r'\[\d+-\d+\]|\[\d+\]', '', paragraphs)  # 删除引用文献的标号
            return data

        def get_img(soup):  # 获取图片
            img_lst = []
            for img in soup.find_all('img', class_='lazy-img'):
                if len(img['alt']) == 0:
                    continue
                dic = dict()
                dic['name'] = img['alt']
                dic['src'] = img['data-src']
                img_lst.append(dic)
            return img_lst

        soup = BeautifulSoup(html_cont, 'html.parser', from_encoding='utf-8')
        urls = get_urls(page_url, soup)
        data = get_data(page_url, soup)
        imgs = get_img(soup)
        data['file_name'] = [img['name'] for img in imgs]
        return urls, data, imgs

    # 记录数据 & 保存附件
    @staticmethod
    def save(data, imgs):
        # 记录数据
        with open(os.path.join(DATA_PATH), 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

        # 存储图片
        if len(imgs) == 0:
            return
        # path = hashlib.md5(data['title'].encode(encoding='UTF-8')).hexdigest()
        path = os.path.join(FILE_PATH, data['title'])
        if not os.path.exists(path):
            os.mkdir(path)
        for img in imgs:
            img_type = re.findall(r'\.[^\.]+\b', img['src'])[-1]  # 获取图片类型
            req = requests.request('get', img['src'])  # 获取图片
            with open(os.path.join(path, img['name'] + img_type), 'wb') as img_file:  # 以图片名+图片类型存储
                img_file.write(req.content)


if __name__ == "__main__":
    hit_url = "https://baike.baidu.com/item/%E5%93%88%E5%B0%94%E6%BB%A8%E5%B7%A5%E4%B8%9A%E5%A4%A7%E5%AD%A6/281616?fromtitle=%E5%93%88%E5%B7%A5%E5%A4%A7&fromid=989894&fr=aladdin"
    obj_spider = Crawler()
    obj_spider.crawl(hit_url, 1000)
