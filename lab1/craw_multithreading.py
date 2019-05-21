#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/8
"""

import os
from concurrent.futures import ThreadPoolExecutor
from urllib import request
import re
from urllib import parse
from bs4 import BeautifulSoup
import threading
import json
import requests

DATA_PATH = os.path.join('data', 'data.json')  # 数据路径
FILE_PATH = os.path.join('data', 'attachment')  # 附件路径

text_len_threshold = 100  # 最小文本长度

if not os.path.exists(FILE_PATH):
    os.mkdir(FILE_PATH)
if os.path.exists(DATA_PATH):
    os.remove(DATA_PATH)


class Crawler(object):
    def __init__(self, page_num=1000):
        self.crawl_pool = ThreadPoolExecutor(max_workers=5)  # 最大并发线程数为5
        self.output_pool = ThreadPoolExecutor(max_workers=5)
        self.count = 0
        self.page_num = page_num
        self.lock = threading.Lock()

    # 回调函数
    def _crawl_future_callback(self, crawl_url_future):
        try:
            urls, data, imgs = crawl_url_future.result()
            # 爬取新的网页
            for new_url in urls:
                self.run(new_url)
            # 保存
            # 过滤过短的文本
            if len(data['paragraphs']) < text_len_threshold:
                return
            self.save(data, imgs)
            # 计数爬取的网页数 需要加锁
            self.lock.acquire()
            if self.count > self.page_num:
                self.lock.release()
                return
            self.count += 1
            print(self.count, '\t', data['title'])
            self.lock.release()
        except Exception as e:
            print(e)
            return

    def run(self, url):
        self.crawl(url, self._crawl_future_callback)

    # 爬取
    def crawl(self, url, complete_callback):

        def request_parse_runnable(url):

            try:
                response = request.urlopen(url)
                content = response.read() if response.getcode() == 200 else None
                urls, data, imgs = self.parse(url, content)
            except Exception as e:
                print(e)
                return None, None, None
            return urls, data, imgs

        future = self.crawl_pool.submit(request_parse_runnable, url)
        future.add_done_callback(complete_callback)  # 运行结束后会调用指定的可调用对象

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

    # 记录数据
    def save(self, data, imgs):
        def output_runnable(data, imgs):
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

        self.output_pool.submit(output_runnable, data, imgs)


if __name__ == '__main__':
    root_url1 = "https://baike.baidu.com/item/%E5%93%88%E5%B0%94%E6%BB%A8%E5%B7%A5%E4%B8%9A%E5%A4%A7%E5%AD%A6/281616?fromtitle=%E5%93%88%E5%B7%A5%E5%A4%A7&fromid=989894&fr=aladdin"
    cm = Crawler()
    cm.run(root_url1)
