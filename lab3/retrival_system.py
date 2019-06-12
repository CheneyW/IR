#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/6/9
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from retriever import Retriever
import sys
import os
from PIL import Image

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ATTACHMENT_PATH = os.path.join(DIR_PATH, 'data', 'attachment')

data_header = ['选择', '主题', 'url', '附件数']
file_header = ['选择', '文件名', '所属主题']


class MyTab(QWidget):
    def __init__(self, table, retriever):
        super().__init__()
        self.setWindowTitle('企业检索系统')
        self.btn_search = QPushButton('查询')
        self.btn_open = QPushButton('打开')
        self.query_text = QLineEdit(self)
        self.table = table

        self._setup()

        self.lines = []
        self.check_box = []
        self.retriever = retriever
        self.show()

    def _setup(self):
        # set layout
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.query_text)
        hbox1.addWidget(self.btn_search)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.table)
        hbox2.addWidget(self.btn_open)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        self.setLayout(vbox)
        # set listener
        self.btn_search.clicked.connect(self.search)
        self.btn_open.clicked.connect(self.open)

    def search(self):
        pass

    def open(self):
        pass


class DataTab(MyTab):
    def __init__(self, table, retriever):
        super().__init__(table, retriever)

    def search(self):
        query = self.query_text.text()
        self.check_box = []
        self.table.clear()
        self.table.setHorizontalHeaderLabels(data_header)
        self.lines = self.retriever.search_data(query)
        self.table.setRowCount(len(self.lines))

        for row, data in enumerate(self.lines):
            ck = QCheckBox()
            self.check_box.append(ck)
            hbox = QHBoxLayout()
            hbox.setAlignment(Qt.AlignCenter)
            hbox.addWidget(ck)
            w = QWidget()
            w.setLayout(hbox)

            # '选择', '主题', 'url', '附件数'
            self.table.setCellWidget(row, 0, w)
            self.table.setItem(row, 1, QTableWidgetItem(data['title']))
            self.table.setItem(row, 2, QTableWidgetItem(data['url']))
            self.table.setItem(row, 3, QTableWidgetItem(str(len(data['file_name']))))

    def open(self):
        choosed_data = [self.lines[i] for i, ck in enumerate(self.check_box) if ck.isChecked()]
        for data in choosed_data:
            QMessageBox.about(self, data['title'], 'title      : %s\n\n'
                                                   'url        : %s\n\n'
                                                   'parapraghs : %s\n\n'
                                                   'file_name  : %s\n'
                              % (data['title'], data['url'], data['paragraphs'], str(data['file_name'])))

        for ck in self.check_box:
            ck.setChecked(False)


class FileTab(MyTab):
    def __init__(self, table, retriever):
        super().__init__(table, retriever)

    def search(self):
        query = self.query_text.text()
        self.check_box = []
        self.table.clear()
        self.table.setHorizontalHeaderLabels(file_header)
        self.lines = self.retriever.search_file(query)
        self.table.setRowCount(len(self.lines))

        for row, file in enumerate(self.lines):
            ck = QCheckBox()
            self.check_box.append(ck)
            hbox = QHBoxLayout()
            hbox.setAlignment(Qt.AlignCenter)
            hbox.addWidget(ck)
            w = QWidget()
            w.setLayout(hbox)

            # '选择', '文件名', '所属主题'
            self.table.setCellWidget(row, 0, w)
            self.table.setItem(row, 1, QTableWidgetItem(file[0]))
            self.table.setItem(row, 2, QTableWidgetItem(file[1]))

    def open(self):
        choosed_data = [self.lines[i] for i, ck in enumerate(self.check_box) if ck.isChecked()]
        for file, title in choosed_data:
            path = os.path.join(ATTACHMENT_PATH, title, file + '.jpg')
            if not os.path.exists(path):
                QMessageBox.warning(self, '找不到文件', '不存在路径 ' + path, QMessageBox.Ok)
            else:
                Image.open(path).show()

        for ck in self.check_box:
            ck.setChecked(False)


class RetrievalSystem(QTabWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('企业检索系统')
        self.resize(700, 500)

        table1 = QTableWidget(self)
        table1.setColumnCount(4)
        table1.setHorizontalHeaderLabels(data_header)

        table2 = QTableWidget(self)
        table2.setColumnCount(3)
        table2.setHorizontalHeaderLabels(file_header)

        retriever = Retriever()

        self.tab1 = DataTab(table1, retriever)
        self.tab2 = FileTab(table2, retriever)
        self.addTab(self.tab1, "数据检索")
        self.addTab(self.tab2, "文档检索")
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = RetrievalSystem()
    sys.exit(app.exec_())
