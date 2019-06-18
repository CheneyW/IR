#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/6/9

检索系统 RetrievalSystem
数据检索的选项卡 DataTab ； 文件检索的选项卡 FileTab
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from retriever import Retriever
import sys
import os
from PIL import Image

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ATTACHMENT_PATH = os.path.join(DIR_PATH, 'data', 'attachment')

data_header = ['选择', '主题', 'url', '附件数', '权限要求']
file_header = ['选择', '文件名', '所属主题', '权限要求']


class MyTab(QWidget):
    def __init__(self, table, retriever):
        super().__init__()
        self.setWindowTitle('企业检索系统')
        self.btn_search = QPushButton('查询')
        self.btn_open = QPushButton('打开')
        self.query_text = QLineEdit(self)  # 查询输入框
        self.search_result_label = QLabel(self)
        self.combo = QComboBox()
        self.table = table

        self._setup()

        self.lines = []
        self.check_box = []
        self.retriever = retriever
        self.show()

    def _setup(self):
        # set layout
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.combo)
        hbox1.addWidget(self.query_text)
        hbox1.addWidget(self.btn_search)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.table)
        hbox2.addWidget(self.btn_open)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addWidget(self.search_result_label)
        vbox.addLayout(hbox2)
        self.setLayout(vbox)
        # set listener
        self.btn_search.clicked.connect(self._search)
        self.btn_open.clicked.connect(self._open)
        # set combo
        self.combo.addItem('权限等级 1')
        self.combo.addItem('权限等级 2')
        self.combo.addItem('权限等级 3')
        self.combo.addItem('权限等级 4')

    def _get_level(self):
        level = self.combo.currentText()
        if level == '权限等级 1':
            return 1
        if level == '权限等级 2':
            return 2
        if level == '权限等级 3':
            return 3
        if level == '权限等级 4':
            return 4

    def _search(self):
        pass

    def _open(self):
        pass


class DataTab(MyTab):
    def __init__(self, table, retriever):
        super().__init__(table, retriever)

    def _search(self):
        query = self.query_text.text()
        self.check_box = []
        self.table.clear()
        self.table.setHorizontalHeaderLabels(data_header)
        self.lines = self.retriever.search_data(query, self._get_level())
        self.table.setRowCount(len(self.lines))
        self.search_result_label.setText('找到 %d 条结果' % len(self.lines))

        for row, data in enumerate(self.lines):
            ck = QCheckBox()
            self.check_box.append(ck)
            hbox = QHBoxLayout()
            hbox.setAlignment(Qt.AlignCenter)
            hbox.addWidget(ck)
            w = QWidget()
            w.setLayout(hbox)

            # '选择', '主题', 'url', '附件数', '权限要求'
            self.table.setCellWidget(row, 0, w)
            self.table.setItem(row, 1, QTableWidgetItem(data['title']))
            self.table.setItem(row, 2, QTableWidgetItem(data['url']))
            self.table.setItem(row, 3, QTableWidgetItem(str(len(data['file_name']))))
            self.table.setItem(row, 4, QTableWidgetItem('>= %d' % data['level']))

    def _open(self):
        choosed_data = [self.lines[i] for i, ck in enumerate(self.check_box) if ck.isChecked()]
        for data in choosed_data:
            QMessageBox.about(self, data['title'], 'title : %s\n\n'
                                                   'url : %s\n\n'
                                                   'parapraghs : %s\n\n'
                                                   'file_name : %s\n\n'
                                                   'privilege level : %d\n\n'
                              % (data['title'], data['url'], data['paragraphs'], str(data['file_name']), data['level']))

        for ck in self.check_box:
            ck.setChecked(False)


class FileTab(MyTab):
    def __init__(self, table, retriever):
        super().__init__(table, retriever)

    def _search(self):
        query = self.query_text.text()
        self.check_box = []
        self.table.clear()
        self.table.setHorizontalHeaderLabels(file_header)
        self.lines = self.retriever.search_file(query, self._get_level())
        self.table.setRowCount(len(self.lines))
        self.search_result_label.setText('找到 %d 条结果' % len(self.lines))

        for row, file in enumerate(self.lines):
            ck = QCheckBox()
            self.check_box.append(ck)
            hbox = QHBoxLayout()
            hbox.setAlignment(Qt.AlignCenter)
            hbox.addWidget(ck)
            w = QWidget()
            w.setLayout(hbox)

            # '选择', '文件名', '所属主题', '权限要求'
            self.table.setCellWidget(row, 0, w)
            self.table.setItem(row, 1, QTableWidgetItem(file[0]))
            self.table.setItem(row, 2, QTableWidgetItem(file[1]))
            self.table.setItem(row, 3, QTableWidgetItem('>= %d' % file[2]))

    def _open(self):
        choosed_data = [self.lines[i][:2] for i, ck in enumerate(self.check_box) if ck.isChecked()]
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
        self.resize(805, 600)

        table1 = QTableWidget(self)
        table1.setColumnCount(5)
        table1.setHorizontalHeaderLabels(data_header)

        table2 = QTableWidget(self)
        table2.setColumnCount(4)
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
