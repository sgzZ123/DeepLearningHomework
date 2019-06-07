#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from net1 import *
from Data1 import *
import torch
import matplotlib.pyplot as plt
from PIL import Image

# window类继承于QMainWindow
class window(QMainWindow):

    def __init__(self):
        # 初始化一个img的ndarray, 用于存储图像
        self.img = np.ndarray(())
        super().__init__()
        self.initUI()

    # 初始化窗口函数
    def initUI(self):
        self.comnolist = 0
        # 窗口大小，位置，名称，图标
        self.resize(1024, 768)
        # self.setMaximumSize(1600, 1200)
        self.setMinimumSize(1024, 768)
        self.setWindowTitle('PHOTO COLORIZATION')
        self.setWindowIcon(QIcon('pic.png'))

        self.setWindowOpacity(0.95)  # 设置窗口透明度
        # self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明

        self.main_widget = QWidget()  # 创建窗口主部件
        self.main_layout = QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局

        self.left_widget = QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QGridLayout()  # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout)  # 设置左侧部件布局为网格

        self.right_widget = QWidget()  # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        # self.right_layout = QGridLayout()
        self.right_layout = QVBoxLayout()
        self.right_widget.setLayout(self.right_layout)  # 设置右侧部件布局为网格

        self.main_layout.addWidget(self.left_widget, 0, 0, 12, 6)  # 左侧部件在第0行第0列，占16行10列
        self.main_layout.addWidget(self.right_widget, 0, 6, 12, 1)  # 右侧部件在第0行第10列，占16行2列
        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

        self.main_label = QLabel(self)
        # 创建图像显示label
        self.grayphoto = QLabel(self)
        self.grayscrollarea = QScrollArea(self)
        self.photoincolor = QLabel(self)
        self.colorscrollarea = QScrollArea(self)


        # 设置按钮
        self.addbtn = QPushButton('添加', self)
        self.colorizebtn = QPushButton('上色', self)
        self.savebtn = QPushButton('保存', self)
        self.quitbtn = QPushButton('退出', self)

        self.grayscrollarea.setWidget(self.grayphoto)
        self.grayscrollarea.setWidgetResizable(True)
        # self.grayscrollarea.setMinimumSize(800, 600)
        self.grayscrollarea.setAlignment(Qt.AlignCenter)

        self.colorscrollarea.setWidget(self.photoincolor)
        self.colorscrollarea.setWidgetResizable(True)
        # self.colorscrollarea.setMinimumSize(800, 600)
        self.colorscrollarea.setAlignment(Qt.AlignCenter)

        self.text = QLabel(self)
        self.text.setText("选择图像分类")
        # 创建一个下拉列表框并填充了3个列表项
        self.combo = QComboBox(self)
        self.combo.addItem("请选择...")
        self.combo.addItem("山林")
        self.combo.addItem("家具")
        self.combo.addItem("海滩")

        # 按钮列表框字体设置
        font = QFont()
        font2 = QFont()
        font.setFamily('黑体')
        font.setPointSize(10)
        font.setWeight(55)
        # font.setLetterSpacing(QFont.PercentageSpacing, 130)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 5)
        self.addbtn.setFont(font)
        self.colorizebtn.setFont(font)
        self.quitbtn.setFont(font)
        self.savebtn.setFont(font)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 3)
        self.combo.setFont(font)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 1)
        self.text.setFont(font)

        # 设置网格坐标
        self.left_layout.addWidget(self.grayscrollarea, 0, 0, 5, 5)
        self.left_layout.addWidget(self.colorscrollarea, 6, 0, 5, 5)
        # 设置按钮布局
        self.right_layout.addSpacing(80)
        self.right_layout.addWidget(self.addbtn)
        self.right_layout.addSpacing(60)
        self.right_layout.addWidget(self.text)
        self.right_layout.addSpacing(0)
        self.right_layout.addWidget(self.combo)
        self.right_layout.addSpacing(80)
        # self.right_layout.addStretch()
        self.right_layout.addWidget(self.colorizebtn)
        self.right_layout.addSpacing(80)
        self.right_layout.addWidget(self.savebtn)
        self.right_layout.addSpacing(80)
        self.right_layout.addWidget(self.quitbtn)
        self.right_layout.addSpacing(100)

        # 背景调色板设置
        palette = QPalette()
        palette1 = QPalette()
        palette1.setColor(self.backgroundRole(), QColor(255, 255, 255))  # 设置背景颜色
        palette.setBrush(self.backgroundRole(), QBrush(QPixmap('D:/USTC learning/python/colorization gui/th.jpg')))
        self.setPalette(palette)
        self.grayphoto.setPalette(palette1)
        self.photoincolor.setPalette(palette1)
        self.setAutoFillBackground(True)

        # self.setWindowFlags(Qt.FramelessWindowHint)

        # 按钮的信号与槽设置
        self.addbtn.clicked.connect(self.openImg)
        self.combo.activated[str].connect(self.combo_onActivated)
        self.colorizebtn.clicked.connect(self.colorization)
        self.savebtn.clicked.connect(self.saveImg)
        self.quitbtn.clicked.connect(QCoreApplication.instance().quit)

        # 显示窗口
        self.show()

    # 打开本地图像方法
    def openImg(self):
        # 调用打开文件diglog
        fileName, tmp = QFileDialog.getOpenFileName(
            self, 'Open Image', './__data', '*.png *.jpg *.bmp')

        if fileName is '':
            return
        # 采用opencv函数读取数据
        self.filename = fileName
        self.img = cv.imread(fileName, 0)
        # cv.imshow(self.img)

        if self.img.size == 1:
            return
        self.refreshShowgray()

    # 图像显示方法
    def refreshShowgray(self):
        # 提取图像的尺寸和通道, 用于将opencv下的image转换成Qimage
        # height, width, channel = self.img.shape
        if np.array(self.img.shape).shape == (2,):
            height, width = self.img.shape
            bytesPerLine = 1 * width
            # self.qImg = QImage(self.img.data, width, height, bytesPerLine, QImage.Format_Grayscale8).rgbSwapped()
            self.grayImg = QImage(self.img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

            # 将Qimage显示出来
            self.grayphoto.setPixmap(QPixmap.fromImage(self.grayImg))
            self.grayphoto.setAlignment(Qt.AlignCenter)

    def refreshShowrgb(self):
        # 提取图像的尺寸和通道, 用于将opencv下的image转换成Qimage
        # height, width, channel = self.img.shape
        if np.array(self.outputimg.shape).shape == (3,):
            height, width = self.outputimg.shape[1], self.outputimg.shape[0]
            bytesPerLine = 3 * width
            self.rgbImg = QImage(self.outputimg.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

            # 将Qimage显示出来
            self.photoincolor.setPixmap(QPixmap.fromImage(self.rgbImg))
            self.photoincolor.setAlignment(Qt.AlignCenter)

    # 列表项选中连接方法
    def combo_onActivated(self, text):
        if text == "山林":
            self.comnolist = 1
        elif text == "家具":
            self.comnolist = 2
        elif text == "海滩":
            self.comnolist = 3

    def graytorgb(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ColorNet()
        model = model.to(device)
        # 调用训练完的模型
        #model.load_model('D:/git/dpgui/furniture-4.pth')
        if self.comnolist == 1:
            pretrained = torch.load('D:/USTC learning/python/colorization gui/mountain.pth',
                                    map_location=lambda storage, loc: storage)
        elif self.comnolist == 2:
            pretrained = torch.load('D:/USTC learning/python/colorization gui/furniture-4.pth',
                                    map_location=lambda storage, loc: storage)
        elif self.comnolist == 3:
            pretrained = torch.load('D:/USTC learning/python/colorization gui/sea-1.pth',
                                    map_location=lambda storage, loc: storage)
        model.load_state_dict(pretrained)
        img_l = Image.open(self.filename)
        img_l = np.asarray(img_l)
        img_l = img_as_float(img_l)
        img_l = torch.from_numpy(img_l).unsqueeze(0).unsqueeze(0).float()

        output = model(img_l)
        self.outputimg = to_rgb(img_l.cpu(), output.detach().cpu())
        self.outputimg = img_as_ubyte(self.outputimg)
        r = self.outputimg[:,:,0]
        g = self.outputimg[:,:,1]
        b = self.outputimg[:,:,2]
        self.outputimg = cv.merge([b,g,r])

    # 上色方法
    def colorization(self):
        self.graytorgb()
        self.refreshShowrgb()


    # 将图像保存到本地方法
    def saveImg(self):
        # 调用存储文件dialog
        fileName, tmp = QFileDialog.getSaveFileName(
            self, 'Save Image', './__data', '*.png *.jpg *.bmp', '*.png')
        if fileName is '':
            return
        if self.outputimg.size == 1:
            return
        # 调用opencv写入图像
        cv.imwrite(fileName, self.outputimg)
