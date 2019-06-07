#!/usr/bin/python3
# -*- coding: utf-8 -*-

from guipackage import window
import sys
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = window()

    sys.exit(app.exec_())