# -*- coding: utf-8 -*-
class Item:
    def __init__(self, img, class_num, file_name, file_path):
        self.__img = img
        self.__class_Num = class_num
        self.__file_name = file_name
        self.__file_path = file_path
        self.__normalized_img = None
        self.__lbp_histogram = None

    @property
    def img(self):
        return self.__img

    @img.setter
    def img(self, img):
        self.__img = img

    @property
    def class_num(self):
        return self.__class_Num

    @class_num.setter
    def class_num(self, class_num):
        self.__class_Num = class_num

    @property
    def normalized_img(self):
        return self.__normalized_img

    @normalized_img.setter
    def normalized_img(self, normalized_img):
        self.__normalized_img = normalized_img

    @property
    def lbp_histogram(self):
        return self.__lbp_histogram

    @lbp_histogram.setter
    def lbp_histogram(self, lbp_histogram):
        self.__lbp_histogram = lbp_histogram

    def __str__(self):
        return 'file_name = ' + self.__file_name + ', class_num = ' + str(self.__class_Num) + ', file_path = ' \
               + self.__file_path
