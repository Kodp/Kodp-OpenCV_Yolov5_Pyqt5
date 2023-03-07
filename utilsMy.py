# 常用函数实现
# 一次导入全局可用，如cv
import cv2 as cv
import numpy as np
import os
import shutil
from loguru import logger
import sys
from time import time
from importlib import import_module
from collections.abc import Iterable

def auto_import(*modules):
    """
    自动导入modules
    :param modules:
    """
    for moduleName in modules:
        if isinstance(moduleName, str):
            if moduleName in sys.modules:
                logger.info("[Note]:{} already imported.".format(moduleName))
            else:
                timeStart = time()
                try:
                    import_module(moduleName)
                except ModuleNotFoundError:
                    logger.warning("import {}  ".format("'" + moduleName + "'") + " ModuleNotFound.")
                except BaseException as er:
                    logger.error(er)
                else:
                    logger.info("import {}  ".format("'" + moduleName + "'") + " successfully in {:.2}s.".format(
                        time()- timeStart))
        elif isinstance(moduleName, Iterable):  # 嵌套import
            auto_import(*moduleName)
        else:
            logger.error("import error, moduleName must be str or Iterable.")



def show(img=None, name="show", sf=1):
    """
    展示img
    :param img:图片
    :param name: 窗口名
    :param sf: 缩放是否开启
    """
    if img is None or not (isinstance(img, np.ndarray)):
        cv.destroyAllWindows()
        print("Not img")
        return
    if sf:
        cv.namedWindow(name, cv.WINDOW_NORMAL)
    try:
        cv.imshow(name, img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except BaseException as er:
        print("show Error!\n", er)
    finally:
        cv.destroyAllWindows()


def shows(name=0, sf=1, *imgs, destory=True):
    """
    展示多张图片. 不要缩放就shows(0, 0, img1,img2, ...)
    :param name:窗口名
    :param sf: 是否开启缩放
    :param imgs: 图片
    """
    try:
        if type(name) == int:
            name = 'shows'
        t = 0

        for i in imgs:
            if sf:
                cv.namedWindow(name + str(t), cv.WINDOW_NORMAL)
            cv.imshow(name + str(t), i)
            t += 1

        cv.waitKey(0)
        if destory:
            cv.destroyAllWindows()
    except BaseException as er:
        print("shows Error!\n", er)
    finally:
        if destory:
            cv.destroyAllWindows()


def get_img_list(dir):
    """
    遍历dir,返回目录一级下所有图片文件的路径list
    :param dir: 图片文件绝对路径的list
    """
    img_list = []
    root_depth = len(dir.split(os.path.sep))  # 根深度
    for root, dirs, files in os.walk(dir):
        for filename in files:
            # 文件名列表，包含完整路径
            td = os.path.join(root, filename)
            depth = len(td.split(os.path.sep))
            # 只遍历一层目录, 通过深度判断是否是一层
            if depth == root_depth + 1 and td.endswith(('jpg', 'png', 'jpeg')):  # 是图片才加
                img_list.append(os.path.join(root, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
    return img_list


def edSlashDel(path):
    """
    path去除末尾slash
    :param path:路径
    :return: 返回去除后的路径
    """
    if path[-1] == '/' or path[-1] == '\\':
        temp = list(path)
        temp[-1] = ''
        path = ''.join(temp)
    return path


def convVH(img, size=5, h_weight=0.5, v_weight=0.5, enhance=1.5):  # 三维
    """
    水平和竖直卷积叠加，size为卷积核大小，默认分配1：1
    注意每个方向需要两个卷积核，否则只会提取一边的特征。
    :param enhance: 边缘增强
    :param img:输入图片，3维
    :param size:卷积核大小，奇数
    :param h_weight:水平特征比重
    :param v_weight:竖直特征比重
    :return:
    """
    kernel_v1 = np.array([[i for i in range(-(size // 2), (size // 2) + 1, 1)] for i in range(size)])
    kernel_v2 = np.array([[i for i in range((size // 2), -(size // 2) - 1, -1)] for i in range(size)])
    kernel_h1 = np.array([[j for i in range(size)] for j in range(-(size // 2), (size // 2) + 1, 1)])
    kernel_h2 = np.array([[j for i in range(size)] for j in range((size // 2), -(size // 2) - 1, -1)])
    # print(kernel_v1, '\n', kernel_v2, '\n', kernel_h1, '\n', kernel_h2)

    dst1 = cv.filter2D(img, -1, kernel_v1)
    dst2 = cv.filter2D(img, -1, kernel_h1)
    dst3 = cv.filter2D(img, -1, kernel_v2)
    dst4 = cv.filter2D(img, -1, kernel_h2)

    imc1 = cv.addWeighted(dst1, v_weight, dst2, h_weight, 0)
    imc2 = cv.addWeighted(dst3, v_weight, dst4, h_weight, 0)
    imc = cv.addWeighted(imc1, enhance, imc2, enhance, 0)
    return imc


def newPath(path):
    """
    不存在path则创建，存在path则删除再创建。
    :param path:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    else:  # 已存在直接删除
        try:
            shutil.rmtree(path)
        except BaseException as er:
            print(f"\033[1;31;43m{os.path.basename(path)}被占用！\n{er}\033[0m")
            return
        else:
            os.mkdir(path)


def moving_average(x, w):
    """
    返回滑动平均后的数组
    :param x: 一维数组
    :param w: 滑动窗口大小
    """
    # convolve计算后长度为len - w + 1,为保持长度一致后面补上w - 1个值为x[-1]的数
    # concatenate接受元组
    return np.concatenate((np.convolve(x, np.ones(w), 'valid') / w, np.full((w - 1), x[-1])))
