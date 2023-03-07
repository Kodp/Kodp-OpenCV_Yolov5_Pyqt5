# 该文件下cut函数可以将位于指定路径+cut中的分数栏照片 中的数字框切分出来,保存至指定路径+cut+%d文件夹下
import matplotlib.pyplot as plt
from utilsMy import *

def cut(path='pic/cut',Show=False,Plt=False,Info=False):
    """
    读取path下的所有图片，cut出来存储到_numi下
    :param path: 文件夹路径
    :param Show: 是否imshow中间图片
    :param Plt: 是否绘制投影直方图
    :param Info: 是否打印列起点终点信息
    :return: 切割后保存的文件夹列表
    """
    k = 1
    save_paths = []  # 切割后保存的文件夹列表
    for i in os.listdir(path):
        save_path = path + "_num" + str(k).zfill(4) + '/'
        cut_one(read_path=os.path.join(path, i),
                save_path=save_path, # 进入xxx_cut内部
                Show=Show,
                Plt=Plt,
                Info=Info)
        save_paths.append(save_path)
        k += 1
    return save_paths


def cut_one(read_path, save_path, Show=False, Plt=False, Info=False):
    """
    读取read_path图片，分割存在savepath下,分割thresh用pixel_thresh
    :param Plt: 是否绘制投影直方图
    :param Show: 是否imshow
    :param Info: 是否打印列信息
    :param read_path:读取图片文件路径
    :param save_path:保存文件夹路径
    """
    newPath(save_path)
    if Info:
        print("\n当前分数栏图片:", read_path)
    img = cv.imread(read_path, cv.IMREAD_GRAYSCALE)  # 读取灰度

    h, w = img.shape[:2]

    # _, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_I NV+cv.THRESH_OTSU)
    sobely = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=5)  # 垂直sobel算子
    sobely = cv.convertScaleAbs(sobely)  # 32f 转8位
    _, sobely = cv.threshold(sobely, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # show(sobely)
    # th = cv.Canny(img, 60, 150, L2gradient=True)  # 数字和竖直线,观察阈值在60左右
    # kernel = np.ones((3, 3), np.float32) / 7
    # th = cv.dilate(th, kernel, iterations=1)
    # th = convVH(th, size=3,h_weight=0.1, v_weight=0.6)
    # show(th)
    # show(sobely)

    w_ = getVProjection(sobely, Show=Show)
    cols_st, cols_ed = getColsStEd(w_,Plt=Plt)
    position = []

    k = 0  # 图片编号，01.jpg， 02.jpg ...
    for i in range(len(cols_st)):
        if (cols_ed[i] - cols_st[i]) / w < 0.045: continue  # 长度 < 1 / 22 认为不是
        k += 1
        if Info:
            print("列的起点和终点:", cols_st[i], cols_ed[i])
        cropImg = img[int(h * 0.4):, cols_st[i]:int(cols_ed[i])]  # 先行后列  行选择0.4到下方,列选择[起点i,终点i]
        # print(k)
        cv.imwrite(os.path.join(save_path, str(k).zfill(2) + '.jpg'), cropImg)  # zfill固定位数
        position.append([cols_st[i], int(h * 0.4), cols_ed[i], h])  # 先行后列 从一半取


    # 确定分割位置
    # if img_origin is not None:
    #     for p in position:
    #         cv.rectangle(img_origin, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 10)
    #     # cv.circle(img_origin,(100,100),100,(0,0,255),-1)
    #     cv.imshow('img_origin', img_origin)
    #     cv.waitKey(0)

def getVProjection(image, Show=False):
    """
    传入黑白图像，得到一维数组，值为图片对应索引上白色像素的个数
    :param image: 需要黑白图像
    :param Show: cvshow
    :param Plt: 是否画投影图
    :return:
    """
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像宽度一致的数组
    w_ = np.zeros(w, dtype=np.uint16)
    # 循环统计每一列黑色像素的个数
    for x in range(w):
        w_[x] = h - np.count_nonzero(image[:, x])

        # w_flip = np.bincount(w_)  # 数据和值反过来
    if Show:
        vProjection = np.zeros(image.shape, np.uint8)
        for x in range(w):
            vProjection[h - w_[x]:, x] = 255  # 垂直正向投影
            # vProjection[w_[x]:, x] = 255  # 垂直反向投影
        print(type(vProjection), type(image))
        shows(0, 1, image, vProjection, destory=False)
    return w_


def getColsStEd(w, Plt=False):
    """
    在垂直投影上分割出每一段区间的起点和终点。
    :param w: 垂直投影数组
    :param Plt: 是否绘制图像
    :return: 起点数组和终点数组
    """
    demarcation = [0]  # 分界点， 第一位
    q2 = np.zeros((len(w) * 2), dtype=np.int32)  # 滑动窗口队列,维护最大值
    w = np.array(w)

    maxv = sorted(w)[-int(len(w) * 0.01)]  # 比较均衡的最大值
    k = int(len(w) * 0.05)  # 滑动窗口长度
    h1, t1 = 0, -1
    h2, t2 = 0, -1
    tmp_max = cnt = 0

    # 滑动窗口
    for i in range(len(w)):
        if i - k + 1 > q2[h2]:
            h2 += 1
        while h2 <= t2 and w[q2[t2]] <= w[i]:
            t2 -= 1
        t2 += 1
        q2[t2] = i

        if i >= k - 1:
            if tmp_max != w[q2[h2]]:  # 更新最大值
                tmp_max = w[q2[h2]]
                cnt = 1  # 坚持了1步
            else:
                cnt += 1
            if cnt > k - 10:  # 已经坚持了几乎整个窗口
                if demarcation[-1] != q2[h2] and w[q2[h2]] > maxv * 0.80:
                    demarcation.append(q2[h2])

            # print(w[q2[h2]], end=' ')
            # 能够维持几乎一整个周期最大值且再maxv81%以上就是分界点
            # 步长选取要小于尖峰间距,但也要尽量大些

    demarcation.append(len(w) - 1)  # 最后一位
    st = demarcation[:-1]
    ed = demarcation[1:]
    if Plt:
        plt.figure(figsize=(20, 5))
        plt.plot(w)
        plt.bar(demarcation,height=maxv*0.8,width=0.005*len(w),color='#fedc5e')
        plt.show()

    return st, ed



def threshStEd(w_, pixel_thresh=110):
    cols_st, cols_ed = [], []  # 行列起点
    position = []
    # 根据水平投影获取垂直分割
    # 起点终点不断交换
    width = len(w_)
    for i in range(len(w_)):  # 横向扫描
        if w_[i] > 0 and start == 0:  # 有数值>0且start=0看为一个框子的起点
            cols_st.append(i)
            start = 1
        elif w_[i] > pixel_thresh and start == 1:  # 数值>60且start=0看为一个框子终点
            #  60 差不多是数字和框的区分
            cols_ed.append(i)
            start = 0

    # 如果终点少了就补一个
    lend, lens = len(cols_ed), len(cols_st)
    if (lend < lens):
        cols_ed = np.concatenate((cols_ed, [width]))  # 两个长度可能不一样,对0032图片特殊补一下
    return cols_st, cols_ed


def getHProjection(image, Show=False):
    hProjection = np.zeros(image.shape, np.uint8)
    (h, w) = image.shape
    h_ = [0] * h
    for y in range(h):
        h_[y] = np.count_nonzero(image[y, :])
    for y in range(h):
        hProjection[y, :h_[y]] = 255  # 速度再快一倍!
    if Show:
        shows(0, 0, image, hProjection, destory=False)
    return h_


# cut(r"E:\Grapro\yolo_trepro\pic\cut")
# path = r"E:\Grapro\Tpreco\src\exam_cut\exam0001.jpg"
# img = cv.imread(path)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# h, w= img.shape[:2]
# # _, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
#
# th = cv.Canny(img,40,150,L2gradient = True)  # 数字和竖直线,观察阈值在60左右
# kernel = np.ones((3,3),np.float32)/9
# th = cv.dilate(th,kernel,iterations = 1)
# # e1 = cv.getTickCount()
# hh=getVProjection(th)



# cut(Plt=True)
# getHProjection(th)

# e2 = cv.getTickCount()
# print((e2 - e1)/cv.getTickFrequency())


# th = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 20)
# _, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# show(th)
# histest = cv.calcHist([gray],[0],None,[256],[0,256])
# plt.hist(img.ravel(),256,[0,256]); plt.show()

# histest = histest.squeeze()
# plt.hist(histest)
# plt.show()
