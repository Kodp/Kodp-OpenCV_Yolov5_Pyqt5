from utilsMy import *
from functools import reduce
import operator
import math


def reordered_clockwise(coords):
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    coords = sorted(coords, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=True)
    return np.array(coords)


def linear_regression(x, y):
    """
    拟合直线
    x, y: list()
    return:  b, k  y = a0 + k*x
    """
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)

    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])

    return np.linalg.solve(A, b)


def angle_regression(x, y):
    """
    拟合角.
    x-y计算一次, y-x计算一次, 求哪个误差小,如果x-y误差小,直接返回angle, 如果y-x误差小, 计算沿45°对称的角
    return: 弧度
    """
    # 垂直的情况
    if x.min() == x.max():
        return math.pi / 2.
    # x-y
    b1, k1 = linear_regression(x, y)
    # y-x
    b2, k2 = linear_regression(y, x)
    # 计算误差
    err1 = 0
    err2 = 0
    for i in np.arange(x.shape[0]):
        err1 += abs(k1 * x[i] - y[i] + b1) / math.sqrt(k1 ** 2 + 1)
        err2 += abs(k2 * y[i] - x[i] + b2) / math.sqrt(k2 ** 2 + 1)
    if err1 <= err2:
        return (math.atan(k1) + 2 * math.pi) % (2 * math.pi) * 180. / math.pi
    else:
        # 计算沿45°对称的角
        return (math.pi / 2. - math.atan(k2) + 2 * math.pi) % (2 * math.pi) * 180. / math.pi


def cal_fit_angle(points, type=1):
    """
    计算两条拟合直线的夹角
    :param points: 点向量,形如[[1,2],[2,3],[4,3]], len为奇数
    :return: 返回夹角,小角
    """

    assert len(points) % 2 == 1
    l = len(points) // 2
    points = np.array(points).transpose()  # transpose 按轴转置
    if type == 1:

        x1, y1 = points[0][:l + 1], points[1][:l + 1]  # 前半段点
        x2, y2 = points[0][l:], points[1][l:]

        linear_model1 = np.polyfit(x1, y1, 1)
        linear_model_fn1 = np.poly1d(linear_model1)  # 返回一个线性函数ax + b, .c返回[a,b]
        linear_model2 = np.polyfit(x2, y2, 1)
        linear_model_fn2 = np.poly1d(linear_model2)

        k1, k2 = linear_model_fn1.c[0], linear_model_fn2.c[0]
        # print(linear_model_fn1, linear_model_fn2)
        # print(k1, k2)
        return abs(math.atan((k2 - k1) / (1 + k1 * k2)) * 180. / math.pi)
    else:
        x1, y1 = points[0][:l + 1], points[1][:l + 1]  # 前半段点
        x2, y2 = points[0][l:], points[1][l:]

        t = abs(angle_regression(x1, y1)) - abs(angle_regression(x2, y2))
        print(t)
        return abs(t)


def unevenLightCompensate(img, blockSize=16):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv.GaussianBlur(dst, (3, 3), 0)
    dst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    return dst


def findMaxContourDP(path, img=None, Show=False, conv=False):
    """
    找到试卷分数栏的轮廓，将其切割、矫正后返回
    :param path: 图片路径
    :param img: 图片，如果没有则从path读取
    :param Show: 展示阈值滤过的图片和最终图片
    :param conv: 是否启用卷积（对无法提取正常分数栏的图片可采用）
    :return:
    """
    if conv:
        imc = cv.imread(path)
        print("using conv")
    if img is None:
        img = cv.imread(path)

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w, _ = img.shape
    Area = h * w
    ret, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cnts, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnt = -1
    # f = lambda x: cv.contourArea(cv.convexHull(x))  # 凸包面积
    # cnts = sorted(cnts, key=f, reverse=True)[:10]  # 按照凸包面积排序
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:20]

    for i in range(len(cnts)):
        eps = 0.03 * cv.arcLength(cnts[i], True)  # 近似距离 ,越小越精确
        approx = cv.approxPolyDP(cnts[i], eps, True)
        area = cv.contourArea(approx)  # 轮廓面积

        if (len(approx) == 4 or len(approx) == 8) and area < 0.5 * Area:
            # img1 = cv.polylines(img, [approx], True, (0, 255, 0), 5)  # 调试用
            # show(img1)
            cnt = approx
            # x, y, w, h = cv.boundingRect(cnt)  # 如果矩形是歪，w/h会较小 可尝试用cv.minAreaRect计算外接矩形，但此函数不一定返回四个点
            rect = cv.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            box = cv.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
            box = np.int0(box)
            distt = cal_dists(box)
            wide, hide = distt[3], distt[0]

            # print(cnt)
            # print(x, y, w, h)
            # print(cv.contourArea(cnt))
            if wide / hide < 3 or area < 0.01 * Area:  # 找到错误的轮廓
                return -1
            break

    cnt = np.array([cnt])  # 顺时针,从右上角开始
    if (cnt.ndim == 1):  # 没找到四个点轮廓,cnt是[-1]
        return None
    pts1 = np.copy(cnt)
    pts1 = np.squeeze(pts1)
    dists = cal_dists(pts1)
    length, width = dists[3], dists[1]

    # ptt = np.array(sorted(pts1, key=lambda d: d[0] + d[1]))  # 按x + y排序
    # x1, y1 = ptt[0]
    # x2, y2 = ptt[-1]
    # if (x1 > x2): x1, x2 = x2, x1; print(f"change x,IMG{os.path.basename(path)}")
    # if (y1 > y2): y1, y2 = y2, y1; print(f"change y, IMG {os.path.basename(path)}")

    pts1 = pts1.astype(np.float32)
    pts1 = np.roll(reordered_clockwise(pts1), axis=0, shift=1)
    pts2 = np.float32([[0, 0], [length, 0], [length, width], [0, width]])  # 新四个点

    try:
        M = cv.getPerspectiveTransform(pts1, pts2)  # pts1 和 pts2的顺序要一样,这里都是顺时针从右上角点开始
    except BaseException as er:
        print("Perspective transform failed! img:", os.path.basename(path), "\n", er, "\n")
        return
    if conv:
        warp = cv.warpPerspective(imc, M, (length, width))
    else:
        warp = cv.warpPerspective(img, M, (length, width))
    if Show:
        shows(0, 1, thresh, warp, destory=True)
    return warp


def cal_dists(pts):
    """
    对顺时针/逆时针点集求距离，返回排好序的距离
    :param pts:
    :return:
    """
    dists = []
    for i in range(len(pts)):
        dists.append(cal_dist(pts[i], pts[(i + 1) % len(pts)]))
    dists.sort()
    return dists


def cal_dist(p1, p2):
    """
    返回整数距离
    :param p1,p2: 两个点
    """
    x1, y1 = p1;
    x2, y2 = p2
    return int(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def cut_path(path):
    for i in range(len(path) - 1, 0, -1):
        if path[i] == '/':
            path = path[:i] + '_cut' + path[i:]
            return path


def find_contour_angle(img, cnt):
    # hull = cv.convexHull(cnt)
    hull = cnt
    cv.drawContours(img, hull, -1, (0, 255, 0), 2)
    hull = np.squeeze(hull)
    assert len(hull.shape) == 2

    min_angle = dict()
    fitsize = 3
    points = np.zeros((fitsize, 2))

    for i in range(len(hull)):
        for j in range(fitsize):
            points[j] = hull[(i + j) % len(hull)]
        print(points)
        min_angle[cal_fit_angle(points, type=1)] = hull[(i + len(hull) // 2) % len(hull)]
        # print(cal_fit_angle(points))
    # print(min_angle)

    tes = sorted(min_angle.items(), key=lambda d: d[0], reverse=True)  # items返回kv元组, lambda 按key排序, 最终得到list元组
    print(tes)
    for i in range(4):  # 角点基本落在10个点以内
        cv.circle(img, tuple(tes[i][1]), 3, (0, 0, 255), -1)
    # show(img, sf=1)


def rec_angle(x, y, z):
    a1 = np.array([y[0] - x[0], y[1] - x[1]], dtype=np.float)
    a2 = np.array([y[0] - z[0], y[1] - z[1]], dtype=np.float)

    # print(a1[0] * a2[0] + a1[1] * a2[1], (a1[0]**2 + a1[1] ** 2)*(a2[0] ** 2 + a2[1] ** 2))
    cos = (1.0 * a1[0] * a2[0] + a1[1] * a2[1]) / math.sqrt(
        1.0 * (a1[0] ** 2 + a1[1] ** 2) * (a2[0] ** 2 + a2[1] ** 2))
    return math.acos(cos) * 180. / math.pi


def find_max_contour(path, img=None, Show=False, find_vetexs=0, margin=20):
    # 找到轮廓
    if img is None:
        img = cv.imread(path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w, _ = img.shape
    Area = h * w  # 面积
    ret, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cnts, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv.findContours(gray_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnt = None

    f = lambda x: cv.contourArea(cv.convexHull(x))  # 凸包面积
    cnts = sorted(cnts, key=f, reverse=True)  # 按照凸包面积排序

    # 找到最大轮廓
    for i in range(len(cnts)):
        hull = cv.convexHull(cnts[i])
        area = cv.contourArea(hull)
        if Area * 0.03 < area < Area * 0.5:
            rect = cv.minAreaRect(cnts[i])
            box = np.int0(cv.boxPoints(rect))
            distt = cal_dists(box)
            wide, hide = distt[3], distt[0]
            cnt = box
            break
    # print(cnt)
    # cv.drawContours(img,[cnt],-1,(0,255,0),5)
    # show(img)
    # find_contour_angle(img, cnt)
    # hull = cnt

    if cnt is None:
        return

    # 找到轮廓两极点后扩展为四点,将这个区域截出来
    cnt = sorted(cnt, key=lambda d: d[0] + d[1])  # 压缩一维, 按x + y排序
    pts1 = np.copy(cnt)
    pts1 = np.squeeze(pts1)
    dists = cal_dists(pts1)
    length, width = dists[3], dists[1]

    ptt = np.array(sorted(pts1, key=lambda d: d[0] + d[1]))  # 按x + y排序
    x1, y1 = ptt[0]
    x2, y2 = ptt[-1]
    if (x1 > x2): x1, x2 = x2, x1; print(f"change x,IMG{os.path.basename(path)}")
    if (y1 > y2): y1, y2 = y2, y1; print(f"change y, IMG {os.path.basename(path)}")

    pts1 = pts1.astype(np.float32)
    pts1 = np.roll(reordered_clockwise(pts1), axis=0, shift=1)
    pts2 = np.float32([[0, 0], [length, 0], [length, width], [0, width]])  # 新四个点

    try:
        M = cv.getPerspectiveTransform(pts1, pts2)  # pts1 和 pts2的顺序要一样,这里都是顺时针从右上角点开始
    except BaseException as er:
        print("Perspective transform failed! img:", os.path.basename(path), "\n", er, "\n")
        return

    warp = cv.warpPerspective(img, M, (length, width))
    if Show:
        shows(0, 1, thresh, warp, destory=True)
    return warp

    # lt, rd = cnt[0], cnt[-1]
    # if (lt[1] > rd[1]): lt[1], rd[1] = rd[1], lt[1]   # 特别歪的情况下,右下角端和左上角关系就变了
    # if (lt[0] > rd[0]): lt[0], rd[0] = rd[0], lt[0]
    # roi_s = np.array([lt[1] - margin,rd[1] + margin, lt[0] - margin,rd[0] + margin])  # 确定左上右下后再减margin,否则可能减反
    # # h, w = img.shape[:2]
    # k = 0
    # # 防止margin过大超过边界使得截图区域错误
    # while (roi_s[0] < 0 or roi_s[1] > h or roi_s[2] < 0 or roi_s[3] > w) and margin > 0:
    #     roi_s = np.array([lt[1] - margin, rd[1] + margin, lt[0] - margin, rd[0] + margin])
    #     margin //= 2
    #     k += 1
    # for i in range(len(roi_s)):
    #     if roi_s[i] < 0 : roi_s[i] = 0
    # if (k != 0):
    #     print('\nmargin change to {0}'.format(margin))
    # if Show:
    #     roi = img[roi_s[0]:roi_s[1], roi_s[2]:roi_s[3]]
    #     show(roi,name=os.path.basename(path))
    # return img[roi_s[0]:roi_s[1], roi_s[2]:roi_s[3]]


# find_max_contour(path=r'E:\Grapro\yolo_trepro\src\A_test0\IMG_20220511_123819.jpg')

# for i in os.listdir(path='exam'):
#     print(i, end = ' ')
#     print(find_max_contour(path=os.path.join('exam',i), Show=True))


# path = 'exam/exam_s0026.jpg'
# img = cv.imread(path)
# find_max_contour_DP(img,Show=True,margin=0)
# find_max_contour(path, Show=True, margin=200)

# e1 = cv.getTickCount()
# for i in os.listdir('src/exam/'):
#     # img = cv.imread('src/exam/' + i)
#     find_max_contour_DP(path = 'src/exam/' + i, margin=20)
# e2 = cv.getTickCount()
# print((e2 - e1)/cv.getTickFrequency())
