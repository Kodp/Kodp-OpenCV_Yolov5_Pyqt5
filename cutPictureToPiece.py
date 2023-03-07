# 该文件下cut函数可以将指定路径中试卷照片中的分数栏切分出来,保存在指定路径+cut下

from findContourAngle import *
from loguru import logger

def cut(path="pic/exam", margin=20):
    """
    读取path路径下的所有图片，在其同级目录下建立cut文件夹，将切割好的分数栏图片放进去，填充margin
    如果同级目录下存在cut文件夹则删除重建
    :param: path是文件夹路径,对path下所有图片做切割后转到path_cut
    :param: method是方法.0为凸包方法,1为approxPolyDP方法
    :return cut图片的保存路径
    """
    # 防止路径以斜杠结尾导致cut文件夹建立在exam内部
    path = edSlashDel(path)

    save_path = os.path.join(os.path.split(path)[0], 'cut')
    print("picture save to ", os.path.join(os.getcwd(), save_path))
    newPath(save_path)

    # 遍历图片 切出分数栏
    # for path, dir_list, i in os.walk(path):
    for i in os.listdir(path):
        cur_path = os.path.join(path, i)
        if not cur_path.endswith(('jpg', 'png', 'jpeg')):
            continue
        print("read", i)

        img_cut = findMaxContourDP(path=cur_path)

        if isinstance(img_cut,int):  # -1
            # print(f"\033[1;33;42m{os.path.basename(cur_path)} try to add conv.\n\033[0m")
            logger.warning("try to add conv.")
            img = convVH(cv.imread(cur_path))
            # show(img) 展示卷积图像
            img_cut = findMaxContourDP(cur_path,img,conv=True)


        if img_cut is None or isinstance(img_cut, int):
            print("img_cut is", type(img_cut))
            # print(f"\033[1;33;42m{os.path.basename(cur_path)} Using DP method failed! try to use convex hull method.\n\033[0m")
            logger.warning("Using DP method failed! try to use convex hull method.")
            img_cut = find_max_contour(cur_path, margin=margin)
            if img_cut is None:
                # print(f"\033[1;31;43m{os.path.basename(cur_path)} The score box is not recognized!\033[0m\n")
                logger.error("The score box is not recognized!")
                continue

        cv.imwrite(os.path.join(save_path, i), img_cut)
    # print("\n\033[0;32mcut finished.\033[0m")
    logger.info("cut finished.")

    return save_path

