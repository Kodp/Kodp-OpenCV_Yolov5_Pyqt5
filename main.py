"""
本项目命名规范：
    类名CamelCase
    包名、控件名camelCase
    一般变量、函数小写+下划线
    常量大写
"""
from utilsMy import *
import os.path as osp
import threading
from PyQt5.QtCore import *  # pyqtSignal
import cutPictureToPiece
import cutPieceToSingle
import torch
import sys
from PyQt5.QtWidgets import *
from src.qtWin.myWin import *
from PyQt5.Qt import QThread


class Message(QThread):  # myWindow中生成这样一个线程（对象）留驻，准备接受信息。
    signal = pyqtSignal(str,str)  # 定义信号，用它发送信号到主线程（gui），让主线程弹窗，避免子线程弹窗
    def __init__(self):
        super(Message, self).__init__()

    def run(self):
        # self.signal.emit()
        pass

class DetectEnded(QThread):
    signal = pyqtSignal()  # 监听detect执行是否结束
    def run(self):
        pass

class CutEnded(QThread):
    signal = pyqtSignal()
    def run(self):
        pass

class MyWindow(QMainWindow, Ui_mainWindow):

    def __init__(self,model):

        super(MyWindow, self).__init__()
        self.setupUi(self)  # 初始化界面
        self.setFixedSize(1259, 804)  # 固定大小
        # 绑定按钮
        self.openDir.triggered.connect(self.open_dir)  # QAction的绑定
        self.openImg.triggered.connect(self.open_img)
        self.nextImg.clicked.connect(self.next_img)
        self.lastImg.clicked.connect(self.last_img)
        self.recoCutButton.clicked.connect(self.detect_button_st)
        self.cutButton.clicked.connect(self.cut_button_st)
        self.saveResult.clicked.connect(self.save_result)

        # 初始化线程
        self.msg = Message()
        self.msg.signal.connect(self.box)  # 线程发送的信号绑定box，box是主线程中的。
        self.detect_ended = DetectEnded()
        self.detect_ended.signal.connect(self.show_score)  # detect_ended绑定show_score
        self.cut_ended = CutEnded()
        self.cut_ended.signal.connect(self.show_single)

        self.model = model
        self.dir_path = None  # 由系统的打开的文件夹界面打开的路径，为绝对格式
        self.save_path = None  # 分数栏的保存路径
        self.single_paths = None # ['E:/Grapro/yolo_trepro/src/test\\cut_num0001/', 'E:/Grapro/yolo_trepro/src/test\\cut_num0002/', ]
        self.file_pointer = 0  # 文件遍历顺序（指针）,从0开始
        self.file_len = 0  # 图片文件夹下图片数, 也是切割出文件夹的数量
        self.file_list = None  # dir_path下图片的绝对路径组成的list

        self.file_path = None  # “打开图片”对应的文件夹位置
        self.label_path = None  # 弃用
        self.label_list = None  # 弃用
        self.detected = False  # 检测是否完成标志

        self.file_cut_list = None  # 存切分分数框的图片路径,list
        self.file_cut_len = 0  # 切分分数框的数量,可能和图片数不一样(没找到凸包/polydp)

        self.score_list = None  # 第一维试卷编号file_pointer,第二维分数框
        self.res_path = None  # txt结果保存位置 // 并没有用

        self.cutting = False  # cut按钮锁
        self.detecting = False  # 检测按钮锁
        self.qmbox = QMessageBox()  # 用于发送信息，单独线程
        # 可编辑分数框绑定onTextEdited
        for i in range(15):
            sc_i = getattr(self, "sc" + str(i))
            sc_i.textEdited.connect(self.on_text_edited)

        self.barlabel = QLabel()  # 用于显示图片名和序号
        self.placeholder = QLabel()  # 占位符，作用为把barlabel挤到右边去
        # 往状态栏中添加组件（stretch应该是拉伸组件宽度）
        self.statusbar.addPermanentWidget(self.placeholder, stretch=10)
        self.statusbar.addPermanentWidget(self.barlabel, stretch=1)
        self.placeholder.setText('by:爱学习的图灵机')  # 添加名字
    def refresh_var(self):
        """
        打开新文件时使用,更新变量
        """
        self.dir_path = None
        self.save_path = None
        self.single_paths = None
        self.file_pointer = 0
        self.file_len = 0
        self.file_list = None
        self.file_path = None
        self.label_path = None
        self.label_list = None
        self.file_cut_list = None
        self.file_cut_len = 0
        self.score_list = None
        self.detected = False
        self.res_path = None
        self.detecting = False
        self.cutting = False


    def box(self,string='文本',Btype="窗口提示",buttons=QMessageBox.Ok):
        self.qmbox.information(self, Btype, string, buttons)

    @property
    def img_info(self,pos=None):
        """
        返回file_pointer位置对应的图片路径（只有文件名）+ 序号
        open_img 不适用
        :param pos: 位置
        :return:
        """
        try:
            if pos is None:
                return '{0:<20}'.format(osp.basename(self.file_list[self.file_pointer])[:20])+'{0:>6}'.format(str(self.file_pointer+1)[:6])  # 左对齐，20位，右对齐， 6位
            else:
                return '{0:<20}'.format(osp.basename(self.file_list[pos]))+'{0:>6}'.format(str(pos+1))
        except BaseException as er:
            print("print info error:", er)


    def next_img(self):
        if (self.file_pointer_plus()):
            self.examPic.setPixmap(QtGui.QPixmap(self.file_list[self.file_pointer]))
            # self.statusbar.showMessage(self.img_info)  # 状态栏展示文件名
            self.barlabel.setText(self.img_info)
            # print(type(osp.basename(self.file_list[self.file_pointer])))
            if self.file_cut_list:
                self.examCutPic.setPixmap(QtGui.QPixmap(self.file_cut_list[self.file_pointer]))
                self.show_single(self.file_pointer)

            if self.score_list is not None:
                self.show_score(self.file_pointer)

    def last_img(self):
        if (self.file_pointer_minus()):
            self.examPic.setPixmap(QtGui.QPixmap(self.file_list[self.file_pointer]))
            # self.statusbar.showMessage(self.img_info)
            self.barlabel.setText(self.img_info)
            if self.file_cut_list:
                self.examCutPic.setPixmap(QtGui.QPixmap(self.file_cut_list[self.file_pointer]))
                self.show_single(self.file_pointer)
            if self.score_list is not None:
                self.show_score(self.file_pointer)

    def open_dir(self):
        if self.detecting or self.cutting:
            self.msg.signal.emit("正在处理，请稍后", "提示")
            return
        # 文件下的打开文件夹,给file_list赋值,显示图片
        self.refresh_var()  # 刷新变量
        self.dir_path = QFileDialog.getExistingDirectory(self, '选择图片文件夹')  # 绝对路径
        self.file_list = get_img_list(self.dir_path)
        print(self.file_list)
        self.file_len = len(self.file_list)  # 文件个数
        if self.file_len > 0:
            self.examPic.setPixmap(QtGui.QPixmap(self.file_list[0]))
            # self.statusbar.showMessage(osp.basename(self.file_list[self.file_pointer]))
            self.barlabel.setText(self.img_info)
        else:
            QMessageBox.information(self, '提示', '选择的目录下没有图片文件')

    def open_img(self):
        if self.detecting or self.cutting:
            self.msg.signal.emit("正在处理，请稍后", "提示")
            return
        if self.detecting or self.cutting:
            return

        self.refresh_var()  # 刷新变量
        self.file_path = QFileDialog.getOpenFileName(self, '选择图片文件夹',
                                                     filter="*.png;*.bmp;*.jpg;*.jpeg")
        print(self.file_path)
        if not self.file_path or self.file_path == ('', ''):
            QMessageBox.information(self, '提示', '未选择图片')
        else:
            self.examPic.setPixmap(QtGui.QPixmap(self.file_path[0]))
            # self.statusbar.showMessage(osp.basename(self.file_path[0]))
            self.barlabel.setText(osp.basename(self.file_path[0]))

    def cut_button_st(self):
        if self.cutting:
            self.msg.signal.emit("正在分割，请稍后", "提示")
            return
        threading.Thread(target=self.cut_button, daemon=False, args=()).start()  # 执行完函数就销毁线程

    def cut_button(self):

        print("from cut_button: dir path",self.dir_path, type(self.dir_path))

        if self.dir_path is None or self.dir_path == "" or self.file_len <= 0:
            self.msg.signal.emit("没有图片文件", "提示")
            return
        self.cutting = True  # 相当于一个锁

        path = cutPictureToPiece.cut(self.dir_path)
        self.save_path = path
        single_list = cutPieceToSingle.cut(path)  # 绝对路径
        self.single_paths = single_list

        print("from cut_button:分割成功!")

        # 获取分数栏图片的路径存入file_cut_list
        self.file_cut_list = []
        for root, dirs, files in os.walk(path):
            for filename in files:
                # 文件名列表，包含完整路径
                td = os.path.join(root, filename)
                if td.endswith(('jpg', 'png', 'jpeg')):  # 是图片才加
                    self.file_cut_list.append(os.path.join(root, filename))

        self.file_cut_len = len(self.file_cut_list)  # 切分出的分数栏数量

        if (self.file_cut_len != self.file_len):  # 切分出的分数栏数量和试卷数量不同
            self.msg.signal.emit("警告", "分割出的图片数量和原图片数量不匹配")
        else:
            print("from cut_button:分割匹配!")

        self.examCutPic.setPixmap(QtGui.QPixmap(self.file_cut_list[self.file_pointer]))
        self.cut_ended.signal.emit()  # 替代 self.show_single(self.file_pointer)
        self.cutting = False

    def detect_button_st(self):
        if self.detecting or self.cutting:
            self.msg.signal.emit("正在处理，请稍后", "提示")
            return
        threading.Thread(target=self.detect_button, daemon=False, args=()).start()  # 执行完函数就销毁线程

    def detect_button(self):
        if self.dir_path is None:
            self.msg.signal.emit("没有图片文件", "提示")
            return
        if self.single_paths is None:
            self.msg.signal.emit("图片未分割", "提示")
            return
        self.detected = False  # 为了保存按钮
        self.detecting = True
        single_paths = self.single_paths
        print(single_paths)

        nums = []  # 每张试卷分数
        for pos, path in enumerate(single_paths):
            imgs, num = [], []  # nums = imgs = [] imgs 和 nums是一个东西
            # num为单个试卷分数
            for i in os.listdir(path):
                img = cv.imread(os.path.join(path, i))
                imgs.append(img)
            try:  # 识别出问题能看到
                res = self.model(imgs, size=416).xywh
            except BaseException as er:
                print(f"\033[1;33;41m{1}识别出错\n\033[0m")
                print(er)
                nums.append([-1 for i in range(15)])
                continue

            for i in range(15):
                if i < len(res):
                    c = res[i][res[i][:, 0].argsort()]  # 按第0列排序 https://codeantenna.com/a/91tbTDDuvl
                    if len(c) == 0:  # 没有识别到数字
                        num.append(-1)  # -1为未识别到分数
                        continue
                    value = 0
                    for j in range(len(c)):
                        value = value * 10 + int(c[j][5])
                    num.append(value)
                else:
                    num.append(-1)  # 为了对齐，以便GUI上没有数字框的地方对应的数字栏可以修改。
            nums.append(num)

        self.score_list = nums

        print(self.file_pointer)
        # self.show_score(self.file_pointer)  # bug，子线程更新主线程界面
        self.detect_ended.signal.emit()  # 使用主线程更新
        self.detected = True
        self.detecting = False

    def show_score(self,pos=None):
        """
        在可编辑分数框展示分数，还展示总分
        :param pos: 展示的图片的指针
        """
        if(pos == None):
            pos = self.file_pointer
        print("pos-------------", pos)
        total = 0

        for i in range(15):  # 先清空框子:
            sc_i = getattr(self, "sc" + str(i))
            sc_i.setText("")

        for i in range(len(self.score_list[pos])):  # 遍历对应行（试卷）的分数
            name = "sc" + str(i)
            try:
                sc_i = getattr(self, name)
            except BaseException as er:
                print(er)
                return
            string = ""
            if(self.score_list[pos][i] != -1):
                total += self.score_list[pos][i]
                string = str(self.score_list[pos][i])
            sc_i.setText(string)  # 只接受string类型

        self.totalSc.setText(str(total))

    def on_text_edited(self):
        """
        当text被修改时执行更新分数score_list和总分totalSc
        """
        if (not self.detected) or self.detecting or self.cutting:
            return
        total = 0
        for i in range(15):
            try:
                sc_i = getattr(self, "sc" + str(i))
                if sc_i.text() != '':  # 框子不为空
                    total += int(sc_i.text())
                    self.score_list[self.file_pointer][i] = int(sc_i.text())
                else:  # 为空就是设-1，保存时自动保存空
                    self.score_list[self.file_pointer][i] = -1

            except BaseException as er:
                print(er)
        self.totalSc.setText(str(total))

    def save_result(self):
        """
        保存txt识别结果，格式为逗号分隔， 主进程中调用
        """
        print(self.detected)
        if not self.detected:
            self.qmbox.information(self,'提示','分数未识别')
            return
        string = os.path.join(self.dir_path, 'results')
        self.qmbox.information(self, '提示', '保存至' + string)
        self.res_path = string  # txt存储路径

        newPath(self.res_path)
        # 遍历file_list取名字，存储
        for i, path in enumerate(self.file_list):
            try:
                name = osp.splitext(osp.split(path)[1])[0] + '.txt'  # split分割文件名+后缀，splitext分割出文件名
                # name是图片名+txt后缀

                res, total = '', 0
                for j in self.score_list[i]:
                    if j == -1:  # 分数为空
                        j = ' '
                    else:
                        total += j
                    res += str(j) + ','
                res += str(total)
                with open(os.path.join(self.res_path, name), 'w') as f:
                    f.write(res)
            except BaseException as er:
                print(er)
        try:
            os.startfile(self.res_path)  # 打开文件夹
        except BaseException as er:
            print(er)

    def file_pointer_plus(self):
        """
        安全的文件指针（位置）加
        """
        if self.file_len == 0:
            QMessageBox.information(self, '提示', '没有图片文件')
            return False
        self.file_pointer += 1
        if (self.file_pointer == self.file_len):
            self.file_pointer -= 1
        return True

    def file_pointer_minus(self):
        """
        安全的文件指针（位置）减
        """
        if self.file_len == 0:
            QMessageBox.information(self, '提示', '没有图片文件')
            return False
        self.file_pointer -= 1
        if (self.file_pointer == -1):
            self.file_pointer += 1
        return True


    def show_single(self, pos=0):
        """
        展示第{pos}个图片切分出的single（分数框）到pic{i}
        """
        path = self.save_path + "_num" + str(pos + 1).zfill(4)  # 耦合cutPieceToSingle.cut的保存路径

        print("show single", path)
        single_list = []

        for root, dirs, files in os.walk(path):
            for filename in files:
                # 文件名列表，包含完整路径
                td = os.path.join(root, filename)
                if td.endswith(('jpg', 'png', 'jpeg')):  # 是图片才加
                    single_list.append(os.path.join(root, filename))

        n = len(single_list)

        for i in range(15):
            name = "pic" + str(i)
            try:
                pici = getattr(self, name)  # 反射,遍历pic0到picn-1
                if i < n:
                    pici.setPixmap(QtGui.QPixmap(single_list[i]))
                else:
                    pici.setPixmap(QtGui.QPixmap(":/pic/pic/ph_white.png"))
            except BaseException as er:
                print(er)
                self.msg.signal.emit(str(er),"提示")
                return


def init():
    # 打包之前在其他路径检查是否能运行, 下面的插入都是为了解决路径问题

    sys.path.insert(0,os.path.dirname(__file__))  # 插入当前目录作为Path，主要是为了找到yolov5
    main_dir = os.path.dirname(__file__)
    yolo_path = os.path.join(main_dir,'yolov5')
    print(yolo_path)
    sys.path.insert(0, yolo_path)  # 插入yolopath
    model = torch.hub.load(yolo_path, 'custom',
                           path=os.path.join(main_dir,'weights/best.pt'),
                           autoshape=True,  # 将model改成AutoShape对象（也是继承nn.module）
                           source='local',
                           force_reload=True,
                           )
    if torch.cuda.is_available():
        model.cuda()
    model.eval()  # 关闭比如dropout，batchNorm等预测时不需要的层

    return model

class CommonHelper:
    """
    帮助设置qss
    """
    def __init__(self):
        pass
    @staticmethod
    def readQss(style):
        with open(style, 'r') as f:
            return f.read()


if __name__ == '__main__':

    app = QApplication(sys.argv)

    model = init()
    main_dir = os.path.dirname(__file__)
    styleFile = os.path.join(main_dir,'qss/aqua.qss')

    style = CommonHelper.readQss(styleFile)

    myWin = MyWindow(model)
    myWin.setStyleSheet(style)

    myWin.show()
    sys.exit(app.exec_())

