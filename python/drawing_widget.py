from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, QPoint, QSize, pyqtSignal, QRect
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QImage, QCursor
import numpy as np


class DrawingWidget(QWidget):
    imageUpdated = pyqtSignal(np.ndarray)  # 当绘图更新时发射信号

    def __init__(self, size=512, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 维度参数
        self._3d_enabled = False
        self._z_depth = 1

        # 初始化画布（白底黑笔）
        self.image = QImage(QSize(size, size), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        # 绘图参数
        self.drawing = False
        self.last_point = QPoint()
        self.current_tool = "pen"  # pen/line/rect/fill
        self.pen_width = 2
        self._init_pens()

        # 在初始化时启用抗锯齿
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_PaintOnScreen)

    def _init_pens(self):
        # """初始化绘图工具"""
        self.black_pen = QPen(Qt.black, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.fill_brush = QBrush(Qt.black)
        self.temp_pixmap = None  # 用于临时绘制

    def setTool(self, tool_name):
        # """设置当前绘图工具"""
        self.current_tool = tool_name
        self.update()

    def set_3d_params(self, is_3d, z_depth):
        # """接收来自MainWindow的三维参数"""
        self._3d_enabled = is_3d
        self._z_depth = z_depth

    def getImageArray(self):
        # """将QImage转换为numpy数组（HxW）"""
        h, w = self.image.height(), self.image.width()
        arr = np.zeros((h, w), dtype=np.float32)

        # 将QImage转换为灰度数组（1为黑，0为白）
        for y in range(h):
            for x in range(w):
                color = QColor(self.image.pixel(x, y))
                arr[y, x] = 1.0 if color == Qt.black else 0.0

        # 三维数组生成
        if self._3d_enabled:
            # 创建三维数组
            depth = self._z_depth
            # 沿第三轴复制二维图像
            return np.stack([arr] * depth, axis=2)
        else:
            return arr



    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

            if self.current_tool == "fill":
                self.floodFill(event.pos())
            else:
                # 创建临时绘图层
                self.temp_pixmap = self.image.copy()

    def mouseMoveEvent(self, event):
        if self.drawing and self.current_tool not in ["fill"]:
            current_point = event.pos()

            if self.current_tool == "pen":
                # 自由绘制模式
                painter = QPainter(self.image)
                painter.setPen(self.black_pen)
                painter.drawLine(self.last_point, current_point)
                painter.end()
                self.last_point = current_point
                self.update()
            else:
                # 实时预览
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False

            if self.current_tool in ["line", "rect","ellipse"]:
                # 完成最终绘制
                painter = QPainter(self.image)
                painter.setPen(self.black_pen)
                self._drawShape(painter, event.pos())
                painter.end()
                self.temp_pixmap = None
                self.update()

            # 发射更新信号
            self.imageUpdated.emit(self.getImageArray())

    def paintEvent(self, event):
        # """处理绘图更新"""
        super().paintEvent(event)
        painter = QPainter(self)

        # 绘制基础图像
        painter.drawImage(self.rect(), self.image)

        # 在paintEvent中使用高质量渲染
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 绘制临时预览
        if self.drawing and self.temp_pixmap:
            painter.drawImage(self.rect(), self.temp_pixmap)
            painter.setPen(self.black_pen)
            self._drawShape(painter, self.mapFromGlobal(QCursor.pos()))

    def _drawShape(self, painter, end_point):
        # """根据当前工具绘制形状"""
        start = self.last_point
        end = end_point

        if self.current_tool == "line":
            painter.drawLine(start, end)
        elif self.current_tool == "rect":
            rect = self._getNormalizedRect(start, end)
            painter.drawRect(rect)
        elif self.current_tool == "ellipse":
            ellipse = self._getNormalizedEllipse(start,end)
            painter.drawEllipse(ellipse)


    def _getNormalizedRect(self, p1, p2):
        # """获取标准化矩形（处理任意方向）"""
        return QRect(
            QPoint(min(p1.x(), p2.x()), min(p1.y(), p2.y())),
            QPoint(max(p1.x(), p2.x()), max(p1.y(), p2.y()))
        )

    def _getNormalizedEllipse(self, p1, p2):
        # """获取标准化椭圆形
        return QRect(
            QPoint(min(p1.x(), p2.x()), min(p1.y(), p2.y())),
            QPoint(max(p1.x(), p2.x()), max(p1.y(), p2.y()))
        )

    def floodFill(self, pos):
        # """泛洪填充算法（使用扫描线算法）"""
        target_color = self.image.pixelColor(pos)
        if target_color == Qt.black:
            return  # 已经是黑色

        # 初始化队列
        stack = [(pos.x(), pos.y())]
        visited = set()

        # 边界检查
        w, h = self.image.width(), self.image.height()

        # 开始填充
        painter = QPainter(self.image)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.fill_brush)

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            # 向左扩展
            left = x
            while left >= 0 and self.image.pixelColor(left, y) == target_color:
                left -= 1
            left += 1

            # 向右扩展
            right = x
            while right < w and self.image.pixelColor(right, y) == target_color:
                right += 1

            # 绘制水平线
            painter.drawRect(left, y, right - left, 1)

            # 检查上下行
            for dy in [-1, 1]:
                ny = y + dy
                if 0 <= ny < h:
                    for nx in range(left, right):
                        if self.image.pixelColor(nx, ny) == target_color:
                            stack.append((nx, ny))

        painter.end()
        self.update()
        self.imageUpdated.emit(self.getImageArray())

    def clear(self):
        # 清空画布
        self.image.fill(Qt.white)
        self.update()
        self.imageUpdated.emit(self.getImageArray())