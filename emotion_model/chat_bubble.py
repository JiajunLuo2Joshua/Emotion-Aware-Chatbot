from PyQt5 import QtWidgets, QtGui, QtCore
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class ChatBubble(QtWidgets.QWidget):
    def __init__(self, role, message):
        super().__init__()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        # Avatar with rounded clipping
        avatar = QtWidgets.QLabel()
        avatar.setFixedSize(44, 44)
        avatar_filename = "avatar_user.png" if role == "user" else "avatar_bot.png"
        avatar_path = os.path.join(BASE_DIR, avatar_filename)
        pixmap = QtGui.QPixmap(avatar_path)
        if not pixmap.isNull():
            # Create circular avatar
            rounded = QtGui.QPixmap(44, 44)
            rounded.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(rounded)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            path = QtGui.QPainterPath()
            path.addEllipse(0, 0, 44, 44)
            painter.setClipPath(path)
            painter.drawPixmap(0, 0, pixmap.scaled(44, 44, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation))
            painter.end()
            avatar.setPixmap(rounded)
        avatar.setStyleSheet("background: transparent;")

        # Chat bubble
        bubble = QtWidgets.QLabel(message)
        bubble.setWordWrap(True)
        bubble.setMaximumWidth(1600)
        bubble.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        if role == "user":
            bubble.setStyleSheet("""
                QLabel {
                    background-color: #dcf8c6;
                    border-radius: 16px;
                    border-top-right-radius: 4px;
                    padding: 14px 16px;
                    font-size: 22px;
                    color: #1a1a1a;
                    line-height: 1.4;
                }
            """)
            layout.addStretch()
            layout.addWidget(bubble)
            layout.addWidget(avatar, alignment=QtCore.Qt.AlignTop)
        else:
            bubble.setStyleSheet("""
                QLabel {
                    background-color: #ffffff;
                    border-radius: 16px;
                    border-top-left-radius: 4px;
                    padding: 14px 16px;
                    font-size: 22px;
                    color: #1a1a1a;
                    line-height: 1.4;
                    border: 1px solid #e8ecf0;
                }
            """)
            layout.addWidget(avatar, alignment=QtCore.Qt.AlignTop)
            layout.addWidget(bubble)
            layout.addStretch()

        self.setStyleSheet("background: transparent;")
        self.setLayout(layout)
