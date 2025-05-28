from PyQt5 import QtWidgets, QtGui, QtCore

class ChatBubble(QtWidgets.QWidget):
    def __init__(self, role, message):
        super().__init__()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # 头像
        avatar = QtWidgets.QLabel()
        avatar.setFixedSize(40, 40)
        avatar_path = "avatar_user.png" if role == "user" else "avatar_bot.png"
        pixmap = QtGui.QPixmap(avatar_path)
        avatar.setPixmap(pixmap.scaled(40, 40, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        # 对话气泡
        bubble = QtWidgets.QLabel(message)
        bubble.setWordWrap(True)
        bubble.setMaximumWidth(1800)
        #bubble.setMinimumWidth(120)
        #bubble.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        

        if role == "user":
            bubble.setStyleSheet("""
                QLabel {
                    background-color: #e1ffc7;
                    border-radius: 14px;
                    padding: 12px;
                    font-size: 28px;
                }
            """)
            layout.addStretch()
            layout.addWidget(bubble)
            layout.addWidget(avatar)
            
            
        else: 
            bubble.setStyleSheet("""
                QLabel {
                    background-color: #ffffff;
                    border-radius: 14px;
                    padding: 12px;
                    font-size: 28px;
                }
            """)
            layout.addWidget(avatar)  
            layout.addWidget(bubble) 
            layout.addStretch()

        self.setLayout(layout)