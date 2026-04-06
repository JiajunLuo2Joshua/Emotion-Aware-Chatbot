from PyQt5 import QtWidgets, QtCore, QtGui

def setup_ui(self):
    # Global window style
    self.setStyleSheet("""
        QWidget {
            background-color: #f5f7fa;
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        }
        QScrollArea {
            border: none;
            background-color: #eef2f7;
        }
        QScrollBar:vertical {
            width: 8px;
            background: transparent;
        }
        QScrollBar::handle:vertical {
            background: #c0c8d4;
            border-radius: 4px;
            min-height: 30px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """)

    layout = QtWidgets.QVBoxLayout(self)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(12)

    # ===== Top section: Video + Info panel =====
    top_layout = QtWidgets.QHBoxLayout()
    top_layout.setSpacing(16)

    # Video feed with rounded border
    self.video_label = QtWidgets.QLabel()
    self.video_label.setFixedSize(960, 540)
    self.video_label.setStyleSheet("""
        QLabel {
            background-color: #1a1a2e;
            border: 2px solid #d0d7e3;
            border-radius: 16px;
        }
    """)
    self.video_label.setAlignment(QtCore.Qt.AlignCenter)
    top_layout.addWidget(self.video_label)

    # Right info panel
    top_right_layout = QtWidgets.QVBoxLayout()
    top_right_layout.setSpacing(12)

    # Emotion label
    self.emotion_label = QtWidgets.QLabel("😶  Emotion: Unknown")
    self.emotion_label.setStyleSheet("""
        QLabel {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #e8f5e9, stop:1 #c8e6c9);
            font-size: 28px;
            font-weight: bold;
            color: #2e7d32;
            border: none;
            border-radius: 14px;
            padding: 16px;
        }
    """)
    self.emotion_label.setAlignment(QtCore.Qt.AlignCenter)
    top_right_layout.addWidget(self.emotion_label, stretch=1)

    # Emotion history
    self.emotion_history_label = QtWidgets.QLabel("📊  Recent Emotions:\n(none yet)")
    self.emotion_history_label.setStyleSheet("""
        QLabel {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #e3f2fd, stop:1 #bbdefb);
            border: none;
            border-radius: 14px;
            padding: 14px;
            font-size: 24px;
            font-weight: bold;
            color: #1565c0;
        }
    """)
    self.emotion_history_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
    self.emotion_history_label.setWordWrap(True)
    top_right_layout.addWidget(self.emotion_history_label, stretch=1)

    # Suggestion box
    self.suggestion_box = QtWidgets.QLabel("💡 Tip: Try talking about a happy memory!")
    self.suggestion_box.setStyleSheet("""
        QLabel {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #fff8e1, stop:1 #ffecb3);
            border: none;
            border-radius: 14px;
            padding: 14px;
            font-size: 24px;
            font-weight: bold;
            color: #e65100;
        }
    """)
    self.suggestion_box.setWordWrap(True)
    top_right_layout.addWidget(self.suggestion_box, stretch=1)

    top_layout.addLayout(top_right_layout)
    layout.addLayout(top_layout, stretch=0)

    # ===== Chat area =====
    self.scroll_area = QtWidgets.QScrollArea()
    self.scroll_area.setWidgetResizable(True)
    self.scroll_area.setStyleSheet("""
        QScrollArea {
            background-color: #eef2f7;
            border-radius: 14px;
        }
    """)

    self.chat_container = QtWidgets.QWidget()
    self.chat_container.setStyleSheet("background-color: #eef2f7;")
    self.chat_layout = QtWidgets.QVBoxLayout(self.chat_container)
    self.chat_layout.setAlignment(QtCore.Qt.AlignTop)
    self.chat_layout.setSpacing(6)
    self.chat_layout.setContentsMargins(10, 10, 10, 10)
    self.chat_container.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred,
        QtWidgets.QSizePolicy.Expanding
    )
    self.scroll_area.setWidget(self.chat_container)
    layout.addWidget(self.scroll_area, stretch=1)

    # ===== Input area =====
    input_layout = QtWidgets.QHBoxLayout()
    input_layout.setSpacing(12)

    self.input_line = QtWidgets.QTextEdit()
    self.input_line.setPlaceholderText("Type your message here...")
    self.input_line.setStyleSheet("""
        QTextEdit {
            font-size: 22px;
            padding: 10px 14px;
            border: 2px solid #d0d7e3;
            border-radius: 14px;
            background-color: #ffffff;
            color: #333333;
        }
        QTextEdit:focus {
            border: 2px solid #64b5f6;
        }
    """)
    self.input_line.setFixedHeight(80)

    # Button styles
    send_btn_style = """
        QPushButton {
            font-size: 20px;
            font-weight: bold;
            color: white;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #42a5f5, stop:1 #1e88e5);
            border: none;
            border-radius: 12px;
            padding: 10px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #64b5f6, stop:1 #42a5f5);
        }
        QPushButton:pressed {
            background: #1565c0;
        }
        QPushButton:disabled {
            background: #b0bec5;
            color: #eceff1;
        }
    """
    toggle_btn_style = """
        QPushButton {
            font-size: 20px;
            font-weight: bold;
            color: white;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #66bb6a, stop:1 #43a047);
            border: none;
            border-radius: 12px;
            padding: 10px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #81c784, stop:1 #66bb6a);
        }
        QPushButton:pressed {
            background: #2e7d32;
        }
        QPushButton:disabled {
            background: #b0bec5;
            color: #eceff1;
        }
    """

    self.send_btn = QtWidgets.QPushButton("📨  Send")
    self.send_btn.setStyleSheet(send_btn_style)
    self.send_btn.setFixedSize(200, 48)
    self.send_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
    self.send_btn.clicked.connect(self.handle_text_input)

    self.mode = "text"
    self.toggle_btn = QtWidgets.QPushButton("🎤  Voice Mode")
    self.toggle_btn.setStyleSheet(toggle_btn_style)
    self.toggle_btn.setFixedSize(200, 48)
    self.toggle_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
    self.toggle_btn.clicked.connect(self.toggle_input_mode)

    button_layout = QtWidgets.QVBoxLayout()
    button_layout.setSpacing(8)
    button_layout.addStretch()
    button_layout.addWidget(self.send_btn, alignment=QtCore.Qt.AlignCenter)
    button_layout.addWidget(self.toggle_btn, alignment=QtCore.Qt.AlignCenter)
    button_layout.addStretch()

    input_layout.addWidget(self.input_line, stretch=1)
    input_layout.addLayout(button_layout)

    input_container = QtWidgets.QWidget()
    input_container.setLayout(input_layout)
    input_container.setStyleSheet("""
        QWidget {
            background-color: #ffffff;
            border-radius: 16px;
        }
    """)
    input_container.setContentsMargins(14, 10, 14, 10)
    layout.addWidget(input_container, stretch=0)
