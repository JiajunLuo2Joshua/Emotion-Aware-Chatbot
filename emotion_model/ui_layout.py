from PyQt5 import QtWidgets, QtCore

def setup_ui(self):
    layout = QtWidgets.QVBoxLayout(self)
    top_layout = QtWidgets.QHBoxLayout()

    self.video_label = QtWidgets.QLabel()
    self.video_label.setFixedSize(960, 540)
    self.video_label.setSizePolicy(
        QtWidgets.QSizePolicy.Fixed,
        QtWidgets.QSizePolicy.Fixed
    )
    top_layout.addWidget(self.video_label)

    top_right_layout = QtWidgets.QVBoxLayout()

    self.emotion_label = QtWidgets.QLabel("Emotion: Unknown")
    self.emotion_label.setStyleSheet("""
        background-color: #e6ffe6;
        font-size: 36px;
        font-weight: bold;
        border: 2px solid #b2fab4;
        border-radius: 12px;
        padding: 12px;
    """)
    self.emotion_label.setAlignment(QtCore.Qt.AlignCenter)
    top_right_layout.addWidget(self.emotion_label, stretch=1)

    self.emotion_history_label = QtWidgets.QLabel("Recent Emotions:\n(none yet)")
    self.emotion_history_label.setStyleSheet("""
        background-color: #f0f8ff;
        border: 2px solid #a0c4ff;
        border-radius: 12px;
        padding: 10px;
        font-size: 32px;
        font-weight: bold;
    """)
    self.emotion_history_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
    top_right_layout.addWidget(self.emotion_history_label, stretch=1)

    self.suggestion_box = QtWidgets.QLabel("💡 Tip: Try talking about a happy memory!")
    self.suggestion_box.setStyleSheet("""
        background-color: #fff8dc;
        border: 1px solid #ffe4b5;
        border-radius: 10px;
        padding: 10px;
        font-size: 32px;
        font-weight: bold;
    """)
    self.suggestion_box.setWordWrap(True)
    top_right_layout.addWidget(self.suggestion_box, stretch=1)

    layout.addStretch()
    top_layout.addLayout(top_right_layout)
    layout.addLayout(top_layout, stretch=1)

    self.scroll_area = QtWidgets.QScrollArea()
    self.scroll_area.setWidgetResizable(True)
    self.chat_container = QtWidgets.QWidget()
    self.chat_layout = QtWidgets.QVBoxLayout(self.chat_container)
    self.chat_layout.setAlignment(QtCore.Qt.AlignTop)
    self.chat_container.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred,
        QtWidgets.QSizePolicy.Expanding
    )
    self.scroll_area.setWidget(self.chat_container)
    layout.addWidget(self.scroll_area, stretch=1)

    input_layout = QtWidgets.QHBoxLayout()
    self.input_line = QtWidgets.QTextEdit()
    self.input_line.setStyleSheet("font-size: 32px; padding: 8px;")
    self.input_line.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding,
        QtWidgets.QSizePolicy.Expanding
    )

    self.send_btn = QtWidgets.QPushButton("SEND")
    self.send_btn.setStyleSheet("font-size: 24px;")
    self.send_btn.setFixedSize(240, 48)
    self.send_btn.clicked.connect(self.handle_text_input)

    self.mode = "text"
    self.toggle_btn = QtWidgets.QPushButton("🎤 Switch to Voice")
    self.toggle_btn.setStyleSheet("font-size: 24px;")
    self.toggle_btn.setFixedSize(240, 48)
    self.toggle_btn.clicked.connect(self.toggle_input_mode)

    button_layout = QtWidgets.QVBoxLayout()
    button_layout.addStretch()
    button_layout.addWidget(self.send_btn, alignment=QtCore.Qt.AlignCenter)
    button_layout.addStretch()
    button_layout.addWidget(self.toggle_btn, alignment=QtCore.Qt.AlignCenter)
    button_layout.addStretch()

    input_layout.addWidget(self.input_line, stretch=1)
    input_layout.addLayout(button_layout)

    input_container = QtWidgets.QWidget()
    input_container.setLayout(input_layout)
    input_container.setContentsMargins(10, 10, 10, 20)
    layout.addWidget(input_container, stretch=1)
