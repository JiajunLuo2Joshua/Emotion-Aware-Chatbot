from collections import Counter

# 从 main.py 中提取的纯逻辑，不需要 GUI
LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

SUGGESTIONS = {
    "Happy": "💡 You seem joyful! What's something fun or exciting that happened today?",
    "Sad": "💡 I'm here for you. Would you like to talk about something or someone that brings you comfort?",
    "Anger": "💡 It's okay to feel frustrated. Want to tell me what happened or what's been bothering you?",
    "Neutral": "💡 Just checking in — is there anything on your mind you'd like to share today?",
    "Surprise": "💡 That caught you off guard! Want to tell me what just happened?",
    "Fear": "💡 You're not alone. Would it help to talk about what's making you uneasy right now?",
    "Disgust": "💡 That didn't sit right with you, huh? Want to switch topics or tell me what happened?"
}

def compute_vote(vote_buffer, threshold=0.8):
    """从 main.py 的 vote_emotion 中提取的核心投票逻辑"""
    if not vote_buffer:
        return None
    counter = Counter(vote_buffer)
    most_common, count = counter.most_common(1)[0]
    ratio = count / len(vote_buffer)
    if ratio >= threshold:
        return most_common
    return None

# --- 测试 ---

def test_unanimous_vote():
    """全部一致 → 返回该情绪"""
    buf = ["Happy"] * 10
    assert compute_vote(buf) == "Happy"

def test_vote_below_threshold():
    """没有达到 80% → 返回 None"""
    buf = ["Happy"] * 7 + ["Sad"] * 3  # 70% Happy
    assert compute_vote(buf) is None

def test_vote_at_exact_threshold():
    """刚好 80% → 通过"""
    buf = ["Sad"] * 8 + ["Happy"] * 2  # 80% Sad
    assert compute_vote(buf) == "Sad"

def test_vote_empty_buffer():
    assert compute_vote([]) is None

def test_all_labels_have_suggestions():
    """每个情绪都有对应建议文本"""
    for label in LABELS:
        assert label in SUGGESTIONS

def test_emotion_history_max_length():
    """最多保留 6 条历史"""
    recent = []
    for e in ["Happy", "Sad", "Angry", "Neutral", "Fear", "Surprise", "Disgust", "Happy"]:
        recent.append(e)
        if len(recent) > 6:
            recent.pop(0)
    assert len(recent) == 6
    assert recent[0] == "Angry"  # 最早的两个被移除
