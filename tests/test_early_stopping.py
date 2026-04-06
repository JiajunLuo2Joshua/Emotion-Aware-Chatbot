import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'emotion_model'))

from train import EarlyStopping

def test_first_call_saves_checkpoint(tmp_path):
    """第一次调用时保存模型"""
    es = EarlyStopping(patience=3)
    model = torch.nn.Linear(10, 2)
    path = str(tmp_path / "model.pt")

    es(val_acc=0.5, model=model, path=path)

    assert os.path.exists(path)
    assert es.best_score == 0.5
    assert es.early_stop is False

def test_improvement_resets_counter(tmp_path):
    """精度提升 → 计数器归零"""
    es = EarlyStopping(patience=3)
    model = torch.nn.Linear(10, 2)
    path = str(tmp_path / "model.pt")

    es(0.5, model, path)
    es(0.4, model, path)  # 下降
    es(0.4, model, path)  # 下降
    assert es.counter == 2

    es(0.6, model, path)  # 提升！
    assert es.counter == 0
    assert es.best_score == 0.6

def test_early_stop_triggers_after_patience(tmp_path):
    """连续下降超过 patience 次 → 触发早停"""
    es = EarlyStopping(patience=3)
    model = torch.nn.Linear(10, 2)
    path = str(tmp_path / "model.pt")

    es(0.9, model, path)  # best
    es(0.8, model, path)  # counter=1
    es(0.7, model, path)  # counter=2
    assert es.early_stop is False

    es(0.6, model, path)  # counter=3 → 触发
    assert es.early_stop is True

def test_equal_score_treated_as_not_worse(tmp_path):
    """相同精度不算下降（代码中 score < best 才算下降，等于时走 else 分支重置计数器）"""
    es = EarlyStopping(patience=2)
    model = torch.nn.Linear(10, 2)
    path = str(tmp_path / "model.pt")

    es(0.5, model, path)
    es(0.5, model, path)  # score == best_score, 走 else 分支
    assert es.counter == 0  # 没有下降，计数器不增加
