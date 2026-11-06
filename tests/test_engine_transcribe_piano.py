# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("piano_transcription_inference")

pytestmark = pytest.mark.slow


def test_checkpoint_path_lazy_and_dir_created():
    from engine.paths import models_dir
    from engine.transcribe_piano import _checkpoint_path

    path = _checkpoint_path()
    assert path.endswith(".pth")
    assert (models_dir() / "piano_transcription").exists()
    assert Path(path).parent == models_dir() / "piano_transcription"
