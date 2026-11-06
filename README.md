# score-ear

English | [繁體中文](README.zh-TW.md)

Upload a recording, get sheet music back. score-ear detects the notes and exports staff notation and numbered (jianpu) notation. It runs locally with a browser UI, so your audio never leaves your machine. It handles solo recordings and full mixes (the mix is first split into vocals / drums / bass / other with AI, then each stem is transcribed).

> Automatic transcription is never perfect. Treat the output as a starting point and correct it yourself.

## Features

- **Input**: WAV / FLAC / OGG / MP3 / M4A / MP4
- **Modes**
  - Solo: `piano`, `vocal`, `guitar`, `other`, `auto`
  - Multi-instrument: demucs separates 4 stems, each transcribed separately
- **Options**: BPM override, time signature (auto or manual), quantization grid, use GPU, harmonic suppression (on by default; only affects solo `auto` and multi-instrument mode)
- **Output**: MusicXML (opens in MuseScore and other notation apps), MIDI, jianpu SVG, staff SVG, JSON

## Requirements

- Windows 11 (only tested there; the code is not Windows-specific, but the setup script is PowerShell)
- Python 3.10 or 3.11 (3.12+ not verified for every dependency)
- An NVIDIA GPU is recommended: piano and multi-instrument modes are much faster on GPU. CPU works, just slower.
- Disk: a few GB for the virtual environment and models

## Install

```powershell
git clone https://github.com/shaoen901109/score-ear.git
cd score-ear
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

`setup_env.ps1` creates a `.venv` and installs everything:

- Regular packages come from the official PyPI. Use a mirror with `setup_env.ps1 -Mirror <url>`.
- PyTorch is installed from the CUDA 12.8 index (RTX 50-series GPUs need cu128+). **Without an NVIDIA GPU**, change `$TORCH_INDEX` in the script to `https://download.pytorch.org/whl/cpu` first.
- You can also install manually from `requirements.txt`; `requirements-lock.txt` pins the versions used by the author.

## Usage

```powershell
run_dev.bat
```

or `.venv\Scripts\python -m app.main`. This starts a local server (default `http://localhost:8000`, or the next free port) and opens your browser. Flags: `--port <port>`, `--no-browser`.

Pick an audio file and a mode, submit, wait for the job to finish, then preview and download the results.

The UI and console messages are in Traditional Chinese.

### Models downloaded on first run

| Mode | First-run download | Location |
|---|---|---|
| Solo `auto` / `vocal` / `guitar` / `other` | none (basic-pitch model ships with the package) | — |
| Solo `piano` | ~165 MB piano transcription model | `%USERPROFILE%\piano_transcription_inference_data\` |
| Multi-instrument | ~350 MB demucs model | torch hub cache |

Models are downloaded by their own packages and have their own licenses (see Third-party components).

## How it works

- Source separation: demucs (4 stems)
- Polyphonic transcription: basic-pitch (ONNX backend, no TensorFlow needed)
- Piano: piano_transcription_inference (ByteDance high-resolution piano transcription)
- Monophonic refinement: torchcrepe (vocal pitch tracking)
- Tempo and rhythm: librosa onset detection, with manual BPM override
- Key detection: Krumhansl-Schmuckler pitch-class profiles
- Staff rendering: verovio (MusicXML → SVG)
- Jianpu rendering: custom SVG (scale degrees, octave dots, accidentals, dots, durations)

See `docs/ARCHITECTURE.md` for module contracts and data model, and `docs/ACCURACY.md` for the accuracy evaluation (both in Traditional Chinese).

## Known limitations

- `auto` mode is sensitive to overtones and tends to add extra notes; use `vocal` for singing and `piano` for piano.
- Drum classification uses band-energy rules, not a trained classifier. On the synthetic test mix it reaches about 80% F1 (runs vary by ±5–15 points); dense hi-hat and snare in real recordings can still be confused.
- demucs separation is unstable on synthetic test audio; real recordings fare better.
- verovio cannot load its bundled fonts from a path with non-ASCII characters; the app copies them to `%LOCALAPPDATA%\ScoreEar\` automatically.

## Tests

```powershell
.\.venv\Scripts\python -m pytest tests -q -m "not slow"
```

Tests marked `slow` download models and are excluded by default.

## Credits

Built on these open-source projects and research:

| Purpose | Project | License |
|---|---|---|
| Source separation | [Demucs](https://github.com/adefossez/demucs) (Meta AI Research) | MIT |
| Polyphonic transcription | [Basic Pitch](https://github.com/spotify/basic-pitch) (Spotify) | Apache-2.0 |
| Piano transcription | [piano_transcription_inference](https://github.com/qiuqiangkong/piano_transcription_inference) (Kong et al., ByteDance high-resolution piano transcription) | see upstream |
| Pitch tracking | [torchcrepe](https://github.com/maxrmorrison/torchcrepe) (PyTorch implementation of CREPE) | MIT |
| Audio analysis | [librosa](https://librosa.org/) | ISC |
| Staff rendering | [Verovio](https://www.verovio.org/) | LGPL-3.0 |
| Deep learning | [PyTorch](https://pytorch.org/), [ONNX Runtime](https://onnxruntime.ai/) | BSD-3-Clause, MIT |
| UI and server | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [pywebview](https://pywebview.flowrl.com/) | MIT, BSD-3-Clause, BSD-3-Clause |
| MIDI | [mido](https://github.com/mido/mido), [pretty_midi](https://github.com/craffel/pretty-midi) | MIT |
| Key detection algorithm | Krumhansl-Schmuckler pitch-class profiles (Krumhansl, 1990) | — |

This repo contains only its own code and no model files. Model weights are downloaded by those packages from their official sources under their own licenses. Only transcribe recordings you have the right to use.

Developed with the help of [Claude Code](https://claude.com/claude-code) (Anthropic).

## Author

- Website: https://shaoenstudio.com
- Email: shaoenstudio@gmail.com
- GitHub: [@shaoen901109](https://github.com/shaoen901109)

## License

MIT, see [LICENSE](LICENSE).
