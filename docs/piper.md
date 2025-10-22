![Piper](etc/logo.png)

A fast and local neural text-to-speech engine that embeds [espeak-ng][] for phonemization.

Install with:

``` sh
pip install piper-tts
```

* 🎧 [Samples][samples]
* 💡 [Demo][demo]
* 🗣️ [Voices][voices]
* 🖥️ [Command-line interface][cli]
* 🌐 [Web server][api-http]
* 🐍 [Python API][api-python]
* 🔧 [C/C++ API][libpiper]
* 🏋️ [Training new voices][training]
* 🛠️ [Building manually][building]

---

People/projects using Piper:

* [Home Assistant](https://github.com/home-assistant/addons/blob/master/piper/README.md)
* [NVDA - NonVisual Desktop Access](https://www.nvaccess.org/post/in-process-8th-may-2023/#voices)
* [Image Captioning for the Visually Impaired and Blind: A Recipe for Low-Resource Languages](https://www.techrxiv.org/articles/preprint/Image_Captioning_for_the_Visually_Impaired_and_Blind_A_Recipe_for_Low-Resource_Languages/22133894)
* [Video tutorial by Thorsten Müller](https://youtu.be/rjq5eZoWWSo)
* [Open Voice Operating System](https://github.com/OpenVoiceOS/ovos-tts-plugin-piper)
* [JetsonGPT](https://github.com/shahizat/jetsonGPT)
* [LocalAI](https://github.com/go-skynet/LocalAI)
* [Lernstick EDU / EXAM: reading clipboard content aloud with language detection](https://lernstick.ch/)
* [Natural Speech - A plugin for Runelite, an OSRS Client](https://github.com/phyce/rl-natural-speech)
* [mintPiper](https://github.com/evuraan/mintPiper)
* [Vim-Piper](https://github.com/wolandark/vim-piper)
* [POTaTOS](https://www.youtube.com/watch?v=Dz95q6XYjwY)
* [Narration Studio](https://github.com/phyce/Narration-Studio)
* [Basic TTS](https://basictts.com/) - Simple online text-to-speech converter.

[![A library from the Open Home Foundation](https://www.openhomefoundation.org/badges/ohf-library.png)](https://www.openhomefoundation.org/)

<!-- Links -->
[espeak-ng]: https://github.com/espeak-ng/espeak-ng
[cli]: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md
[api-http]: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_HTTP.md
[api-python]: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
[training]: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md
[building]: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/BUILDING.md
[voices]: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md
[samples]: https://rhasspy.github.io/piper-samples
[demo]: https://rhasspy.github.io/piper-samples/demo.html
[libpiper]: https://github.com/OHF-Voice/piper1-gpl/tree/main/libpiper



===

# 🐍 Python API

Install with:

``` sh
pip install piper-tts
```

Download a voice, for example:

``` sh
python3 -m piper.download_voices en_US-lessac-medium
```

Use `PiperVoice.synthesize_wav`:

``` python
import wave
from piper import PiperVoice

voice = PiperVoice.load("/path/to/en_US-lessac-medium.onnx")
with wave.open("test.wav", "wb") as wav_file:
    voice.synthesize_wav("Welcome to the world of speech synthesis!", wav_file)
```

Adjust synthesis:

``` python
syn_config = SynthesisConfig(
    volume=0.5,  # half as loud
    length_scale=2.0,  # twice as slow
    noise_scale=1.0,  # more audio variation
    noise_w_scale=1.0,  # more speaking variation
    normalize_audio=False, # use raw audio from voice
)

voice.synthesize_wav(..., syn_config=syn_config)
```

To use CUDA for GPU acceleration:

``` python
voice = PiperVoice.load(..., use_cuda=True)
```

This requires the `onnxruntime-gpu` package to be installed.

For streaming, use `PiperVoice.synthesize`:

``` python
for chunk in voice.synthesize("..."):
    set_audio_format(chunk.sample_rate, chunk.sample_width, chunk.sample_channels)
    write_raw_data(chunk.audio_int16_bytes)
```


===

