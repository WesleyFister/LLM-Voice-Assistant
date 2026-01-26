from openwakeword.model import Model
from scipy.signal import resample_poly
import openwakeword
import pyaudio
import numpy as np

def wakeWord(model):
    if model == "alexa":
        openwakeword.utils.download_models(['alexa_v0.1'])
        model = Model(wakeword_models = ["alexa"])

    elif model == "hey mycroft":
        openwakeword.utils.download_models(['hey_mycroft_v0.1'])
        model = Model(wakeword_models = ["hey mycroft"])

    elif model == "hey jarvis":
        openwakeword.utils.download_models(['hey_jarvis_v0.1'])
        model = Model(wakeword_models = ["hey jarvis"])

    else:
        model = Model(wakeword_models = model)

    p = pyaudio.PyAudio()

    default_input_info = p.get_default_input_device_info()
    FORMAT = pyaudio.paInt16
    RATE = int(default_input_info['defaultSampleRate'])  # e.g., 44100, 48000, etc.
    CHANNELS = default_input_info['maxInputChannels']    # e.g., 1 or 2
    CHUNK = int(default_input_info['defaultSampleRate'] / 10)

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Listening for wake word...")
    sentinel = False
    while True:
        # Read raw audio
        data = stream.read(CHUNK, False)
        audio = np.frombuffer(data, dtype=np.int16)

        # Convert to mono if multi-channel
        if CHANNELS > 1:
            audio = audio.reshape(-1, CHANNELS).mean(axis=1)

        # Downsample to 16kHz
        if RATE != 16000:
            audio = resample_poly(audio, 16000, RATE)

        prediction = model.predict(x=audio, threshold={'hey jarvis': 0.9}, debounce_time=5.0)

        for mdl in model.prediction_buffer.keys():
            scores = list(model.prediction_buffer[mdl])
            curr_score = format(scores[-1], '.20f').replace("-", "")
            
            if float(curr_score) >= 0.9:
                print("Wake word detected!")
                
                sentinel = True
                break
                
        if sentinel == True:
            stream.stop_stream()
            stream.close()
            break