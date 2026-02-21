# coding=utf-8
"""SMARTEAR KWS - Jetson Nano Optimized Inference Pipeline.

Optimized version of KWS_jetson_mems (220901) with:
- Bug fixes (variable shadowing, undefined references, deprecated APIs)
- Proper argument types and error handling
- Logging instead of commented print statements
- Configurable parameters (block size, labels, channels)
- Memory optimization for Jetson Nano (4GB RAM)
- Clean resource management (no exit() calls)
- SOTA speech enhancement for low-SNR environments:
  * IMCRA noise estimation + LogPower spectral subtraction (pre-DTLN)
  * Decision-directed Wiener postfilter (post-DTLN)
  * Pre-emphasis / de-emphasis for high-freq SNR boost
  * Perceptual Mel-band weighting for KWS optimization
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import json
import logging
import math
import sys
import time

import numpy as np
import soundfile as sf
import tensorflow.compat.v1 as tf1
import tensorflow as tf
import tensorflow.lite as tflite
import webrtcvad

import audio_recorder
from kws_streaming.models import models
from speech_enhancement import SpeechEnhancementPipeline

# ---------------------------------------------------------------------------
# Hot-reload support for runtime model switching
# ---------------------------------------------------------------------------

class ModelManager:
    """Manages KWS model with hot-reload capability.

    Watches for model updates (e.g., from fine-tuning pipeline) and
    reloads the model without restarting the inference process.

    Usage:
        manager = ModelManager(model_path, labels_file)
        kws_model, labels = manager.load()
        # In inference loop:
        if manager.check_reload():
            kws_model = manager.model
            labels = manager.labels
    """

    def __init__(self, model_path, labels_file=None):
        self.model_path = model_path
        self.labels_file = labels_file
        self.model = None
        self.labels = None
        self._weights_mtime = 0
        self._active_model_mtime = 0

    def load(self):
        """Initial model load."""
        self.model = load_kws_model(self.model_path)
        if self.labels_file:
            self.labels = read_labels(self.labels_file)
        else:
            # Try labels.txt in model dir
            model_labels = os.path.join(self.model_path, 'labels.txt')
            if os.path.exists(model_labels):
                self.labels = read_labels(model_labels)

        self._update_mtime()
        return self.model, self.labels

    def _update_mtime(self):
        """Track file modification times."""
        weights_path = os.path.join(self.model_path, 'best_weights.index')
        if os.path.exists(weights_path):
            self._weights_mtime = os.path.getmtime(weights_path)

        active_config = os.path.join(
            os.path.dirname(self.model_path), 'active_model.json')
        if os.path.exists(active_config):
            self._active_model_mtime = os.path.getmtime(active_config)

    def check_reload(self):
        """Check if model files have been updated and reload if needed.

        Returns:
            True if model was reloaded, False otherwise.
        """
        # Check active_model.json for model path change
        active_config = os.path.join(
            os.path.dirname(self.model_path), 'active_model.json')
        if os.path.exists(active_config):
            mtime = os.path.getmtime(active_config)
            if mtime > self._active_model_mtime:
                try:
                    import json as _json
                    with open(active_config, 'r') as f:
                        config = _json.load(f)
                    new_path = config.get('model_path', self.model_path)
                    new_labels = config.get('labels_file', self.labels_file)
                    if new_path != self.model_path:
                        logger.info("Active model changed: %s -> %s",
                                    self.model_path, new_path)
                        self.model_path = new_path
                        self.labels_file = new_labels
                        self.load()
                        return True
                    self._active_model_mtime = mtime
                except Exception as e:
                    logger.warning("Failed to read active_model.json: %s", e)

        # Check weights file modification
        weights_path = os.path.join(self.model_path, 'best_weights.index')
        if os.path.exists(weights_path):
            mtime = os.path.getmtime(weights_path)
            if mtime > self._weights_mtime:
                logger.info("Model weights updated, reloading: %s",
                            self.model_path)
                try:
                    self.model.load_weights(
                        os.path.join(self.model_path, 'best_weights')
                    ).expect_partial()
                    self._weights_mtime = mtime
                    logger.info("Model reloaded successfully")
                    return True
                except Exception as e:
                    logger.error("Failed to reload model: %s", e)

        return False


logging.basicConfig(
    stream=sys.stdout,
    format="%(levelname)-8s %(asctime)-15s %(name)s %(message)s")
audio_recorder.logger.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Audio frame utilities
# ---------------------------------------------------------------------------

class Frame:
    """Represents a frame of audio data for VAD processing."""
    __slots__ = ('bytes', 'timestamp', 'duration')

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generate audio frames from PCM data.

    Args:
        frame_duration_ms: Frame length in milliseconds.
        audio: PCM audio bytes (int16).
        sample_rate: Audio sample rate in Hz.

    Yields:
        Frame objects of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def float2pcm(sig, dtype='int16'):
    """Convert float [-1, 1] audio to PCM integer format."""
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def read_labels(filename):
    """Read classification labels from a text file."""
    with open(filename, "r") as f:
        return [line.rstrip() for line in f.readlines()]


# ---------------------------------------------------------------------------
# VAD (Voice Activity Detection)
# ---------------------------------------------------------------------------

def vad_function(sample_rate, frame_duration_ms, vad_percent, vad, audio):
    """Run WebRTC VAD on audio and return True if speech is detected.

    Args:
        sample_rate: Audio sample rate in Hz.
        frame_duration_ms: Frame duration for VAD analysis.
        vad_percent: Minimum ratio of speech frames to trigger detection.
        vad: webrtcvad.Vad instance.
        audio: Float audio array.

    Returns:
        True if speech ratio exceeds vad_percent threshold.
    """
    audio_pcm = float2pcm(audio)
    frames = list(frame_generator(frame_duration_ms, audio_pcm, sample_rate))
    if not frames:
        return False
    vad_frame_num = sum(1 for f in frames if vad.is_speech(f.bytes, sample_rate))
    return (vad_frame_num / len(frames)) > vad_percent


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_kws_model(model_path):
    """Load KWS Transformer model from checkpoint.

    Args:
        model_path: Directory containing flags.json and best_weights.

    Returns:
        Loaded Keras model ready for inference.
    """
    flags_path = os.path.join(model_path, 'flags.json')
    if not os.path.exists(flags_path):
        raise FileNotFoundError(f"Model config not found: {flags_path}")

    with open(flags_path, 'r') as fd:
        flags_json = json.load(fd)

    class DictStruct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    flags = DictStruct(**flags_json)

    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf1.Session(config=config)
    tf1.keras.backend.set_session(sess)
    tf1.keras.backend.set_learning_phase(0)
    flags.batch_size = 1

    kws_model = models.MODELS[flags.model_name](flags)
    weights_path = os.path.join(model_path, 'best_weights')
    kws_model.load_weights(weights_path).expect_partial()

    logger.info("KWS model loaded from %s", model_path)
    return kws_model


def load_aec_model(model_path, model_type):
    """Load AEC (Acoustic Echo Cancellation) model.

    Args:
        model_path: Path to AEC model files.
        model_type: "tf" for SavedModel, "tflite" for TFLite.

    Returns:
        For tflite: (interpreters_list, states_buffer)
        For tf: infer function
    """
    if model_type == "tf":
        aec_model = tf.saved_model.load(model_path)
        aec_infer = aec_model.signatures["serving_default"]
        aec_infer._num_positional_args = 2
        logger.info("AEC model loaded (TF SavedModel)")
        return aec_infer, None

    if model_type == "tflite":
        path_1 = model_path + "_1.tflite"
        path_2 = model_path + "_2.tflite"
        if not os.path.exists(path_1):
            raise FileNotFoundError(f"AEC model not found: {path_1}")

        interpreter_1 = tflite.Interpreter(model_path=path_1)
        interpreter_1.allocate_tensors()
        interpreter_2 = tflite.Interpreter(model_path=path_2)
        interpreter_2.allocate_tensors()

        # Pre-allocate LSTM states
        input_details_1 = interpreter_1.get_input_details()
        input_details_2 = interpreter_2.get_input_details()
        states_1 = np.zeros(input_details_1[1]["shape"], dtype=np.float32)
        states_2 = np.zeros(input_details_2[1]["shape"], dtype=np.float32)

        logger.info("AEC model loaded (TFLite)")
        return [interpreter_1, interpreter_2], [states_1, states_2]

    raise ValueError(f"Unknown model_type: {model_type}")


# ---------------------------------------------------------------------------
# AEC inference
# ---------------------------------------------------------------------------

def inference_aec_tflite(infer, out_buffer, states_buffer, input_audio,
                         lpb_audio, len_audio, block_len=512, block_shift=128,
                         enhancer=None):
    """Run AEC inference using TFLite two-stage DTLN model.

    Now with integrated SOTA speech enhancement pipeline:
    - Pre-DTLN: pre-emphasis + IMCRA noise estimation + spectral subtraction
    - Post-DTLN: Wiener postfilter + perceptual weighting + de-emphasis

    Args:
        infer: List of [interpreter_1, interpreter_2].
        out_buffer: Overlap-add output buffer (block_len,).
        states_buffer: [states_1, states_2] LSTM state arrays.
        input_audio: Microphone audio with cache prefix.
        lpb_audio: Loopback/reference audio with cache prefix.
        len_audio: Number of new samples to process.
        block_len: FFT block length (must match model).
        block_shift: Block shift / hop size.
        enhancer: SpeechEnhancementPipeline instance (None = disabled).

    Returns:
        (predicted_speech, out_buffer, states_buffer)
    """
    out_buffer = np.squeeze(out_buffer)
    audio = np.squeeze(input_audio)
    lpb = np.squeeze(lpb_audio)

    audio = audio[-(len_audio + block_len - block_shift):]
    lpb = lpb[-(len_audio + block_len - block_shift):]

    interpreter_1, interpreter_2 = infer
    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()
    input_details_2 = interpreter_2.get_input_details()
    output_details_2 = interpreter_2.get_output_details()

    states_1, states_2 = states_buffer

    out_file = np.zeros(len(audio), dtype=np.float32)
    in_buffer = np.zeros(block_len, dtype=np.float32)
    in_buffer_lpb = np.zeros(block_len, dtype=np.float32)

    num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift

    for idx in range(num_blocks):
        # Shift and fill input buffer
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[idx * block_shift:(idx * block_shift) + block_shift]

        # Shift and fill loopback buffer
        in_buffer_lpb[:-block_shift] = in_buffer_lpb[block_shift:]
        in_buffer_lpb[-block_shift:] = lpb[idx * block_shift:(idx * block_shift) + block_shift]

        # === PRE-DTLN: SOTA speech enhancement ===
        if enhancer is not None:
            enhanced_buffer = enhancer.pre_enhance_block(in_buffer)
        else:
            enhanced_buffer = in_buffer

        # Stage 1: frequency domain masking
        in_block_fft = np.fft.rfft(enhanced_buffer).astype(np.complex64)
        in_mag = np.abs(in_block_fft).reshape(1, 1, -1).astype(np.float32)
        lpb_block_fft = np.fft.rfft(in_buffer_lpb).astype(np.complex64)
        lpb_mag = np.abs(lpb_block_fft).reshape(1, 1, -1).astype(np.float32)

        interpreter_1.set_tensor(input_details_1[0]["index"], in_mag)
        interpreter_1.set_tensor(input_details_1[2]["index"], lpb_mag)
        interpreter_1.set_tensor(input_details_1[1]["index"], states_1)
        interpreter_1.invoke()

        out_mask = interpreter_1.get_tensor(output_details_1[0]["index"])
        states_1 = interpreter_1.get_tensor(output_details_1[1]["index"])

        # Apply mask and convert back to time domain
        estimated_block = np.fft.irfft(in_block_fft * out_mask)

        # Stage 2: time domain refinement
        estimated_block = estimated_block.reshape(1, 1, -1).astype(np.float32)
        in_lpb = in_buffer_lpb.reshape(1, 1, -1).astype(np.float32)

        interpreter_2.set_tensor(input_details_2[1]["index"], states_2)
        interpreter_2.set_tensor(input_details_2[0]["index"], estimated_block)
        interpreter_2.set_tensor(input_details_2[2]["index"], in_lpb)
        interpreter_2.invoke()

        out_block = interpreter_2.get_tensor(output_details_2[0]["index"])
        states_2 = interpreter_2.get_tensor(output_details_2[1]["index"])

        # === POST-DTLN: Wiener postfilter + perceptual weighting ===
        dtln_out = np.squeeze(out_block)
        if enhancer is not None:
            dtln_out = enhancer.post_enhance_block(dtln_out)

        # Overlap-add output
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = 0.0
        out_buffer += dtln_out
        out_file[idx * block_shift:(idx * block_shift) + block_shift] = out_buffer[:block_shift]

    # Trim to original length and prevent clipping
    predicted_speech = out_file[:len_audio]
    max_val = np.max(np.abs(predicted_speech))
    if max_val > 1.0:
        predicted_speech = predicted_speech / max_val * 0.99

    return predicted_speech.reshape(-1, 1), out_buffer, [states_1, states_2]


def inference_aec_tf(infer, out_buffer, input_audio, lpb_audio, len_audio,
                     block_len=512, block_shift=128):
    """Run AEC inference using TF SavedModel (slower, for reference)."""
    out_buffer = np.squeeze(out_buffer)
    audio = np.squeeze(input_audio)
    lpb = np.squeeze(lpb_audio)

    audio = audio[-(len_audio + block_len - block_shift):]
    lpb = lpb[-(len_audio + block_len - block_shift):]

    out_file = np.zeros(len(audio), dtype=np.float32)
    in_buffer = np.zeros(block_len, dtype=np.float32)
    in_buffer_lpb = np.zeros(block_len, dtype=np.float32)

    num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift

    for idx in range(num_blocks):
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[idx * block_shift:(idx * block_shift) + block_shift]

        in_buffer_lpb[:-block_shift] = in_buffer_lpb[block_shift:]
        in_buffer_lpb[-block_shift:] = lpb[idx * block_shift:(idx * block_shift) + block_shift]

        in_block = np.expand_dims(in_buffer, axis=0).astype(np.float32)
        in_block_lpb = np.expand_dims(in_buffer_lpb, axis=0).astype(np.float32)

        out_block = infer(
            time_data_x=tf.constant(in_block_lpb),
            time_data_y=tf.constant(in_block))['conv1d_2']

        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = 0.0
        out_buffer += np.squeeze(out_block)
        out_file[idx * block_shift:(idx * block_shift) + block_shift] = out_buffer[:block_shift]

    predicted_speech = out_file[:len_audio]
    max_val = np.max(np.abs(predicted_speech))
    if max_val > 1.0:
        predicted_speech = predicted_speech / max_val * 0.99

    return predicted_speech.reshape(-1, 1), out_buffer


# ---------------------------------------------------------------------------
# KWS inference
# ---------------------------------------------------------------------------

def inference_kws(kws_model, input_audio, audio_sample_length):
    """Run KWS inference on audio clip.

    Args:
        kws_model: Loaded Keras KWS model.
        input_audio: Audio array of shape (N, 1).
        audio_sample_length: Expected sample count (e.g., 24000 for 1.5s@16kHz).

    Returns:
        Prediction array of shape (1, num_classes).
    """
    audio_sample_length = int(audio_sample_length)
    if audio_sample_length > len(input_audio):
        padded = np.pad(
            input_audio,
            ((0, audio_sample_length - len(input_audio)), (0, 0)),
            'constant')
    else:
        padded = input_audio[:audio_sample_length]
    return kws_model.predict(padded.T.astype(np.float32))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def add_model_flags(parser):
    """Add model-related CLI arguments to an argparse.ArgumentParser."""
    parser.add_argument(
        "--kws_model_path",
        default="models_data/KWS/HKSC_v0.01/kwt3_softmax",
        help="Path to KWS model directory.")
    parser.add_argument(
        "--aec_model_path",
        default="models_data/AEC/tflite_model/dtln_aec_128",
        help="Path to AEC model (without _1.tflite/_2.tflite suffix).")
    parser.add_argument(
        "--labels_file",
        default="config/labels_HKSC_12.txt",
        help="Path to labels text file.")
    parser.add_argument(
        "--mic", type=int, default=0,
        help="Input microphone device ID (0=auto-detect).")
    parser.add_argument(
        "--vad", type=int, default=0, choices=[0, 1],
        help="Enable VAD preprocessing (0=off, 1=on).")
    parser.add_argument(
        "--aec", type=int, default=1, choices=[0, 1],
        help="Enable AEC preprocessing (0=off, 1=on).")
    parser.add_argument(
        "--kws", type=int, default=1, choices=[0, 1],
        help="Enable KWS inference (0=off, 1=on).")
    parser.add_argument(
        "--kws_threshold", type=float, default=0.98,
        help="KWS softmax confidence threshold.")
    parser.add_argument(
        "--vad_percent", type=float, default=0.2,
        help="Minimum speech ratio for VAD activation.")
    parser.add_argument(
        "--vad_level", type=int, default=3, choices=[0, 1, 2, 3],
        help="WebRTC VAD aggressiveness (0=least, 3=most).")
    parser.add_argument(
        "--write_aec_result", type=int, default=0, choices=[0, 1],
        help="Write AEC debug wav files and exit.")
    parser.add_argument(
        "--aec_output_filename", default="aec_output.wav",
        help="Output filename for AEC debug wav.")
    parser.add_argument(
        "--sample_rate_hz", type=int, default=16000,
        help="Audio sample rate (only 16000 is supported).")
    parser.add_argument(
        "--aec_model_type", default="tflite", choices=["tf", "tflite"],
        help="AEC model format.")
    parser.add_argument(
        "--num_channels", type=int, default=6,
        help="Number of audio input channels.")
    parser.add_argument(
        "--loopback_channel", type=int, default=5,
        help="Channel index for AEC loopback reference.")
    parser.add_argument(
        "--clip_duration", type=float, default=1.5,
        help="KWS clip duration in seconds.")
    parser.add_argument(
        "--shift_duration", type=float, default=0.25,
        help="Sliding window shift in seconds.")
    # --- SOTA Speech Enhancement options ---
    parser.add_argument(
        "--enhance", type=int, default=1, choices=[0, 1],
        help="Enable SOTA speech enhancement pipeline (0=off, 1=on).")
    parser.add_argument(
        "--enhance_aggressive", type=float, default=0.5,
        help="Enhancement aggressiveness 0.0(mild)~1.0(extreme). "
             "Higher = more noise removal but risk of speech distortion.")
    parser.add_argument(
        "--enhance_spectral_sub", type=int, default=1, choices=[0, 1],
        help="Enable IMCRA + spectral subtraction pre-cleaner.")
    parser.add_argument(
        "--enhance_wiener", type=int, default=1, choices=[0, 1],
        help="Enable Wiener postfilter for residual noise.")
    parser.add_argument(
        "--enhance_perceptual", type=int, default=1, choices=[0, 1],
        help="Enable Mel-band perceptual weighting for KWS.")
    # --- Model hot-reload ---
    parser.add_argument(
        "--model_watch", type=int, default=0, choices=[0, 1],
        help="Watch for model updates and hot-reload (0=off, 1=on). "
             "Checks every inference loop for new best_weights or "
             "active_model.json changes.")


# ---------------------------------------------------------------------------
# Main classification loop
# ---------------------------------------------------------------------------

def classify_audio(audio_device_index,
                   kws_model_path,
                   aec_model_path,
                   labels_file,
                   result_callback=None,
                   vad=0,
                   aec=1,
                   kws=1,
                   kws_threshold=0.98,
                   vad_percent=0.2,
                   vad_level=3,
                   write_aec_result=0,
                   aec_output_filename="aec_output.wav",
                   sample_rate_hz=16000,
                   aec_model_type="tflite",
                   num_channels=6,
                   loopback_channel=5,
                   clip_duration=1.5,
                   shift_duration=0.25,
                   enhance=1,
                   enhance_aggressive=0.5,
                   enhance_spectral_sub=1,
                   enhance_wiener=1,
                   enhance_perceptual=1,
                   model_watch=0):
    """Main real-time audio classification pipeline.

    Captures audio from microphone, applies optional VAD, SOTA speech
    enhancement, AEC, and KWS inference in a continuous loop.

    If model_watch=1, periodically checks for model updates and
    hot-reloads the KWS model without restarting.
    """
    if sample_rate_hz != 16000:
        raise ValueError("Only 16kHz sample rate is supported by the models.")

    AUDIO_SAMPLE_RATE_HZ = sample_rate_hz
    len_shift_frame = math.trunc(
        (shift_duration * AUDIO_SAMPLE_RATE_HZ) / 128) * 128
    downsample_factor = 1
    frame_duration_ms = 30

    tf1.disable_eager_execution()

    # --- Load models ---
    vad_model = None
    vad_result = True
    if vad:
        vad_model = webrtcvad.Vad(int(vad_level))

    kws_model = None
    model_mgr = None
    if kws:
        if model_watch:
            model_mgr = ModelManager(kws_model_path, labels_file)
            kws_model, mgr_labels = model_mgr.load()
            logger.info("Model watch enabled — will auto-reload on updates")
        else:
            kws_model = load_kws_model(kws_model_path)
        # Warm-up inference to avoid first-call latency
        empty_audio = np.zeros((int(clip_duration * AUDIO_SAMPLE_RATE_HZ), 1))
        _ = inference_kws(kws_model, empty_audio,
                          clip_duration * AUDIO_SAMPLE_RATE_HZ)
        logger.info("KWS model initialized (warm-up done)")

    aec_infer = None
    states_buffer = None
    if aec:
        aec_infer, states_buffer = load_aec_model(aec_model_path, aec_model_type)
        logger.info("AEC model initialized")

    cache_aec_buffer = np.zeros((512, 1), dtype=np.float32)
    cache_aec_audio = np.zeros(
        (int((clip_duration - shift_duration) * AUDIO_SAMPLE_RATE_HZ), 1),
        dtype=np.float32)

    # --- Initialize SOTA speech enhancement ---
    enhancer = None
    if enhance:
        enhancer = SpeechEnhancementPipeline(
            block_len=512,
            block_shift=128,
            sr=AUDIO_SAMPLE_RATE_HZ,
            spectral_sub_enabled=bool(enhance_spectral_sub),
            wiener_post_enabled=bool(enhance_wiener),
            perceptual_weight_enabled=bool(enhance_perceptual),
            aggressive=enhance_aggressive)
        logger.info("SOTA speech enhancement initialized (aggressive=%.2f)",
                    enhance_aggressive)

    labels = read_labels(labels_file)
    logger.info("Labels loaded: %s", labels)

    # --- Initialize audio recorder ---
    recorder = audio_recorder.AudioRecorder(
        AUDIO_SAMPLE_RATE_HZ,
        downsample_factor=downsample_factor,
        device_index=audio_device_index,
        num_channels=num_channels)

    results = []
    loop_num = 0
    sum_total_time = 0.0

    with recorder:
        # Pre-fill buffer with initial audio
        start_raw_audios = recorder.get_audio(
            (clip_duration - shift_duration) * AUDIO_SAMPLE_RATE_HZ)[0]
        cache_audios = start_raw_audios
        cache_aec_audio = start_raw_audios[:, 0].reshape(-1, 1)

        logger.info("Recording started - listening for keywords...")

        try:
            while True:
                shift_audios = recorder.get_audio(len_shift_frame)[0]
                raw_audios = np.concatenate(
                    (cache_audios, shift_audios), axis=0)
                start_time = time.time()

                input_audio = raw_audios[:, 0].reshape(-1, 1)

                # --- VAD on raw audio ---
                if vad:
                    vad_result = vad_function(
                        AUDIO_SAMPLE_RATE_HZ, frame_duration_ms,
                        vad_percent, vad_model, input_audio)

                # --- AEC when VAD active ---
                if aec and vad_result:
                    lpb_audio = raw_audios[:, loopback_channel].reshape(-1, 1)
                    if aec_model_type == "tflite":
                        shift_aec_audio, cache_aec_buffer, states_buffer = \
                            inference_aec_tflite(
                                aec_infer, cache_aec_buffer, states_buffer,
                                input_audio, lpb_audio, len_shift_frame,
                                enhancer=enhancer)
                    else:
                        shift_aec_audio, cache_aec_buffer = \
                            inference_aec_tf(
                                aec_infer, cache_aec_buffer,
                                input_audio, lpb_audio, len_shift_frame)
                    input_audio = np.concatenate(
                        (cache_aec_audio, shift_aec_audio), axis=0)
                    cache_aec_audio = input_audio[int(len_shift_frame):]

                cache_audios = raw_audios[int(len_shift_frame):]

                # --- VAD on AEC-processed audio ---
                if aec and vad:
                    vad_result = vad_function(
                        AUDIO_SAMPLE_RATE_HZ, frame_duration_ms,
                        vad_percent, vad_model, input_audio)

                # --- Debug: write AEC result and stop ---
                if vad_result and aec and write_aec_result:
                    sf.write(aec_output_filename, input_audio,
                             AUDIO_SAMPLE_RATE_HZ)
                    sf.write("input_audio_" + aec_output_filename,
                             raw_audios[:, 0].reshape(-1, 1),
                             AUDIO_SAMPLE_RATE_HZ)
                    sf.write("lpb_audio_" + aec_output_filename,
                             lpb_audio, AUDIO_SAMPLE_RATE_HZ)
                    logger.info("AEC debug files written. Stopping.")
                    break

                # --- Hot-reload: check for model updates ---
                if model_watch and model_mgr and loop_num % 10 == 0:
                    if model_mgr.check_reload():
                        kws_model = model_mgr.model
                        if model_mgr.labels:
                            labels = model_mgr.labels
                        logger.info("KWS model hot-reloaded, labels: %s",
                                    labels)

                # --- KWS inference ---
                if kws and vad_result:
                    result = inference_kws(
                        kws_model, input_audio,
                        clip_duration * AUDIO_SAMPLE_RATE_HZ)

                    if result_callback.__name__ == "print_results_v2":
                        results.append(result[0])
                        if len(results) == 3:
                            result_callback(results, labels, kws_threshold)
                            results = []
                    else:
                        result_callback(result[0], labels, kws_threshold)
                else:
                    print("__listening__")
                    if not aec:
                        time.sleep(0.05)

                total_time = time.time() - start_time
                loop_num += 1
                sum_total_time += total_time
                logger.debug(
                    "Loop %d: %.3fs (avg: %.3fs)",
                    loop_num, total_time, sum_total_time / loop_num)

        except KeyboardInterrupt:
            logger.info("Stopped by user (Ctrl+C)")
        except audio_recorder.TimeoutError:
            logger.error("Audio capture timed out")
