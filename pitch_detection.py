# -*- coding: utf-8 -*-
"""Pitch Detection

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NavCvbeTgx-Z9hGPWkuWfD3EwnyGh9M7
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import librosa
import time
import logging
import math
import statistics
import sys
import soundfile as sf
from scipy.io import wavfile
from pydub import AudioSegment

# Suppress tf's warining
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

logging.getLogger('tensorflow').disabled = True
# Loading the SPICE model is easy:
model = hub.load("https://tfhub.dev/google/spice/2")
filename = '00001.wav'
# Function that converts the user-created audio to the format that the model
# expects: bitrate 16kHz and only one channel (mono).
EXPECTED_SAMPLE_RATE = 16000

A4 = 440

note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def convert_audio_for_model(user_file, output_file='converted_audio_file.wav'):
    audio = AudioSegment.from_file(user_file)
    audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)
    audio.export(output_file, format="wav")
    return output_file

def output2hz(pitch_output):
    # Constants taken from https://tfhub.dev/google/spice/2
    PT_OFFSET = 25.58
    PT_SLOPE = 63.07
    FMIN = 10.0;
    BINS_PER_OCTAVE = 12.0;
    cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET;
    return FMIN * 2.0 ** (1.0 * cqt_bin / BINS_PER_OCTAVE)

def hz2offset(freq):
    # This measures the quantization error for a single note.
    if freq == 0:  # Rests always have zero error.
        return None
    # Quantized note.
    h = round(12 * math.log2(freq / A4))
    return 12 * math.log2(freq / A4) - h

def quantize_predictions(group, ideal_offset):
    # Group values are either 0, or a pitch in Hz.
    non_zero_values = [v for v in group if v != 0]
    zero_values_count = len(group) - len(non_zero_values)

    # Create a rest if 80% is silent, otherwise create a note.
    if zero_values_count > 0.8 * len(group):
        # Interpret as a rest. Count each dropped note as an error, weighted a bit
        # worse than a badly sung note (which would 'cost' 0.5).
        return 0.51 * len(non_zero_values), "Rest", 0
    else:

        # Interpret as note, estimating as mean of non-rest predictions.
        h = round(
            statistics.mean([
                12 * math.log2(freq / A4) - ideal_offset for freq in non_zero_values
            ])) + 69

        octave = (h - 12) // 12
        n = (h - 12) % 12
        note = note_names[n] + str(octave)
        # Quantization error is the total difference from the quantized note.
        error = sum([
            abs(12 * math.log2(freq / A4) - ideal_offset - (h - 69))
            for freq in non_zero_values
        ])
        return error, note, h


def get_quantization_and_error(pitch_outputs_and_rests, predictions_per_eighth,
                               prediction_start_offset, ideal_offset):
  # Apply the start offset - we can just add the offset as rests.
    pitch_outputs_and_rests = [0] * prediction_start_offset + \
                              pitch_outputs_and_rests
    # Collect the predictions for each note (or rest).
    groups = [
        pitch_outputs_and_rests[i:i + predictions_per_eighth]
        for i in range(0, len(pitch_outputs_and_rests), predictions_per_eighth)
    ]
    quantization_error = 0

    notes_and_rests = []
    p_vals = []
    for group in groups:
        error, note_or_rest, p = quantize_predictions(group, ideal_offset)
        quantization_error += error
        notes_and_rests.append(note_or_rest)
        p_vals.append(p)

    return quantization_error, notes_and_rests, p_vals

def get_notes(filename):
    converted_audio_file = convert_audio_for_model(filename)

    # Loading audio samples from the wav file:
    sample_rate, audio_samples = wavfile.read(converted_audio_file, 'rb')
    origin_audio_samples, origin_sample_rate = librosa.load(filename)
    # Show some basic information about the audio.
    duration = len(audio_samples)/sample_rate
    # print(f'Sample rate: {sample_rate} Hz')
    # print(f'Total duration: {duration:.2f}s')
    # print(f'Size of the input: {len(audio_samples)}')

    # The audio samples are in int16 format. They need to be normalized to floats between -1 and 1.
    MAX_ABS_INT16 = 32768.0
    audio_samples = audio_samples / float(MAX_ABS_INT16)

    # We now feed the audio to the SPICE tf.hub model to obtain pitch and uncertainty outputs as tensors.
    #model_output = model.signatures["serving_default"](tf.constant(audio_samples, tf.float32))
    onset_frames = librosa.onset.onset_detect(origin_audio_samples, sr=origin_sample_rate, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1, hop_length=512)
    onset_times = librosa.frames_to_time(onset_frames, hop_length=512)
    onset_ticks = onset_times * origin_sample_rate
    onset_ticks = np.array([int(t) for t in onset_ticks])
    onset_in_seconds = onset_ticks / origin_sample_rate
    onset_in_frames = (onset_in_seconds * 1000 + 16) // 32


    out_onset = []
    out_note = []
    model_onset_ticks = onset_in_seconds * EXPECTED_SAMPLE_RATE
    model_onset_ticks = np.array([int(t) for t in model_onset_ticks])
    model_onset_ticks = np.append(model_onset_ticks, len(audio_samples))
    for i in range(1, len(onset_ticks)):
        #pitches, magnitudes = librosa.core.piptrack(y=y, sr=origin_audio_samples[onset_ticks[i-1]:onset_ticks[i]], fmin=75, fmax=1600)
        model_output = model.signatures["serving_default"](tf.constant(audio_samples[model_onset_ticks[i-1]:model_onset_ticks[i]], tf.float32))
        pitch_outputs = model_output["pitch"]
        uncertainty_outputs = model_output["uncertainty"]
        indices = range(len (pitch_outputs))
        confidence_outputs = 1.0 - uncertainty_outputs
        pitch_outputs_and_rests = [
            output2hz(p) if c >= 0.9 else 0
            for i, p, c in zip(indices, pitch_outputs, confidence_outputs)
        ]
        #max_confidecne_hz = output2hz(pitch_outputs[np.argmax(confidence_outputs)])
        #max_confidence = np.max(confidence_outputs)
        # print(pitch_outputs_and_rests, max_confidecne_hz, max_confidence)
        #if max_confidence > 0.8:
        #    out_onset.append(onset_ticks[i-1])
        offsets = [hz2offset(p) for p in pitch_outputs_and_rests if p != 0]
        if offsets:
            ideal_offset = statistics.mean(offsets)
        else:
            continue

        error, note, p = quantize_predictions(pitch_outputs_and_rests, ideal_offset)
        out_onset.append(onset_in_seconds[i-1])
        out_note.append(p)
    return out_note, out_onset
    """
    exit()

    pitch_outputs = model_output["pitch"]
    uncertainty_outputs = model_output["uncertainty"]

    # 'Uncertainty' basically means the inverse of confidence.
    confidence_outputs = 1.0 - uncertainty_outputs

    confidence_outputs = list(confidence_outputs)
    pitch_outputs = [ float(x) for x in pitch_outputs]

    indices = range(len (pitch_outputs))
    onset_in_frames = np.append(onset_in_frames, len(pitch_outputs))
    onset_in_frames = onset_in_frames.astype("int32")

    pitch_outputs_and_rests = [
        output2hz(p) if c >= 0.9 else 0
        for i, p, c in zip(indices, pitch_outputs, confidence_outputs)
    ]
    # The ideal offset is the mean quantization error for all the notes
    # (excluding rests):
    offsets = [hz2offset(p) for p in pitch_outputs_and_rests if p != 0]
    ideal_offset = statistics.mean(offsets)

    pitch_based_on_onset = []
    for start, end in zip(onset_in_frames[:-1], onset_in_frames[1:]):
        #p = pitch_outputs[start:end][np.argmax(confidence_outputs[start:end])]
        #pitch_based_on_onset.append(output2hz(p))
        pitch_based_on_onset.append(pitch_outputs_and_rests[start:end])

    # print(pitch_based_on_onset)
    ps = []
    for p in pitch_based_on_onset:
        if not p:
            ps.append(0)
        else:
            error, note, p = quantize_predictions(p, ideal_offset)
            ps.append(p)
    ps = np.array(ps)
    # print(ps[ps!=0])
    # print(onset_in_seconds[ps!=0])
    return ps[ps!=0], onset_in_seconds[ps!=0]
    """
if __name__ == "__main__":
    t = time.time()
    print(get_notes(filename))
    print('Time: ' + str((time.time() - t)) + ' s')

