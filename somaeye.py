##############################
#
# Written at 3am, don't judge.
#
##############################

import math
from threading import Lock
import numpy as np  # 1.22.3
from scipy.ndimage.filters import gaussian_filter # 1.7.3
import samplerate # https://github.com/tuxu/python-samplerate
import cv2  # 4.5.4.60

# for installing pyaudio on ubuntu
#   sudo apt-get install portaudio19-dev
#   pip3 install pyaudio
import pyaudio # 0.2.11


def fit_signal(signal, target):
    """Takes a 0-255 signal of arbitrary length and formats it
    to about +/- 0.5 and resamples it to the target length.
    """
    signal = signal / 255
    signal = signal - np.median(signal)
    ratio = target / len(signal)
    signal = samplerate.resample(signal, ratio, "sinc_best")
    return signal


def generate_window(num_samples):
    """ windowing function to reduce discontinuities at wave edges (for looping)"""
    return np.sin(np.pi*np.arange(num_samples)/num_samples).astype(np.float32)


def blend_waves(wave1, wave2, amount):
    """Blend two waves"""
    return (wave1 * amount) + (wave2 * (1 - amount))


def sive_wave(samples_per_cycle):
    """Make me sine wave"""
    return (np.sin(2*np.pi*np.arange(samples_per_cycle)/samples_per_cycle)).astype(np.float32)


def update_wave(wave, image, phase, blend, spread, scan):
    """Takes in an image and returns an audio wave."""

    num_samples = len(wave) # wave already in buffer
    center = image.shape[0]//2 # center row index
    scan = int(scan) # just making sure

    # construct signal
    pixel_line = np.mean(image[((center + scan) - spread):(center + scan) + spread, :, 1], axis=0)
    signal = pixel_line.astype(np.float32)
    signal = fit_signal(signal, num_samples)

    # Unused test signals for debugging
    # signal = np.zeros(signal.shape)
    # signal = np.random.normal(0, 1, signal.shape)
    # signal = sive_wave(num_samples)
    # signal = gaussian_filter(signal, sigma=1)

    signal = signal * generate_window(num_samples)

    # align phase of new wave with that of current wave
    signal = np.roll(signal, -phase)

    # blend new wave with current wave
    signal = blend_waves(signal, wave, blend)

    return signal


def main():
    """Main entry point"""

    # default camera is 0, my microscope is 2
    cap = cv2.VideoCapture(0)
    pa = pyaudio.PyAudio()
    mutex = Lock() # PyAudio is threaded
    
    # Lowest possible frequency in Hz
    frequency_hz =  30 

    # Audio playback ample rate in Hz
    sample_rate_hz = 44100

    # percentage of new wave to blend into existing wave, 0-1
    wave_blending = 0.2

    # number of samples in one wave cycle
    samples_per_cycle = int(sample_rate_hz / frequency_hz)

    global phase, wave, lfo, rate
    phase = 0   # track phase
    lfo = 0.0   # track lfo value
    rate = 0.002 # adjustable LFO rate.  Keep in mind you blend rate too.
    wave = sive_wave(samples_per_cycle) #init wave kernel bufer


    def run_lfo():
        global lfo, rate
        lfo = (lfo + rate) % 1.0


    def callback(in_data, frame_count, time_info, status):
        """Reads frame_count samples from buffer"""
        global wave, phase
        mutex.acquire()

        num_buffer_samples = len(wave)

        # roll wave to next n frames requested by pyaudio
        wave = np.roll(wave, -frame_count)

        # loop wave n-times to fill requested frames
        temp = np.tile(wave, math.ceil(frame_count/num_buffer_samples))

        # update phase tracking
        phase = (phase + frame_count) % num_buffer_samples

        # update the LFO
        run_lfo()

        mutex.release()

        return (temp, pyaudio.paContinue)

    # init the audio stream, will also start playing whatever is in `wave`
    stream = pa.open(format=pyaudio.paFloat32,
                     channels=1,
                     rate=sample_rate_hz,
                     output=True,
                     stream_callback=callback)

    # test image if you don't want to wait for the camera
    # frame = cv2.imread("test.jpg", cv2.IMREAD_COLOR)

    while True:
        # read webcam
        _, frame = cap.read()

        # scale LFO value for wave scanning
        scan = np.sin(2 * np.pi * lfo) * 100

        # update the wave kernel
        mutex.acquire()
        wave = update_wave(wave, frame, phase, 0.2, 1, scan)
        mutex.release()

        # show the camera frame
        cv2.imshow("Webcam", frame)
    
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27: #escape
            break

    # stop stuff
    cap.release()
    cv2.destroyAllWindows()
    stream.stop_stream()
    stream.close()
    pa.terminate()


if __name__ == "__main__":
    main()