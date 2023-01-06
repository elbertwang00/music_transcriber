import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle
import os, sys
from scipy.fft import rfft, rfftfreq
from scipy.signal import stft
import ffmpeg
import pandas as pd

audioPath = "Music_Audio"
videoPath = "Music_Video"
PCMOutputFile = "PCMOutput.raw"
noteFreqsFile = "Note_Frequencies.csv"

inputSampleRate = 16000 # Hz
FFTWindowTime = 0.05 # s
FFTWindowSamplingRatio = 10 # Number of audio samples between each sample of FFT
maxFreq = 2000 # Limit for musical frequencies to truncate FFTs, rounded to thousands

displayUpdateRate = 0.1 # Rate at which displayed graphs are updated
peakDetectionThres = 30 # Number of std deviations away from FFT mean to find peaks

noteFreqs = {} # Dictionary to map notes to frequencies


class Track:
    def __init__(self):
        self.title = None

        self.audio = None
        self.time = None

        self.FFTTime = None
        self.FFTFreq = None
        self.FFTAmp = None

        self.FFTSliceMeans = None
        self.FFTSliceStds = None
        self.FFTAmpMean = None
        self.FFTAmpStd = None

    def LoadTrack(self):
        fileType = self.title[-4:]

        if fileType == ".mp4": # Conversion to PCM for mp4 files
            self.audio = mp4_2_PCM(self.title)
            self.time = np.arange(0, len(self.audio)/inputSampleRate, 1/inputSampleRate)
        
        PrintFilledLine(" FINISHED LOADING TRACK AUDIO ")

        self.FFTTime, self.FFTFreq, self.FFTAmp, self.FFTSliceMeans, self.FFTSliceStds = PCM_2_STFT(self.audio)

        self.FFTAmpMean = self.FFTSliceMeans.mean()
        self.FFTAmpStd = self.FFTSliceStds.mean()

        PrintFilledLine(" FINISHED LOADING TRACK AUDIO FFT ")

        return
    
    def FindPeaks(self):
        # Function to find where "notes" are detected, based on the variable peakDetectionThres

        global peakDetectionThres

        peakTimes = []
        peakFreqs = []

        peakThres = self.FFTAmpMean + peakDetectionThres * self.FFTAmpStd

        for i, FFTSlice in enumerate(self.FFTAmp):
            for j, Amp in enumerate(FFTSlice):
                if Amp > peakThres:
                    peakTimes.append(self.FFTTime[i])
                    peakFreqs.append(self.FFTFreq[j])

        return peakTimes, peakFreqs

def initNoteFreqs():
    # Pulls information .csv containing frequencies for different notes

    df = pd.read_csv(noteFreqsFile)
    notes = df["Note"]
    freqs = df["Frequency"]

    for i in range(len(notes)):
        noteFreqs[notes[i]] = freqs[i]

def mp4_2_PCM(title):
    # Function to extract PCM audio data from mp4 files

    video = ffmpeg.input(os.path.join(videoPath, title))
    out, _ = (video.output('PCMOutput.raw', format='s16le', acodec='pcm_s16le', ac=1, ar=str(inputSampleRate))
            .overwrite_output()
            .run(capture_stdout = False))

    data = np.memmap(PCMOutputFile, dtype='h', mode='r')

    return np.array(data)

def PCM_2_STFT(audio):
    # Function to find the Short Time Fourier Transform of audio data by performing FFTs on discrete time windows

    FFTWindowWidth = int(FFTWindowTime * inputSampleRate)

    audioOrigLen = len(audio)

    audio = np.concatenate((audio, np.zeros(FFTWindowWidth - 1)))
    time = np.arange(0, len(audio)/inputSampleRate, 1/inputSampleRate)

    windowFFTs = []
    freqFFT = rfftfreq(FFTWindowWidth, 1/inputSampleRate)
    truncateIndex = len(freqFFT)

    for i in range(len(freqFFT)):
        if round(freqFFT[i], -3) == maxFreq:
            truncateIndex = i
    freqFFT = freqFFT[:truncateIndex]
    
    sampledTime, windowFFTs, sliceMeans, sliceStds = [], [], [], []

    for i in range(audioOrigLen):
        if i%FFTWindowSamplingRatio == 0:
            audioWindow = audio[i : i + FFTWindowWidth]

            audioWindow -= audioWindow.mean()

            windowFFT = np.real(rfft(audioWindow))
            windowFFT = windowFFT[:truncateIndex]
            windowFFT = np.array([max([0, x]) for x in windowFFT])

            sampledTime.append(time[i])
            windowFFTs.append(windowFFT)
            sliceMeans.append(windowFFT.mean())
            sliceStds.append(windowFFT.std())

    sampledTime = np.array(sampledTime)
    windowFFTs = np.array(windowFFTs)
    sliceMeans = np.array(sliceMeans)
    sliceStds = np.array(sliceStds)

    return sampledTime, freqFFT, windowFFTs, sliceMeans, sliceStds

def PrintFilledLine(text):
    # Prints a line filled on each side with symbols up to a certain length

    print(f"{text:=^200}")
    return

def ChooseTrack():
    # Sets up the track - whatever file type to be processed by the software

    global studiedTrack
    availableTracks = []

    for root, dirs, files in os.walk("."):
        for file in files:
            if file[-4:] in [".wav", ".mp4"]:
                availableTracks.append(file)

    print("Tracks available for transcription:")
    for i, track in enumerate(availableTracks):
        print(f"{i+1}) {track}")
    
    try:
        choice = 2 # int(input("\nEnter index of track to transcribe from above:")) - 1
    except:
        print("Invalid index entered")
        choice = 0

    studiedTrack.title = availableTracks[choice]
    print(f"Track chosen: {studiedTrack.title}")
    PrintFilledLine(" PROCESSING TRACK AUDIO NOW ")
    return

def PlotMusic(track):
    # Displaying function to show interactive plot with results
    
    peakTimes, peakFreqs = track.FindPeaks()

    print(len(track.FFTTime), len(track.FFTFreq))

    # Initiating plot display layout
    fig = plt.figure(figsize=(20, 10))
    axFFT = plt.axes([0.05, 0.15, 0.75, 0.8])
    axColorbar = plt.axes([0.85, 0.15, 0.02, 0.8])
    axThresSlider = plt.axes([0.15, 0.05, 0.8, 0.05])

    thresSlider = Slider(axThresSlider, "StdDev Multiplication\n Factor for \nThresholding Notes",
                        0, 50, valinit=0, valstep=0.1)

    thresSlider.set_val(peakDetectionThres)

    # Plot FFT over time graph
    plt.axes(axFFT)
    plt.xlabel("Time / s")
    plt.ylabel("Frequency / Hz")

    axFFT.set_xticks(np.arange(0, max(track.FFTTime), 0.25))
    axFFT.set_yticks(np.arange(0, max(track.FFTFreq), 100))
    axFFT.set_xlim(0, max(track.FFTTime))
    axFFT.set_ylim(0, maxFreq)

    FFTAmpTrans = np.transpose(track.FFTAmp)
    plt.contourf(track.FFTTime, track.FFTFreq, FFTAmpTrans, 100, cmap="inferno")
    plt.colorbar(cax=axColorbar)

    # Plot scatter of peaks detected
    peakScatter = axFFT.scatter(peakTimes, peakFreqs, color="g", s=0.5)

    # Plot notes and respective frequencies
    for i, note in enumerate(noteFreqs.keys()):
        plt.hlines(noteFreqs[note], 0, max(track.FFTTime), color="g", linewidth=0.3)
        axFFT.text(0.05, noteFreqs[note], note, transform=axFFT.get_yaxis_transform(), color="g", size=8)

    def ThresSliderChanged(slider):
        # Function for updating scatter plot of notes found when slider changed
        global peakDetectionThres

        peakDetectionThres = thresSlider.val
        peakTimes, peakFreqs = track.FindPeaks()
        peakScatter.set_offsets(np.c_[peakTimes, peakFreqs])

        fig.canvas.draw_idle()

        return

    thresSlider.on_changed(ThresSliderChanged)

    plt.show()
    return


initNoteFreqs()
studiedTrack = Track()
ChooseTrack()
studiedTrack.LoadTrack()
PlotMusic(studiedTrack)
