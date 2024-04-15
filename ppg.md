# PPG at a distance

## Intro
<div align="justify">
Photoplethysmography (PPG) is a measurement technique used to observe volumetric changes in peripheral blood vessels. Usual measurement setups operate by having LEDs in close proximity to skin (finger) and measuring either the transmitted or reflected light.
</div>
<br>
I've heard the same measurement is possible at a distance using a regular phone camera. The goal of this project is to verify this claim.

## Strategy

1. **Gather data**
    1. Stationary body part (back of hand, wrist, face) with natural light
    2. Stationary body part with artificial light

2. **Preprocessing**

     Stabilize video (optional), extract metadata (lenght, fps ...) and crop frames of video 

4. **Analysis**

   Look at difference between adjacent frames, apply filters, search for local maxima, examine periodicity



## Gathering data

<div align="justify">
    Since the final result of such a measurement represents the number of heartbeats over a minute (Beats Per Minute, BPM) it seemed beneficial to start with approximately 1 minute long videos. This also minimises the error coming from extrapolating results from a short video. 


I recorded my hand, wrist and face for around 1 minute each in natural and artificial light (non-flickering LEDs to avoid 60 Hz artefact). I tried to minimise movement by fixing the phone and/or bodypart’s position. I used the main camera of a Samsung Galaxy S8 (1920x1080, 30 fps). Simultaneously measured my pulse by the old reliable finger-on-wrist method.
</div>
<br>

<p float="center">
  <img src="images/ppg/face_roi.png" width=480 />
  <img src="images/ppg/wrist_roi.png" width=480 /> 
</p>
<div align="center">
    <em>Frame of facial (left) and wrist-videos (right) with the region of interests highlighted with red rectangles. 
</em>
</div>

## Preprocessing

<div align="justify">
    First I read the metadata of the video to obtain precise info about the framerate, dimensions, number of frames.
Then I cropped each frame in the video to only contain the region of interest. This was usually relatively homogeneous (colour wise) so darker/lighter parts drifting in and out of the crop wouldn't throw off the calculations (in case of minor movements).
<br>
<br>
I separated the red, green and blue colour channels of the cropped images while keeping the original one. This allowed for separate examination to see if any of the channels contained more information or seemed less noisy. I checked the signals’ waveforms visually and also calculated the signal-to-noise ratio (SNR) for each of the 4 versions. Then performed the rest of the processing only on the most promising ones. 
</div>
<br>

<center>
    <img  src="images/ppg/face_long_raw_chs.png">
</center>

<div align="center">
<em>Color channels over time (in samples, here there are 1849) extracted from a face video. Large fluctuations are probably due to movement and/or lighting condition changes.</em>
</div>
<br>
<br>

| Channel     |      SNR    |
| ----------- | ----------- |
| Red         |    41.12    |
| Green       |    42.64    |
| Blue        |    41.88    |
| All         |    42.63    |


<em> Signal-to-noise ratio for the used color channels. "All" refers to the non-separated case.</em>

Based on the above table I carried out the analytic steps on the green channel and all channels combined as their SNR is the highest and  almost identical.


## Analysis

<div align="justify">
I tried several different methods to clean the signals up a bit. In the end a Butterworth filter (bandpass) worked the best. I set the lower cutoff frequency to 0.6 Hz (equivalent to 36 BPM, severe bradycardia) and the higher cutoff to a generous 3 Hz (180 BPM). 
</div>
<br>
<center>
    <img src="images/ppg/butter_fr_vs_gain.png" >
</center>

<div align="center">
<em> Frequency response of Butterworth filters with different orders. The higher the order tha sharper the cutoff. </em>
</div>
<br>

<div align="justify">
I also compared my signals with ones obtained with a built-in PPG sensor in my phone (Samsung Galaxy S8). The recordings were not concurrent but the waveforms look really similar and exhibit the <a href="https://www.researchgate.net/publication/335023100_Non-invasive_evaluation_of_coronary_heart_disease_in_patients_with_chronic_kidney_disease_using_photoplethysmography">characteristics</a> of reflected PPG measurements. 
</div>
<br>


<center>
  <img src="images/ppg/pusle_waveform_samsung_app.png" > 
</center>

<div align="center">
<em>Filtered waveforms obtained via phone built-in PPG sensor and camera.</em>
</div>
<br>

<div align="justify">
    
After smoothing the signals with this filter, the next step was to detect the peaks corresponding to heartbeats.
First I tried using numpy’s find_peaks method which aims to find all peaks fitting given criteria. I set the peak-to-peak distance (fs/high cutoff) and peak prominences (how much each peak protrudes out of its environment). With these 2 parameters I was able to identify almost all peaks relatively reliably (given clean enough input). For more noisy signals where the peak detection was less reliable I also tried estimating it by looking at the mean, modus and median of the peak-to-peak distances. While these were effective in some cases, overall they seemed too simple to be reliable. 
</div>
<br>


<center>
    <img src="images/ppg/face_long_smooth_signals.png" >
</center>


<div align="center">
<em> Signal (all channels combined) filtered with Butterworth filters with different orders. After about the 5th order there were no useful changes (order = 9 is completely distorted / overdamped). The graph also shows the detected peaks with black asterisks. Notice how the last peak is only detected in just one case. I've counted 62 pulses by hand, so it is just off by one.</em>
</div>
<br>

<div align="justify">
Another approach not reliant on counting peaks is to analyse the signal in the frequency domain. I opted for a Fast Fourier Transform (FFT) which decomposes the given signal into the sum of periodic functions with different frequencies. This method allows us to observe how much a frequency component contributes to building the original signal.  
</div>
<br>

<center>
    <img src="images/ppg/al_wrist_76_fft.png" >
</center>
<div align="center">
<em>Frequency domain representations of a signal (video of wrist, 76 BPM) filtered with different orders. Notice how the most dominant frequency component (highest peak) is the same in all 3 cases.</em>
</div>
<br>

<div align="justify">
The graphs above show that the frequency domain representation can be quite effective in determining the heart rate from a video. In this case the most dominant frequency component is clearly 1.25 Hz which would mean 75 BPM. That is only 1 off from the actual number of heartbeats for this given recording. I’ve had similarly accurate estimations for different videos, so it seems this is a rather reliable way of obtaining the heart rate.
</div>


## Conclusions

<div align="justify">
I’ve found that it is indeed possible without any special tools to measure one's heart rate just by taking a video of a body part. I’ve also managed to get pretty accurate results using both time domain (peak finding) and frequency domain (FFT) analysis. 

<br>
While some of my results are accurate I’ve had to carefully produce the videos to have favourable lighting conditions. Also the obtained signals are quite noisy as the sensor (camera) is quite far away from the subject (skin) compared to traditional PPG measurement setups.
</div>


## Code

```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.signal import lfilter, find_peaks, butter,peak_prominences, freqz
from scipy.fft import fft, fftfreq

```


```python
def signaltonoise(array, axis=0, ddof=0):
    array = np.asanyarray(array)
    avg = np.mean(array,axis)
    std = np.std(array,axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(std == 0, 0, avg/std)))



def moving_avg(series, half_wlen = 3):
    res = []
    for ind, el in enumerate(series):
        start = ind - half_wlen
        stop = ind + half_wlen
        if start >= 0 and stop <= len(series):
            res.append(np.mean(series[start:stop]))
    return res
```

```python
file_name = 'nl_face_long_62_bpm.mp4'
folder = "./data/"


# if the input is taken from the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(folder+file_name)

# framerate used during recording (fps)
fs = round(cap.get(cv2.CAP_PROP_FPS),2)

# total number of frames in video
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# getting the lenght of the video
vid_len_seconds = round(num_frames / fs, 2)
vid_len_minutes = round(vid_len_seconds / 60.0, 2)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"fps: {fs}")
print(f"total frames: {num_frames}")
print(f"length: {round(vid_len_seconds,2)} s, ({vid_len_minutes} min)")
print(f"frame dims: {width}, {height}")
```

```python
extracted_images = []

cap = cv2.VideoCapture(folder+file_name)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("End of video")
        print(f"Extracted {len(extracted_images)} frames")
        break

    # setting image up to be able to modify it
    image.flags.writeable = True

    # top left corner of crop area
    topLeft = (350, 850)

    # bottom right corner of crop area
    bottomRight = (600, 1000)

    # crop the image, keep the selection
    extracted_images.append(
        image[topLeft[1] : bottomRight[1], topLeft[0] : bottomRight[0]]
    )

    # draw rectangle around the cropped section
    cv2.rectangle(image, topLeft, bottomRight, color=(0, 0, 255), thickness=3)
    
    # open video in new window, just to check if cropping is correct
     cv2.imshow(
         "SkyNetV1",
         cv2.resize(image, (int(width / 2), int(height / 2))),
     )

    # press Esc to stop
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

```python
# m*n frames in a single color
red_ch = extracted_images[:, :, :, 0]
green_ch = extracted_images[:, :, :, 1]
blue_ch = extracted_images[:, :, :, 2]

# taking the avg of m*n frames, length is the num of frames in video
# result is just one number for each frame
red_mean = np.mean(red_ch, axis=(1, 2))
green_mean = np.mean(green_ch, axis=(1, 2))
blue_mean = np.mean(blue_ch, axis=(1, 2))

# averaging of all color channels, result has same dim as single color mean
all_ch_mean = np.mean(extracted_images, axis=(3,2,1))
```

```python
# lower cutoff frequency. Chosen as 0.6 Hz (36 BPM)
low_cut = 0.6

# upper cutoff frequency. 3 Hz is used (180 BPM)
high_cut = 3

# creating and plotting bandpass filters with same cutoffs but different orders
plt.figure()
for order in [1, 2, 3, 5, 7]:
    b, a = signal.butter(
        order, [low_cut, high_cut], btype="bandpass", analog=False, fs=fs
    )
    w, h = freqz(b, a, fs=fs, worN=512)
    plt.plot(w, abs(h), label="order = %d" % order)

plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)], "--", label="sqrt(0.5)")
plt.xlabel("$Frequency (Hz)$")
plt.ylabel("$Gain$")
plt.grid(True)
plt.legend(loc="best")
```

```python
fig = plt.figure(figsize=(20, 10))
color = ["r", "b" , "m", 'c']

for count, order in enumerate([1,3,5,9]):
    b, a = signal.butter(
        order, [low_cut, high_cut], btype="bandpass", analog=False, fs=fs
    )
    smooth_signal = signal.filtfilt(b, a, all_ch_mean)

    all_peaks, all_meta = find_peaks(
        smooth_signal, distance= fs / high_cut, prominence=0.2
    )

    plt.subplot(6, 1, count + 1)
    plt.plot(smooth_signal, color[count])
    plt.plot(all_peaks, smooth_signal[all_peaks], "k*")
    plt.title(f"Order={order}, Num peaks={len(all_peaks)}")
    
    plt.ylabel("Amplitude")
    plt.grid()

fig.tight_layout()
plt.xlabel("Time (samples)")
```



```python
N = int(fs * vid_len_seconds)

plt.figure(figsize=(20,5))

for count, order in enumerate([1,3,7]):
    b, a = signal.butter(
        order, [low_cut, high_cut], btype="bandpass", analog=False, fs=fs
    )
    smooth_signal = signal.filtfilt(b, a, all_ch_mean)
    yf = fft(smooth_signal)
    xf = fftfreq(N, 1/fs )[:N//2]

    dominant_frequency = xf[np.argmax(np.abs(yf[:N//2]))]
    
    plt.subplot(1,4,count+1)
    plt.title(f"order={order}, dominant fr.={round(dominant_frequency,2)} Hz") 

    plt.plot(xf, np.abs(yf[:N//2]))
    plt.plot(dominant_frequency,0,'rv')

    plt.xlabel("$Frequency (Hz)$")
    plt.ylabel("$Amplitude$")
    plt.xlim([0,4])
```
