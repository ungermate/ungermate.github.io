# PPG at a distance

## Intro
Photoplethysmography (PPG) is a measurement technique used to observe volumetric changes in peripheral blood vessels.  
Usual measurement setups operate by having LEDs in close proximity to skin (finger) and measuring either the transmitted or refracted light.

I've heard the same measurement is possible at a distance using a regular phone camera. This goal of this project is to verify this claim (also it would be pretty cool if it did work).

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


I recorded my hand, wrist and face for 1 minute each in natural and artificial light (non-flickering LEDs to avoid 60 Hz artefact). I tried to minimise movement by fixing the phone and/or bodypart’s position. I used the main camera of a Samsung Galaxy S8 (1920x1080, 30 fps)

</div>

## Preprocessing

<div align="justify">
    First I read the metadata of the video to obtain precise info about the framerate, dimensions, number of frames.
Then I cropped each frame in the video to only contain the region of interest. This was usually relatively homogeneous (colour wise) so darker/lighter parts drifting in and out of the crop wouldn't throw off the calculations (in case of minor movements).

I separated the red, green and blue colour channels of the cropped images while keeping the original one. This allowed for separate examination to see if any of the channels contained more information or seemed less noisy. I checked the signals’ waveforms visually and also calculated the signal-to-noise ratio (SNR) for each of the 4 versions. Then performed the rest of the processing only on the most promising ones. 
</div>

<center>
    <img  src="images/ppg/face_long_raw_chs.png">
</center>
<em>Color channels over time (in samples, here there are 1849) extracted from a face video. Large fluctuations are probably due to movement and/or lighting condition changes.</em>


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

<center>
    <img src="images/ppg/butter_fr_vs_gain.png" >
</center>

<em> Frequency response of Butterworth filters with different orders. The higher the order tha sharper the cutoff. </em>




<div align="justify">
    
After smoothing the signals with this filter, the next step was to detect the peaks corresponding to heartbeats.
First I tried using numpy’s find_peaks method which aims to find all peaks fitting given criteria. I set the peak-to-peak distance (fs/high cutoff) and peak prominences (how much each peak protrudes out of its environment). With these 2 parameters I was able to identify almost all peaks relatively reliably (given clean enough input). For more noisy signals where the peak detection was less reliable I also tried estimating it by looking at the mean, modus and median of the peak-to-peak distances. While these were effective in some cases, overall they seemed too simple to be reliable. 


</div>

<center>
    <img src="images/ppg/face_long_smooth_signals.png" >
</center>


<div align="justify">
<em> Signal (all channels combined) filtered with Butterworth filters with different orders. After about the 5th order there were no useful changes (order = 9 is completely distorted). The graph also shows the detected peaks with black asterisks. Notice how the last peak is only detected in just one case.</em>
</div>



