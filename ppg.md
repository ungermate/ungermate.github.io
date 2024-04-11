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

## Analysis

<div align="justify">
    
</div>
