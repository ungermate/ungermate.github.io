# PPG at a distance

## Intro
Photoplethysmography (PPG) is a measurement technique used to observe volumetric changes in peripheral blood vessels.  
Usual measurement setups operate by having LEDs in close proximity to skin (finger) and measuring either the transmitted or refracted light.

I've heard the same measurement is possible at a distance using a regular phone camera. This goal of this project is to verify this claim (also it would be pretty cool if it did work).

## Strategy

1. **Gather data**
    1. Stationary body part with artifical and natural light
    2. Moving body part with stabilization (also artifical and natural light)

2. **Preprocessing**

     Stabilize video (optional), extract metadata (lenght, fps ...) and crop frames of video 

4. **Analysis**

   Look at difference between adjacent frames, apply filters, search for local maxima, examine periodicity

