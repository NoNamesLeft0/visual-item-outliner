# visual-item-outliner
Use any 2D camera to measure and outline tools and other items for scaled shadow board outlines and tool box drawer organizers.

## Current State
This is a very rough proof of concept.  Exposure may cause irritability, rash, and loss of bowel control.

It's been tested on exactly one laptop with one USB camera, and it works well enough to be useful in that configuration.  I will add to this when time and inspriration overlap.  If you find a scenario that doesn't work or would like to request a feature, let me know and I'll see what I can do (probably not quickly).  Contributions welcome.

## Requirements
- Python 3.11.9

## Installation
### For Programmers
1. Ensure you have Python 3.11.9 installed (other versions will probably work but haven't been tested)
2. Clone this repository
3. Install dependencies: `pip install -r requirements.txt`

### For Non-Technical Users
An exe file is provided in the dist folder.  I expect this to run on most systems without requiring any installation, but I've only tested it on a Windows 10 machine with all the python dependencies installed for development.

## General Function
VIO (visual item outliner) scans items and outputs a scaled pdf file with black silhouettes on a white background.  This can be printed, imported into laser cutter software, or in other ways.  The intent is to cut out foam toolbox organizers, make shadow boards, etc without having to measure and draw each item. 

## Usage
### Physical Setup
Mount a camera looking down at a table or other surface with uniform color.  It's helpful if this color is significantly different than the items that will be scanned (think green screen).  The camera should be roughly perpendicular to the table surface, but small misalignment is fine because we'll calibrate it anyway.  

Uniform lighting is very helpful to prevent shadows, which can result in inaccurate outlines.

### Setup the camera
`Settings | Camera Setup`
A live feed will display the current camera image.  Ensure good focus and set the exposure to give good contrast on the calibration grid.
If multiple cameras are connected, select the desired camera.  

### Calibration
This process allows the software to measure items in engineering units (mm), and also compensate for lens distortion.  A calibration grid will be required.  This is just a checkerboard pattern with a known square size.  For most purposes, an office printer is accurate enough to make this.  See the folder "Calibration Grids" or bring your own (https://markhedleyjones.com/projects/calibration-checkerboard-collection is a handy source).  Make sure the printer settings DO NOT SCALE when printing.  Options like "size to fit" when printing will change the size of the grid and throw off the calibration.

`Settings | Calibration`

Ensure all grid vertices (where the squares' corners meet) are visible to the camera.  Count these vertices in the horizontal and vertical directions and enter them before clicking "Calibrate."

### Take Pictures
A picture will be required of the background and the items being "scanned".  Check out the *Difference and Sum to BW* tab if you're curious what's happening inside the program.  

### Binarize the Image
On the *Binarized Image* tab, a histogram and black and white image are displayed.  Use the mouse to adjust the binarization threshold on the histogram until the picture looks good (silhouettes the items accurately).

### Check and Save
Look at the final, calibrated image on the *Calibration* tab.  If it looks good, 'File | Save Image'.  A pdf file will be saved to the current directory.