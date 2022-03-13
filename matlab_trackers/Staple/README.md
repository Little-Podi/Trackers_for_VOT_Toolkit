# Staple

Code from: http://data.votchallenge.net/vot2018/trackers/Staple-code-2018-06-17T14_05_02.087450.zip

## Preparation

### The tracker requires OpenCV 2.4 to be installed on the system.

#### Specifically, it depends on libraries libopencv_core.so.2.4 and libopencv_imgproc.so.2.4.

> #### p.s. I installed OpenCV 4.0 on my system, it seems that version 4.0 also works for this tracker.

### Please replace the followings, remember to change the absolute path.

- #### in tracker_Staple.m

  ##### tracker_command = generate_matlab_command('staple', {'abs_path/Staple'});

### Move tracker_Staple.m to your workspace.

### Then, you can run the tracker within VOT toolkit following [VOT Chanllenge support](http://www.votchallenge.net/howto/).

## Trouble Shooting

1. ### If you met with an ERROR like this: "Tracker execution interrupted: Did not receive response.", try to replace Staple/vot.m with the vot.m in vot-toolkit/tracker/examples/matlab/. If it still failed, you'd better use the same version of the VOT toolkit which is given in the README of this repository.

## System Requirements

- ### Ubuntu (tested with 18.04LTS)

- ### MATLAB (tested with R2018b)

- ### The result is obtained using a single CPU