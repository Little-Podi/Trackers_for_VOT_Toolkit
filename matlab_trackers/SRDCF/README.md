# SRDCF

Code from: https://codeload.github.com/martin-danelljan/ECO/zip/master

## Preparation

### The same as ECO and CCOT, SRDCF is also proposed by Martin Danelljan and its integration to VOT toolkit is included in ECO which can be found in this repository. If you have already successfully done with ECO, you can integrate SRDCF tracker easily by following the steps below.

### Please replace the followings, remember to change the absolute path.

- #### in tracker_SRDCF.m

  ##### tracker_command = generate_matlab_command('eco(''SRDCF'', ''SRDCF_settings'', true)', {'abs_path/ECO'});

### Move tracker_SRDCF.m to your workspace.

### Then, you can run the tracker within VOT toolkit following [VOT Chanllenge support](http://www.votchallenge.net/howto/).

## Trouble Shooting

1. ### If you met with an ERROR like this: "Tracker execution interrupted: Did not receive response.", you can try to replace ECO/vot.m with the vot.m in vot-toolkit/tracker/examples/matlab/. If it still failed, you'd better use the same version of the VOT toolkit which is given in the README of this repository.

2. ### If you use the imagenet-vgg-m-2048.mat downloaded by install.m, the MAT file may can't be loaded. In that case, you'd better download the pre-trained model independently from the link given in the Preparation part above.

## System Requirements

- ### Ubuntu (tested with 18.04LTS)

- ### MATLAB (tested with R2018b)

- ### The result is obtained using a single CPU