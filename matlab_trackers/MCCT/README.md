# MCCT

Code from: http://data.votchallenge.net/vot2018/trackers/MCCT-code-2018-06-15T03_41_51.957322.zip

## Preparation

### This software is implemented using [MatConvNet](http://www.vlfeat.org/matconvnet/).

#### Please compile MatConvNet according to the [installation guideline](http://www.vlfeat.org/matconvnet/install/).

### Download the VGG-Net-19 and place the mat file it in MCCT/model/.

#### You can use [the link](https://uofi.box.com/shared/static/kxzjhbagd6ih1rf7mjyoxn2hy70hltpl.mat) to download.

#### If you are in China, please use [this link](http://pan.baidu.com/s/1kU1Me5T).

### Please replace the followings, remember to change the absolute path.

- #### in mcct.m

  ##### addpath 'abs_path/MCCT/tracker';

  ##### addpath 'abs_path/MCCT/external/matconvnet/matlab';

  ##### addpath 'abs_path/MCCT/model';

  ##### addpath 'abs_path/MCCT/utility';

- #### in tracker/initial_net.m

  ##### net = load(fullfile('abs_path/MCCT/model/imagenet-vgg-verydeep-19.mat'));

- #### in tracker_MCCT.m

  ##### tracker_command = generate_matlab_command('mcct', {'abs_path/MCCT'});

### Move tracker_MCCT.m to your workspace.

### Then, you can run the tracker within VOT toolkit following [VOT Chanllenge support](http://www.votchallenge.net/howto/).

## Trouble Shooting

1. ### If you met with an ERROR like this: "Tracker execution interrupted: Did not receive response.", you can try to replace MCCT/vot.m with the vot.m in vot-toolkit/tracker/examples/matlab/. If it still failed, you'd better use the same version of the VOT toolkit which is given in the README of this repository.

## System Requirements

- ### Ubuntu (tested with 18.04LTS)

- ### MATLAB (tested with R2018b)

- ### The result is obtained using a single CPU
