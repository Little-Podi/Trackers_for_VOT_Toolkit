# ECO

Code from: https://codeload.github.com/martin-danelljan/ECO/zip/master

## Preparation

### This software is implemented using MatConvNet.

#### Please download [MatConvNet](https://codeload.github.com/vlfeat/matconvnet/zip/master) and unzip it in external_libs/matconvnet/.

#### Please compile MatConvNet according to the [installation guideline](http://www.vlfeat.org/matconvnet/install/).

### Deep CNN features need pre-trained CNN models.

#### You can dowload [imagenet-vgg-m-2048](https://www.vlfeat.org/matconvnet/models/imagenet-vgg-m-2048.mat) and place it in feature_extraction/networks/.

#### You can also use [other networks](https://www.vlfeat.org/matconvnet/pretrained/), by placing them in the feature_extraction/networks/ folder.

### HOG features need the PDollar Toolbox.

#### Please download [PDollar Toolbox](https://codeload.github.com/pdollar/toolbox/zip/master) and unzip it in external_libs/pdollar_toolbox/.

### Make sure you have run install.m before the test.

### Please replace the followings, remember to change the absolute path.

- #### in tracker_ECO.m

  ##### tracker_command = generate_matlab_command('eco(''ECO'', ''VOT2016_DEEP_settings'', true)', {'abs_path/ECO'});

### Move tracker_ECO.m to your workspace.

### Then, you can run the tracker within VOT toolkit following [VOT Chanllenge support](http://www.votchallenge.net/howto/).

## Trouble Shooting

1. ### If you met with an ERROR like this: "Tracker execution interrupted: Did not receive response.", you can try to replace ECO/vot.m with the vot.m in vot-toolkit/tracker/examples/matlab/. If it still failed, you'd better use the same version of the VOT toolkit which is given in the README of this repository.

2. ### If you use the imagenet-vgg-m-2048.mat downloaded by install.m, the MAT file may can't be loaded. In that case, you'd better download the pre-trained model independently from the link given in the Preparation part above.

## System Requirements

- ### Ubuntu (tested with 18.04LTS)

- ### MATLAB (tested with R2018b)

- ### The result is obtained using a single CPU