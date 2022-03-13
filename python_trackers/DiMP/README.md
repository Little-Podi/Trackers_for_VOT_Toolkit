# DiMP

Code from: https://github.com/visionml/pytracking

## Preparation

### Note that this test needs a GPU with cuda toolkit, if you want to run on a single CPU, you'd better check the [GitHub issues](https://github.com/visionml/pytracking/issues?q=) for help.

### Because of system difference, it is recommended to set the environment step-by-step. My setting is a bit different with the official steps and I will write my steps later. Before that, you can follow the steps in the official repository and contact me if you meet any issues.

### To run the trackers, you need several parameters files downloaded. If you are in China and have trouble in downloading the file using official links, you can use my copy at [BaiduNetdisk](https://pan.baidu.com/s/1cj6ozS0OTWwOIghurwG-zw) with code: wyx0.

### Please replace the followings, remember to change the environment path and absolute path.

- #### in tracker_DiMP.m

  ##### python_path = 'env_path/bin/python';

  ##### pytracking_path = 'abs_path/trackers/DiMP/pytracking';

  ##### trax_path = 'abs_path/vot-toolkit/native/trax';

### Move tracker_DiMP.m to your workspace.

### Then, you can run the tracker within VOT toolkit following [VOT Chanllenge support](http://www.votchallenge.net/howto/).

## Trouble Shooting

1. ### Somehow when I want to test the tracker again, it stuck in the process of compiling C++ extensions. In terminal, it might be "Using /tmp/torch_extensions as PyTorch extensions root..." and if you use a keyboard interrupt, it will be at pytorch/torch/utils/file_baton.py, line 49, in wait time.sleep(self.wait_seconds). After I check the source code, I finally find the way out. It is caused by:

   ```python
   while os.path.exists(self.lock_file_path):
       time.sleep(self.wait_seconds)
   ```

   ### where os.path.exists is true when the path exists. Thus, I remove the existing file according to terminal outputs and in this case, it should be /tmp/torch_extensions/_prroi_pooling. Then I run the tracker and it will automatically compile the PreciseRoIPooling for us. The first time of compiling may time-consuming.

## System Requirements

- ### Ubuntu (recommended, tested with 16.04LTS)

- ### MATLAB (tested with R2017a)

- ### Nvidia GPU with cuda toolkit installed

