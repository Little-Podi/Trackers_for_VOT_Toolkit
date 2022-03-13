# Trackers for VOT Toolkit

## Introduction

### This repository contains several trackers I have integrated into [VOT toolkit](https://github.com/votchallenge/vot-toolkit), you can see the README in each tracker's folder for more details.

### Successfully integrated trackers at present (some of them haven't uploaded at present):

- #### MATLAB trackers: DAT, ECO, GFS-DCF, KCF, LADCF, MCCT, SRDCF, Staple, UPDT, ASRCF, BACF, CCOT, CFNet, MDNet, SiamFC, SiamFC-triplet, DSST.

- #### Python trackers: ATOM, DiMP.

### Specifically, the code is tested with the following version of VOT toolkit: [version link](https://github.com/votchallenge/vot-toolkit/tree/c53fa23ba1fe59181af63fff783ab243ffeeff4b).

> ### Note: Codes provided here are mainly for evaluation by VOT toolkit, for further development, please visit the link (if given) in the README for each tracker to get more information.

> ### Update: The [MATLAB toolkit](https://github.com/votchallenge/toolkit-legacy) has been archived now. The new [Python toolkit](https://github.com/votchallenge/toolkit) is recommanded.

## System Requirements

### All trackers included were tested on Ubuntu18.04 with MATLAB2018b. Only a single CPU is needed for the evaluation in this repository and it's not a problem because VOT toolkit provides an evaluation strategy that can ignore the influence of different hardware.

## Suggestions

### Trying to run NCC first with VOT toolkit is recommended. To achieve this, you can follow  [VOT Challenge support](http://www.votchallenge.net/howto/). You may also need  [VOT Challenge technical support](https://groups.google.com/forum/?hl=en#!forum/votchallenge-help) and [VOT toolkit issues](https://github.com/votchallenge/vot-toolkit/issues?utf8=%E2%9C%93&q=https://github.com/votchallenge/vot-toolkit/issues?utf8=✓&q=) to search for more help.

### You'd better check whether the benchmark wrapper is provided by the author before you want to integrate a new tracker. This can save your time. If there aren't any benchmark wrappers provided or there is but crushed, you can follow NCC example and the guide on the official website to integrate the tracker by yourself.

### By default, each tracker will repeat 3 times when you execute run_experiments.m. If you want to run your tracker only once during the experiment, you can add the following code between workspace_load() and workspace_evaluate().

```matlab
experiments{1,1}.parameters.repetitions = 1;
```

### By default, the evaluation includes three experiments, i.e. baseline, unsupervised and realtime. If you want to perform the evaluation only on baseline experiment or some of the experiments above. You can alter the variable in the corresponding experiment stack e.g. vot-toolkit/stacks/stack_vot2018.m for VOT2018.

- #### Find this:

```matlab
experiments = {baseline, unsupervised, realtime};
```

- #### Change to:

```matlab
experiments = {baseline};
```

#### You can also set any other combinations you want.

## Methods

### As far as I know, there mainly three ways to integrate a new tracker.

1. ### As mentioned in suggestions, if the wrapper is provided by the author, you will be lucky because all you need is to replace the vot function files such as vot.m or something.

2. ### However, because of the big difference between each version of VOT toolkit, the method above doesn't work in many ways. During the integration work, I found that if the input of the run file is the sequence while the output is the results (i.e. OTB format), then the wrapper provided by Martin Danelljan may be very useful.

   ```matlab
   function results = runfile(seq, res_path, bSaveImage, parameters)
   ```

   ### Needless to say, it works for the trackers proposed by Martin himself, such as CCOT, ECO, SRDCF, UPDT, etc. It seems that it also works when I apply it to GFSDCF's demo.

3. ### The most universal approach is following  [VOT Chanllenge support](http://www.votchallenge.net/howto/) and the example NCC to add handle at the proper positions so that the information can be recognized and obtained by VOT toolkit. More examples like KCF, Staple, DAT etc. can be found in this repository.

## Trouble Shooting

1. ### <u>Tracker execution interrupted: Unable to establish connection</u>

#### When I was trying to run NCC, I met with this ERROR: "Tracker execution interrupted: Unable to establish connection" and it indicates that "TraX support not detected" or "Tracker has not passed the TraX support test".

#### Actually, this problem can be avoided by modifying vot-toolkit/tracker/tracker_run.m a little bit.

- #### Find this:

  ```matlab
  connection = 'standard';
  ```

- #### Replace the above with:

  ```matlab
  connection = 'socket';
  ```

#### After that, if the issue still exists, then the main reason could be the absent or the wrong version of vot.m in your tracker directory. In that case, you'd better move the vot.m in vot-toolkit/tracker/examples/matlab/ to the right place in the corresponding tracker directory.

2. ### <u>Tracker execution interrupted: Did not receive response</u>

#### If you have already passed the TraX support test but receive this ERROR: "Tracker execution interrupted: Did not receive response". This is mainly caused by tracker crash. You'd better check the generated log file for the tracker you want to run under the file vot-workspace/logs/tracker_name.

#### If there is no problem with your code, then the problem may come from the environment. Take my experience as an example, when I tried to run NCC and some other MATLAB trackers, I always met the ERROR above and the log files told me some functions such as normxcorr2, configureKalmanFilter, etc. were undefined. This is because some required MATLAB toolboxes were not installed. If you only install the default toolboxes, then it will be hard for visual object trackers written in MATLAB to run and the VOT toolkit also cannot work. In that case, you can search the undefined function to find the corresponding toolbox. The following toolboxes are recommended  to install.

- MATLAB (default in MATLAB R2018b)
- Simulink (default in MATLAB R2018b)
- Computer Vision System Toolbox
- Deep Learning Toolbox
- Image Acquisition Toolbox
- Image Processing Toolbox (required by VOT toolkit)
- Parallel Computing Toolbox
- Sensor Fusion and Tracking Toolbox
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox (default in MATLAB R2018b)
- Symbolic Math Toolbox (default in MATLAB R2018b)
- Vision HDL Toolbox

#### Some of these may be redundant, but I promise that you can successfully run the tracker with these toolboxes.

#### If you have already installed the MATLAB, don't worry, you can run the MATLAB install file again to add the toolboxes above or use add-ons in MATLAB GUI to search and install.

3. ### <u>Tracker execution interrupted: Invalid MEX-file</u>

#### I also met this ERROR: "Tracker execution interrupted: Invalid MEX-file". The list of the invalid MEX-files might be short or long, but I think the reason is the same. This issue is due to the old version of the libstdc++. To solve this, you can execute the following commands in terminal.

```l
cd /usr/local/MATLAB/R2018b/sys/os/glnxa64
sudo mv libstdc++.so.6.0.20 bak-libstdc++.so.6.0.20
sudo mv libstdc++.so.6 bak-libstdc++.so.6
sudo ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21 ./
sudo ln -sf ./libstdc++.so.6.0.21 ./libstdc++.so.6
```

#### Note that in the first command, you should change to the directory where your MATLAB is installed. Otherwise, the followed commands may cause some system problem.

4. ### <u>Warning: The version of gcc is not supported</u>

#### During the compile process, you may keep getting the warning that warn you that your version is not suitable. Actually, it doesn't matter. Just ignore it and compiling still can be done.

#### However, if you are using a very old version of MATLAB and the toolkit still cannot work, then maybe you really need to change the version of gcc.

#### You can check the version of gcc in your system by:

```
gcc --version
```

#### or simply by:

```
gcc -v
```

#### You can change the version of gcc and g++ (e.g. to version 4.7 for old version MATLAB) by executing the following commands:

```
sudo apt-get install gcc-4.7
sudo apt-get install g++-4.7
cd usr/bin
sudo rm gcc
sudo ln -s gcc-4.7 gcc
sudo rm g++
sudo ln -s g++-4.7 g++
ls -al gcc g++
```

5. ### <u>Tracker execution interrupted: Unable to start the tracker process</u>

#### This problem is caused by incorrect path setting. It always comes with an error notifies you that some functions cannot be found. You can check your path setting according to the location of the missing function. What's more, make sure there are not any other languages expect for English in your path.

6. ### <u>Tracker execution interrupted: Tracker process not alive</u>

#### This problem can be caused by wrong runfiles (maybe runfiles for other benchmarks such as OTB). Check whether you are using the correct runfile for VOT dataset and your issue will be solved.

7. ### <u>(Python) Tracker has not passed the TraX support test</u>

#### Note that the configuration files for other languages are different from MATLAB. Before trying a new language, you'd better check the template in vot-toolkit/tracker/examples/tracker_Demo_language.m for corresponding language.

#### For Python trackers, you may need two more steps for preparation on the basis of the MATLAB settings.

1. #### Compile TraX by CMake:

   #### To get the libtrax.so you will have to compile libtrax manually with CMake or download precompiled bundles from bintray because MATLAB cannot compile anything that is not a Mex file. The instruction can be found in the [TraX library documentation](https://trax.readthedocs.io/en/latest/). As for me, I use the first way by executing the following commands:

   ```
   cd vot-toolkit/native/trax
   mkdir build
   cd build 
   cmake ..
   make
   ```

2. #### Set the configuration file for Python tracker:

   #### The template in tracker_Demo_py.m is like this:

   ```matlab
   tracker_label = 'Demo_py';
   
   % Note: be carefull for double backslashes on Windows
   tracker_command = generate_python_command('python_static_rgbd', ...
       {'<path-to-toolkit-dir>\\tracker\\examples\\python', ...  % tracker source and vot.py are here
       '<path-to-trax-dir>\\support\\python'});
   
   tracker_interpreter = 'python';
   
   tracker_linkpath = {'<path-to-trax-build-dir>\bin', ...
       '<path-to-trax-build-dir>\lib'};
   ```

   #### Note the path-to-trax-build-dir here is the location of libtrax.so. If you have compiled the TraX as step 1, then the configuration file in your workspace can be set like this (take python_ncc as a example):

   ```matlab
   tracker_label = ['NCCpy'];
   
   tracker_command = generate_python_command('python_ncc', {'abs_path/vot-toolkit/tracker/examples/python', 'abs_path/vot-toolkit/native/trax/support/python'});
   
   tracker_interpreter = 'python';
   
   tracker_linkpath = {'abs_path/vot-toolkit/native/trax/build'};
   ```

   #### Remember to change the abs_path.

#### After the settings above, if there is still a error says tracker has not passed the TraX support test, then it may caused by your environment. You should check the log files and you may need to install the missing modules using pip and conda, or set the path correctly.

#### Recently I found that a new official [VOT toolkit](https://github.com/votchallenge/vot-toolkit-python) implemented in Python 3 has been released, still trying.

#### For more trouble shooting, you can check the README file for each tracker in this repository. I hope these will help you.