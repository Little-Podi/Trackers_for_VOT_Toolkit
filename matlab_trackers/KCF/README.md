# KCF

Code from: https://codeload.github.com/scott89/KCF/zip/master

## Preparation

### Please replace the followings, remember to change the absolute path.

- #### in tracker_KCF.m

  ##### tracker_command = generate_matlab_command('kcf', {'abs_path/KCF'});

### Move tracker_KCF.m to your workspace.

### Then, you can run the tracker within VOT toolkit following [VOT Chanllenge support](http://www.votchallenge.net/howto/).

## Trouble Shooting

1. ### If you met with an ERROR like this: "Tracker execution interrupted: Did not receive response.", you can try to replace KCF/vot.m with the vot.m in vot-toolkit/tracker/examples/matlab/. If it still failed, you'd better use the same version of the VOT toolkit which is given in the README of this repository.

## System Requirements

- ### Ubuntu (tested with 18.04LTS)

- ### MATLAB (tested with R2018b)

- ### The result is obtained using a single CPU