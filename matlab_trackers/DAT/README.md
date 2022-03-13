# DAT

Code from: https://data.votchallenge.net/vot2018/trackers/DAT-code-2018-06-15T07_36_06.008096.zip

## Preparation

### Remember to change the absolute path in the configuration file.

- #### in tracker_DAT.m

  ##### tracker_command = generate_matlab_command('dat', {'abs_path/DAT', 'abs_path/scale_regression'});

### Move tracker_DAT.m to your workspace.

### Then, you can run the tracker within VOT toolkit following [VOT Chanllenge support](http://www.votchallenge.net/howto/).

## Trouble Shooting

1. ### If you met with an ERROR like this: "Tracker execution interrupted: Did not receive response.", you can try to replace DAT/vot.m with the vot.m in vot-toolkit/tracker/examples/matlab/. If it still failed, you'd better use the same version of the VOT toolkit which is given in the README of this repository.

## System Requirements

- ### Ubuntu (tested with 18.04LTS)

- ### MATLAB (tested with R2018b)

- ### The result is obtained using a single CPU