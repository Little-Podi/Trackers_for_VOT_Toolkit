function dat()
% DAT integration wrapper

% *************************************************************
% VOT: Set random seed to a different value every time.
% *************************************************************
try
    % Simple check for Octave environment
    OCTAVE_VERSION;
    rand('seed', sum(clock));
    pkg load image;
catch
    RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));
end

% **********************************
% VOT: Get initialization data
% **********************************
[handle, image, region] = vot('polygon');

% Initialize the tracker
cfg = default_parameters_dat();
[state, ~] = tracker_dat_initialize(imread(image), region, cfg);

while true
    % **********************************
    % VOT: Get next frame
    % **********************************
    [handle, image] = handle.frame(handle);

    if isempty(image)
        break;
    end

    % Perform a tracking step, obtain new region
    [state, region, confidence] = tracker_dat_update(state, imread(image), cfg);

    % **********************************
    % VOT: Report position for frame
    % **********************************
    handle = handle.report(handle, region, confidence);

end

% **********************************
% VOT: Output the results
% **********************************
handle.quit(handle);

end
