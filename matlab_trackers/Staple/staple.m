function staple
% *************************************************************
% VOT: Always call exit command at the end to terminate Matlab!
% *************************************************************
cleanup = onCleanup(@() exit());

% *************************************************************
% VOT: Set random seed to a different value every time.
% *************************************************************
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));

% **********************************
% VOT: Get initialization data
% **********************************
[handle, image_init, region_init] = vot('polygon'); % Obtain communication object
[cx, cy, w, h] = get_axis_aligned_BB(region_init);
region = [cx, cy, w, h];
% image_init = '/home/berti/datasets/validation/tc_Ball_ce2/0001.jpg';
% region_init = [344,339,40,43];
% region = [region_init(1)+region_init(3)/2 region_init(2)+region_init(4)/2 region_init(3) region_init(4)];
image = imread(image_init);
p = params_init(image);

pos = [region(2), region(1)];
target_sz = round([region(4), region(3)]);

[p, bg_area, fg_area, area_resize_factor] = initializeAllAreas(image, target_sz, p);
% Init visualization
videoPlayer = [];
if p.visualization && isToolboxAvailable('Computer Vision System Toolbox')
    videoPlayer = vision.VideoPlayer('Position', [100, 100, [size(image, 2), size(image, 1)] + 30]);
end

% patch of the target + padding
patch_padded = getSubwindow(image, pos, p.norm_bg_area, bg_area);
% initialize hist model
new_pwp_model = true;
[bg_hist, fg_hist] = updateHistModel(new_pwp_model, patch_padded, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence);
new_pwp_model = false;
% Hann (cosine) window
if isToolboxAvailable('Signal Processing Toolbox')
    hann_window = single(hann(p.cf_response_size(1))*hann(p.cf_response_size(2))');
else
    hann_window = single(myHann(p.cf_response_size(1))*myHann(p.cf_response_size(2))');
end

% gaussian-shaped desired response, centred in (1,1)
% bandwidth proportional to target size
output_sigma = sqrt(prod(p.norm_target_sz)) * p.output_sigma_factor / p.hog_cell_size;
y = gaussianResponse(p.cf_response_size, output_sigma);
yf = fft2(y);
% Scale adaptation initialization
if p.scale_adaptation
    scale_factor = 1;
    base_target_sz = target_sz;
    scale_sigma = sqrt(p.num_scales) * p.scale_sigma_factor;
    ss = (1:p.num_scales) - ceil(p.num_scales/2);
    ys = exp(-0.5*(ss.^2)/scale_sigma^2);
    ysf = single(fft(ys));
    if mod(p.num_scales, 2) == 0
        scale_window = single(hann(p.num_scales+1));
        scale_window = scale_window(2:end);
    else
        scale_window = single(hann(p.num_scales));
    end

    ss = 1:p.num_scales;
    scale_factors = p.scale_step.^(ceil(p.num_scales/2) - ss);

    if p.scale_model_factor^2 * prod(p.norm_target_sz) > p.scale_model_max_area
        p.scale_model_factor = sqrt(p.scale_model_max_area/prod(p.norm_target_sz));
    end

    scale_model_sz = floor(p.norm_target_sz*p.scale_model_factor);
    % find maximum and minimum scales
    min_scale_factor = p.scale_step^ceil(log(max(5./bg_area))/log(p.scale_step));
    max_scale_factor = p.scale_step^floor(log(min([size(image, 1), size(image, 2)]./target_sz))/log(p.scale_step));
end

frame = 1;

while true

    if frame > 1
        % **********************************
        % VOT: Get next frame
        % **********************************
        [handle, image] = handle.frame(handle); % Get the next frame
        % image = '/home/berti/datasets/validation/tc_Ball_ce2/0002.jpg';
        if isempty(image) % Are we done?
            break;
        end

        % Testing step
        image = imread(image);
        % extract patch of size bg_area and resize to norm_bg_area
        im_patch_cf = getSubwindow(image, pos, p.norm_bg_area, bg_area);
        pwp_search_area = round(p.norm_pwp_search_area/area_resize_factor);
        % extract patch of size pwp_search_area and resize to norm_pwp_search_area
        im_patch_pwp = getSubwindow(image, pos, p.norm_pwp_search_area, pwp_search_area);
        % compute feature map
        xt = getFeatureMap(im_patch_cf, p.feature_type, p.cf_response_size, p.hog_cell_size);
        % apply Hann window
        xt_windowed = bsxfun(@times, hann_window, xt);
        % compute FFT
        xtf = fft2(xt_windowed);
        % Correlation between filter and test patch gives the response
        % Solve diagonal system per pixel.
        if p.den_per_channel
            hf = hf_num ./ (hf_den + p.lambda);
        else
            hf = bsxfun(@rdivide, hf_num, sum(hf_den, 3)+p.lambda);
        end
        response_cf = ensure_real(ifft2(sum(conj(hf).*xtf, 3)));

        % Crop square search region (in feature pixels).
        response_cf = cropFilterResponse(response_cf, ...
            floor_odd(p.norm_delta_area/p.hog_cell_size));
        if p.hog_cell_size > 1
            % Scale up to match center likelihood resolution.
            response_cf = mexResize(response_cf, p.norm_delta_area, 'auto');
        end

        [likelihood_map] = getColourMap(im_patch_pwp, bg_hist, fg_hist, p.n_bins, p.grayscale_sequence);
        likelihood_map(isnan(likelihood_map)) = 0;

        % each pixel of response_pwp loosely represents the likelihood that
        % the target (of size norm_target_sz) is centred on it
        response_pwp = getCenterLikelihood(likelihood_map, p.norm_target_sz);

        % Estimation
        response = mergeResponses(response_cf, response_pwp, p.merge_factor, p.merge_method);
        [row, col] = find(response == max(response(:)), 1);
        center = (1 + p.norm_delta_area) / 2;
        pos = pos + ([row, col] - center) / area_resize_factor;

        % Scale space search
        if p.scale_adaptation
            im_patch_scale = getScaleSubwindow(image, pos, base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
            xsf = fft(im_patch_scale, [], 2);
            scale_response = real(ifft(sum(sf_num.*xsf, 1)./(sf_den + p.lambda)));
            recovered_scale = ind2sub(size(scale_response), find(scale_response == max(scale_response(:)), 1));
            % set the scale
            scale_factor = scale_factor * scale_factors(recovered_scale);

            if scale_factor < min_scale_factor
                scale_factor = min_scale_factor;
            elseif scale_factor > max_scale_factor
                scale_factor = max_scale_factor;
            end
            % use new scale to update bboxes for target, filter, bg and fg models
            target_sz = round(base_target_sz*scale_factor);
            avg_dim = sum(target_sz) / 2;
            bg_area = round(target_sz+avg_dim);
            if (bg_area(2) > size(image, 2)), bg_area(2) = size(image, 2) - 1; end
            if (bg_area(1) > size(image, 1)), bg_area(1) = size(image, 1) - 1; end

            bg_area = bg_area - mod(bg_area-target_sz, 2);
            fg_area = round(target_sz-avg_dim*p.inner_padding);
            fg_area = fg_area + mod(bg_area-fg_area, 2);
            % Compute the rectangle with (or close to) params.fixed_area and
            % same aspect ratio as the target bboxgetScaleSubwindow
            area_resize_factor = sqrt(p.fixed_area/prod(bg_area));
        end

        region = [pos([2, 1]) - target_sz([2, 1]) / 2, target_sz([2, 1])];

        % **********************************
        % VOT: Report position for frame
        % **********************************
        handle = handle.report(handle, region); % Report position for the given frame
    end

    rectPosition = [pos([2, 1]) - target_sz([2, 1]) / 2, target_sz([2, 1])];
    if p.visualization
        vis_im = gather(image);
        if isempty(videoPlayer)
            figure(1), imshow(vis_im);
            figure(1), rectangle('Position', rectPosition, 'LineWidth', 4, 'EdgeColor', 'y');
            drawnow
            % fprintf('Frame %d\n', p.startFrame+i);
        else
            vis_im = insertShape(vis_im, 'Rectangle', rectPosition, 'LineWidth', 4, 'Color', 'yellow');
            % Display the annotated video frame using the video player object.
            step(videoPlayer, vis_im);
        end
    end

    % Training
    % extract patch of size bg_area and resize to norm_bg_area
    im_patch_bg = getSubwindow(image, pos, p.norm_bg_area, bg_area);
    % compute feature map, of cf_response_size
    xt = getFeatureMap(im_patch_bg, p.feature_type, p.cf_response_size, p.hog_cell_size);
    % apply Hann window
    xt = bsxfun(@times, hann_window, xt);
    % compute FFT
    xtf = fft2(xt);
    % Filter update
    % Compute expectations over circular shifts,
    % therefore divide by number of pixels.
    new_hf_num = bsxfun(@times, conj(yf), xtf) / prod(p.cf_response_size);
    new_hf_den = (conj(xtf) .* xtf) / prod(p.cf_response_size);

    if frame == 1
        % first frame, train with a single image
        hf_den = new_hf_den;
        hf_num = new_hf_num;
    else
        % subsequent frames, update the model by linear interpolation
        hf_den = (1 - p.learning_rate_cf) * hf_den + p.learning_rate_cf * new_hf_den;
        hf_num = (1 - p.learning_rate_cf) * hf_num + p.learning_rate_cf * new_hf_num;

        % BG/FG model update
        % patch of the target + padding
        [bg_hist, fg_hist] = updateHistModel(new_pwp_model, im_patch_bg, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence, bg_hist, fg_hist, p.learning_rate_pwp);
    end

    % Scale update
    if p.scale_adaptation
        im_patch_scale = getScaleSubwindow(image, pos, base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
        xsf = fft(im_patch_scale, [], 2);
        new_sf_num = bsxfun(@times, ysf, conj(xsf));
        new_sf_den = sum(xsf.*conj(xsf), 1);
        if frame == p.startFrame
            sf_den = new_sf_den;
            sf_num = new_sf_num;
        else
            sf_den = (1 - p.learning_rate_scale) * sf_den + p.learning_rate_scale * new_sf_den;
            sf_num = (1 - p.learning_rate_scale) * sf_num + p.learning_rate_scale * new_sf_num;
        end
    end
    frame = frame + 1;
end

% **********************************
% VOT: Output the results
% **********************************
handle.quit(handle); % Output the results and clear the resources

end

function p = params_init(image, varargin)
p.grayscale_sequence = false; % suppose that sequence is colour
p.hog_cell_size = 4;
p.fixed_area = 22452; % standard area to which we resize the target
p.n_bins = 16; % number of bins for the color histograms (bg and fg models)
p.learning_rate_pwp = 0.023; % bg and fg color models learning rate
p.feature_type = 'fhog'; % {'fhog','gray'}
p.inner_padding = 0.0149; % defines inner area used to sample colors from the foreground
p.output_sigma_factor = 0.0892; % standard deviation for the desired translation filter output
p.lambda = 1e-3; % regularization weight
p.learning_rate_cf = 0.0153; % HOG model learning rate
p.merge_factor = 0.567; % fixed interpolation factor - how to linearly combine the two responses
p.merge_method = 'const_factor'; % {'const_factor','fit_gaussian'}
p.den_per_channel = false;
% scale related
p.scale_adaptation = true;
p.hog_scale_cell_size = 4; % Default DSST=4
p.learning_rate_scale = 0.0245;
p.scale_sigma_factor = 0.355;
p.num_scales = 27;
p.scale_model_factor = 1.0;
p.scale_step = 1.0292;
p.scale_model_max_area = 32 * 16;
% environment stuff
p.visualization = 0; % show output bbox on frame
p.init_gpu = 0;
p.visualization_dbg = 0; % show also per-pixel scores, desired response and filter output
p.video = '';
p.track_lost = [];
p.startFrame = 1;
p.fout = -1;
p.imgFiles = [];
p.targetPosition = [];
p.targetSize = [];
p.track_lost = [];
p.ground_truth = [];
% p = vl_argparse(p, varargin);
if (size(image, 3) == 1)
    p.grayscale_sequence = true;
end
end

% Reimplementation of Hann window (in case signal processing toolbox is missing)
function H = myHann(X)
H = 0.5 * (1 - cos(2*pi*(0:X - 1)'/(X - 1)));
end

% We want odd regions so that the central pixel can be exact
function y = floor_odd(x)
y = 2 * floor((x - 1)/2) + 1;
end

function y = ensure_real(x)
assert(norm(imag(x(:))) <= 1e-5*norm(real(x(:))));
y = real(x);
end
