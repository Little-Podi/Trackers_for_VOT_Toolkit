function mcct

% *************************************************************
% VOT: Always call exit command at the end to terminate Matlab!
% *************************************************************
cleanup = onCleanup(@() exit());

% *************************************************************
% VOT: Set random seed to a different value every time.
% *************************************************************
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));

addpath('abs_path/MCCT/tracker');
addpath('abs_path/MCCT/external/matconvnet/matlab');
addpath('abs_path/MCCT/model');
addpath('abs_path/MCCT/utility');

vl_setupnn();

% tracking-by-detection
p.grayscale_sequence = false; % suppose that sequence is colour
p.hog_cell_size = 4;
p.fixed_area = 200^2; % standard area to which we resize the target
p.n_bins = 2^5; % number of bins for the color histograms (bg and fg models)
p.learning_rate_pwp = 0.01; % bg and fg color models learning rate
p.inner_padding = 0.2; % defines inner area used to sample colors from the foreground
p.output_sigma_factor = 0.1; % standard deviation for the desired translation filter output
p.lambda = 1e-4; % regularization weight
p.learning_rate_cf = 0.01; % HOG model learning rate
p.period = 5;
p.update_thres = 0.7;
% scale related
p.hog_scale_cell_size = 4; % from DSST
p.learning_rate_scale = 0.025;
p.scale_sigma_factor = 1 / 2;
p.num_scales = 33;
p.scale_model_factor = 1.0;
p.scale_step = 1.03;
p.scale_model_max_area = 32 * 16;

% **********************************
% VOT: Get initialization data
% **********************************
[handle, image, region] = vot('polygon');

nv = numel(region);
assert(nv == 8 || nv == 4);
if (nv == 8)
    % polygon format
    [cx, cy, w, h] = getAxisAlignedBB(region);
else
    x = region(1);
    y = region(2);
    w = region(3);
    h = region(4);
    cx = x + w / 2;
    cy = y + h / 2;
end

p.init_pos = [cy, cx];
p.target_sz = round([h, w]);
[p, bg_area, fg_area, area_resize_factor] = initializeAllAreas(imread(image), p);
im = imread(image);

pos = p.init_pos;
target_sz = p.target_sz;
period = p.period;
weight_num = 0:period - 1;
weight = (1.1).^(weight_num);

% patch of the target + padding
patch_padded = getSubwindow(im, pos, p.norm_bg_area, bg_area);

% initialize hist model
new_pwp_model = true;
[bg_hist, fg_hist] = updateHistModel(new_pwp_model, patch_padded, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence);
new_pwp_model = false;

% Hann (cosine) window
hann_window_cosine = single(hann(p.cf_response_size(1))*hann(p.cf_response_size(2))');
output_sigma = sqrt(prod(p.norm_target_sz)) * p.output_sigma_factor / p.hog_cell_size;
y = gaussianResponse(p.cf_response_size, output_sigma);
yf = fft2(y);

% Scale adaptation initialization
% code from DSST
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
max_scale_factor = p.scale_step^floor(log(min([size(im, 1), size(im, 2)]./target_sz))/log(p.scale_step));

% The CNN layers Conv5-4, Conv4-4 in VGG-Net-19
indLayers = [37, 28];
numLayers = length(indLayers);
new_hf_num_deep = cell(1, 2);
hf_den_deep = cell(1, 2);
new_hf_den_deep = cell(1, 2);
hf_num_deep = cell(1, 2);

xtf_deep = cell(1, 2);
IDensemble = zeros(1, 7);

frame = 0;

while true

    frame = frame + 1;

    if frame > 1

        [handle, image] = handle.frame(handle);

        if isempty(image)
            break;
        end
        im = imread(image);

        % extract patch of size bg_area and resize to norm_bg_area
        im_patch_cf = getSubwindow(im, pos, p.norm_bg_area, bg_area);
        % color histogram
        [likelihood_map] = getColourMap(im_patch_cf, bg_hist, fg_hist, p.n_bins, p.grayscale_sequence);
        likelihood_map(isnan(likelihood_map)) = 0;
        likelihood_map = imResample(likelihood_map, p.cf_response_size);
        likelihood_map = (likelihood_map + min(likelihood_map(:))) / (max(likelihood_map(:)) + min(likelihood_map(:)));
        if (sum(likelihood_map(:)) / prod(p.cf_response_size) < 0.01), likelihood_map = 1; end
        likelihood_map = max(likelihood_map, 0.1);
        hann_window = hann_window_cosine .* likelihood_map;

        % compute feature map
        xt = getFeatureMap(im_patch_cf, p.cf_response_size, p.hog_cell_size);
        % apply Hann window
        xt_windowed = bsxfun(@times, hann_window, xt);
        % compute FFT
        xtf = fft2(xt_windowed);

        % Correlation between filter and test patch gives the response
        hf = bsxfun(@rdivide, hf_num, sum(hf_den, 3)+p.lambda);
        response_cfilter = ensure_real(ifft2(sum(conj(hf).*xtf, 3)));
        % Crop square search region (in feature pixels)
        response_cf = cropFilterResponse(response_cfilter, floor_odd(p.norm_delta_area/p.hog_cell_size));
        % Scale up to match center likelihood resolution
        responseHandLow = mexResize(response_cf, p.norm_delta_area, 'auto');

        xt_deep = getDeepFeatureMap(im_patch_cf, hann_window, indLayers);
        response_deep = cell(1, 2);
        for ii = 1:length(indLayers)
            xtf_deep{ii} = fft2(xt_deep{ii});
            hf_deep = bsxfun(@rdivide, hf_num_deep{ii}, sum(hf_den_deep{ii}, 3)+p.lambda);
            response_deep{ii} = ensure_real(ifft2(sum(conj(hf_deep).*xtf_deep{ii}, 3)));
        end

        % Crop square search region (in feature pixels)
        responseDeepHigh = cropFilterResponse(response_deep{1}, floor_odd(p.norm_delta_area/p.hog_cell_size));
        responseDeepHigh = mexResize(responseDeepHigh, p.norm_delta_area, 'auto');
        % Crop square search region (in feature pixels)
        responseDeepMiddle = cropFilterResponse(response_deep{2}, floor_odd(p.norm_delta_area/p.hog_cell_size));
        responseDeepMiddle = mexResize(responseDeepMiddle, p.norm_delta_area, 'auto');
        % Expert pool
        expert(1).response = responseHandLow;
        expert(2).response = responseDeepMiddle;
        expert(3).response = responseDeepHigh;
        expert(4).response = responseDeepMiddle + 0.5 * responseHandLow;
        expert(5).response = responseDeepHigh + 0.5 * responseHandLow;
        expert(6).response = responseDeepHigh + 0.5 * responseDeepMiddle;
        expert(7).response = responseDeepHigh + 0.5 * responseDeepMiddle + 0.02 * responseHandLow;

        % Per-expert prediction
        center = (1 + p.norm_delta_area) / 2;
        for i = 1:7
            [row, col] = find(expert(i).response == max(expert(i).response(:)), 1);
            expert(i).pos = pos + ([row, col] - center) / area_resize_factor;
            expert(i).rect_position(frame, :) = [expert(i).pos([2, 1]) - target_sz([2, 1]) / 2, target_sz([2, 1])];
            expert(i).center(frame, :) = [expert(i).rect_position(frame, 1) + (expert(i).rect_position(frame, 3) - 1) / 2, expert(i).rect_position(frame, 2) + (expert(i).rect_position(frame, 4) - 1) / 2];
            expert(i).smooth(frame) = sqrt(sum((expert(i).center(frame, :) - expert(i).center(frame-1, :)).^2));
            % Smoothness between two frames (self-evaluation score)
            expert(i).smoothScore(frame) = exp(-(expert(i).smooth(frame)).^2/(2 * p.avg_dim.^2));
        end

        % Calculate robustness score
        if frame > period - 1
            for i = 1:7
                expert(i).RobScore(frame) = RobustnessEva(expert, i, frame, period, weight, 7);
                % expert(i).RobScore(frame) = DisRobustness(expert, i, frame, weight, p);
                IDensemble(i) = expert(i).RobScore(frame);
            end
            meanScore(frame) = sum(IDensemble) / 7;
            [~, ID] = sort(IDensemble, 'descend');
            pos = expert(ID(1)).pos;
            Final_rect_position = expert(ID(1)).rect_position(frame, :);
        else
            for i = 1:7
                expert(i).RobScore(frame) = 1;
            end
            pos = expert(7).pos;
            Final_rect_position = expert(7).rect_position(frame, :);
        end

        % adaptive update
        Score1 = calculatePSR(response_cfilter); % hogScore
        Score2 = calculatePSR(response_deep{1});
        Score3 = calculatePSR(response_deep{2});
        PSRScore(frame) = (Score1 + Score2 + Score3) / 3;
        if frame > period - 1
            % calculate average score
            FinalScore = meanScore(frame) * PSRScore(frame);
            AveScore = sum(meanScore(period:frame).*PSRScore(period:frame)) / (frame - period + 1);
            if (FinalScore > p.update_thres * AveScore)
                p.learning_rate_pwp = 0.01;
                p.learning_rate_cf = 0.01;
            else
                p.learning_rate_pwp = 0; % we want pure color model, just discard unreliable samples
                p.learning_rate_cf = (FinalScore / (p.update_thres * AveScore))^3 * 0.01; % penalize the sample with low score
            end
        end

        % Scale space search
        im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size, hann_window_cosine);
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
        bg_area = round(target_sz+p.padding*avg_dim);
        if (bg_area(2) > size(im, 2)), bg_area(2) = size(im, 2) - 1; end
        if (bg_area(1) > size(im, 1)), bg_area(1) = size(im, 1) - 1; end

        bg_area = bg_area - mod(bg_area-target_sz, 2);
        fg_area = round(target_sz-avg_dim*p.inner_padding);
        fg_area = fg_area + mod(bg_area-fg_area, 2);
        % Compute the rectangle with (or close to) params.fixed_area and same aspect ratio as the target bboxgetScaleSubwindow
        area_resize_factor = sqrt(p.fixed_area/prod(bg_area));
    end

    % extract patch of size bg_area and resize to norm_bg_area
    im_patch_bg = getSubwindow(im, pos, p.norm_bg_area, bg_area);
    % compute feature map, of cf_response_size
    xt = getFeatureMap(im_patch_bg, p.cf_response_size, p.hog_cell_size);
    % apply Hann window
    xt = bsxfun(@times, hann_window_cosine, xt);
    % compute FFT
    xtf = fft2(xt);
    % Filter update
    % Compute expectations over circular shifts, therefore divide by number of pixels.
    new_hf_num = bsxfun(@times, conj(yf), xtf) / prod(p.cf_response_size);
    new_hf_den = (conj(xtf) .* xtf) / prod(p.cf_response_size);

    xt_deep = getDeepFeatureMap(im_patch_bg, hann_window_cosine, indLayers);
    for ii = 1:numLayers
        xtf_deep{ii} = fft2(xt_deep{ii});
        new_hf_num_deep{ii} = bsxfun(@times, conj(yf), xtf_deep{ii}) / prod(p.cf_response_size);
        new_hf_den_deep{ii} = (conj(xtf_deep{ii}) .* xtf_deep{ii}) / prod(p.cf_response_size);
    end

    if frame == 1
        % first frame, train with a single image
        hf_den = new_hf_den;
        hf_num = new_hf_num;
        for ii = 1:numLayers
            hf_den_deep{ii} = new_hf_den_deep{ii};
            hf_num_deep{ii} = new_hf_num_deep{ii};
        end
    else
        % subsequent frames, update the model by linear interpolation
        hf_den = (1 - p.learning_rate_cf) * hf_den + p.learning_rate_cf * new_hf_den;
        hf_num = (1 - p.learning_rate_cf) * hf_num + p.learning_rate_cf * new_hf_num;
        for ii = 1:numLayers
            hf_den_deep{ii} = (1 - p.learning_rate_cf) * hf_den_deep{ii} + p.learning_rate_cf * new_hf_den_deep{ii};
            hf_num_deep{ii} = (1 - p.learning_rate_cf) * hf_num_deep{ii} + p.learning_rate_cf * new_hf_num_deep{ii};
        end
        % BG/FG MODEL UPDATE   patch of the target + padding
        im_patch_color = getSubwindow(im, pos, p.norm_bg_area, bg_area*(1 - p.inner_padding));
        [bg_hist, fg_hist] = updateHistModel(new_pwp_model, im_patch_color, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence, bg_hist, fg_hist, p.learning_rate_pwp);
    end

    % Scale update
    im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size, hann_window_cosine);
    xsf = fft(im_patch_scale, [], 2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf.*conj(xsf), 1);
    if frame == 1
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - p.learning_rate_scale) * sf_den + p.learning_rate_scale * new_sf_den;
        sf_num = (1 - p.learning_rate_scale) * sf_num + p.learning_rate_scale * new_sf_num;
    end

    % update bbox position
    if (frame == 1)
        Final_rect_position = [pos([2, 1]) - target_sz([2, 1]) / 2, target_sz([2, 1])];
        for i = 1:7
            expert(i).rect_position(1, :) = [pos([2, 1]) - target_sz([2, 1]) / 2, target_sz([2, 1])];
            expert(i).center(1, :) = [Final_rect_position(:, 1) + (Final_rect_position(:, 3) - 1) / 2, Final_rect_position(:, 2) + (Final_rect_position(:, 4) - 1) / 2];
            expert(i).RobScore(1) = 1;
            expert(i).smooth(1) = 0;
            expert(i).smoothScore(1) = 1;
        end
    end

    if frame > 1
        % **********************************
        % VOT: Report position for frame
        % **********************************
        handle = handle.report(handle, Final_rect_position);
    end

end

% **********************************
% VOT: Output the results
% **********************************
handle.quit(handle);

end

% We want odd regions so that the central pixel can be exact
function y = floor_odd(x)
y = 2 * floor((x - 1)/2) + 1;
end

function y = ensure_real(x)
assert(norm(imag(x(:))) <= 1e-5*norm(real(x(:))));
y = real(x);
end

function PSR = calculatePSR(response_cf)
cf_max = max(response_cf(:));
cf_average = sum(response_cf(:)) / (size(response_cf, 1) * size(response_cf, 2));
cf_sigma = sqrt(sum(sum((response_cf - cf_average).^2))/(size(response_cf, 1) * size(response_cf, 2)));
PSR = (cf_max - cf_average) / cf_sigma;
end
