% Copyright 2012 The CARFAC Authors. All Rights Reserved.
% Author: Richard F. Lyon
%
% This file is part of an implementation of Lyon's cochlear model:
% "Cascade of Asymmetric Resonators with Fast-Acting Compression"
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

%% Test/demo hacking for CARFAC Matlab stuff:

% Mangled by Axel

function smooth_nap = CARFAC_hacking_axel(filename) %used to be filename

  dB = -40;  % -60:20:40

  file_signal = wavread(filename);
  file_signal = file_signal(:, 1);  % Mono test only.
  test_signal = file_signal * 10^(dB/20);

  n_ears = 1;
  
  CF_struct = CARFAC_Design(n_ears);  % default design
  CF_struct = CARFAC_Init(CF_struct);
  
  [CF_struct, nap_decim, nap] = CARFAC_Run(CF_struct, test_signal);
  
  smoothed = filter(1, [1, -0.995], nap(:, :, :));
  
  % only ear 1:
  ear = 1;
  smoothed = max(0, smoothed(50:50:end, :, 1));
  %MultiScaleSmooth(smoothed.^0.5, 1);
  
  %figure(1)
  %starti = 0;  % Adjust if you want to plot a later part.
  %imagesc(nap(starti+(1:15000), :)');
  
  smooth_nap = nap_decim(:, :, ear);
  mono_max = max(smooth_nap(:));
  %figure(3 + ear + n_ears)  % Makes figures 5, 6, 7.
  smooth_nap = ((max(0, smooth_nap)/mono_max)' .^ 0.5); % used to be 63*
  %image(smooth_nap)
  %title('smooth nap from nap decim')
  %colormap(1 - gray);

  CF_struct
  CF_struct.ears(1).CAR_state
  CF_struct.ears(1).AGC_state
  min_max_decim = [min(nap_decim(:)), max(nap_decim(:))]

  % Expected result:  Figure 3 looks like figure 2, a tiny bit darker.
  % and figure 4 is empty (all zero)
