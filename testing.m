% This script is to test the new method of clustering the pedestrians

%% Init
clear variables
close all
clc

%% Inputs
% Groups parameters
nGroups = 200;
npgDistr = [0 0; 1 0.2; 2 0.3; 4 0.2; 5 0.15; 6 0.1; 7 0.025; 8 0.025]; % Number of pedestrians per group distribution
% Check if npgDistr sums 1
if sum(npgDistr(:,2)) ~= 1
    error('The sum of the second column of npgDistr must be 1')
end

% Pedestrian mass normal distribution, year 1999 (Johnson et al 2008)
mu_m = 71.91;       % kg                                                    % Mean
sigma_m = 14.89;    % kg                                                    % Standar deviation
m_min = 40;         % kg                                                    % Minimum mass
m_max = 150;        % kg                                                    % Maximum mass

% Pedestrian Walking speed normal distribution, year 2005 (Pachi & Ji, 2005)
mu_v = 1.3;         % m/s                                                   % Mean
sigma_v = 0.13;     % m/s                                                   % Standard deviation
v_min = 0.1;        % m/s                                                   % Minimum walking speed (before correlation)
v_max = 10;         % m/s                                                   % Maximum walking speed (before correlation)

% Lateral gait frequency normal distribution, year 2005 (Pachi & Ji 2005)
mu_freq = 1.8;      % hz                                                    % Mean                                                       
sigma_freq = 0.11;  % hz                                                    % Standard deviation
freq_min = 1.2;     % hz                                                    % Minimum frequency
freq_max = 2.4;     % hz                                                    % Maximum frequency

% Waling speed vs frequency correlation (Pachi & Ji 2005)
rhofv = 0.51;                                                            

% a (van der pol oscillator)
mu_ai = 1.5;                                                                % Mean
ai_min = 1;                                                                 % Minimum
ai_max = 2*mu_ai - ai_min;                                                  % Maximum
sigma_ai = (ai_max - mu_ai)/3;                                              % Standard deviation

% lambda (vdp oscillator)
mu_lambdai = 10;                                                            % Mean
lambdai_min = 8;                                                            % Minimum
lambdai_max = 2*mu_lambdai - lambdai_min;                                   % Maximum
sigma_lambdai = (lambdai_max - mu_lambdai)/3;                               % Standard deviation

% b (frequency adapting parameter)
b = 2*pi;                                                                   % Constant

% Walking direction
p_dir = 0.5; % probability that the group enters to the bridge by the left side

%% Save data in structs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pedParams
pedParams = struct();
pedParams.mass = [mu_m; sigma_m; m_min; m_max];
pedParams.speed = [mu_v; sigma_v; v_min; v_max];
pedParams.freq = [mu_freq; sigma_freq; freq_min; freq_max];
pedParams.rhofv = rhofv;
pedParams.ai = [mu_ai; sigma_ai; ai_min; ai_max];
pedParams.lambdai = [mu_lambdai; sigma_lambdai; lambdai_min; lambdai_max];
pedParams.b = b;
pedParams.p_dir = p_dir;

%% groupParams
groupParams = struct();
groupParams.nGroups = nGroups;
groupParams.npgDistr = npgDistr;

%% Create Parameters
[pRealizations, gRealizations] = createParameters(pedParams, groupParams);

%% Create histograms
% Pedestrian mass
figure
histogram(pRealizations.mass, 50, 'Normalization', 'pdf')

% Pedestrian speed
figure
histogram(pRealizations.speed, 50, 'Normalization', 'pdf')

% Pedestrian frequency
figure
histogram(pRealizations.freq, 50, 'Normalization', 'pdf')

% Pedestrian ai
figure
histogram(pRealizations.ai, 50, 'Normalization', 'pdf')

% Pedestrian lambdai
figure
histogram(pRealizations.lambdai, 50, 'Normalization', 'pdf')

% Pedestrian direction
figure
histogram(pRealizations.dir, 50, 'Normalization', 'pdf')

% Group size
figure
histogram(gRealizations.npg, 50, 'Normalization', 'pdf')

% Group speed
figure
histogram(gRealizations.speed, 50, 'Normalization', 'pdf')
