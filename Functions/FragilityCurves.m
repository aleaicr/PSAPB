%% Fragility Curves
% Author: Alexis Contreras R.
% This script generates the fragility curves for the probability analysis
% The first plot is a single stripe of data (n simulations for a single pedestrian quantity)
%
% Notes:
% This script should be executed after executing the script in main.m that creates the "results" .txt files


%% Init
clear variables
close all
clc

%% Inputs
% File direction
fileFolder = 'Results\m1_w1_z1';
fileName = 'yppN.txt';                  % yppN.txt have the accelerations

% Maximum pedestrian quantity
np = 200;

% Inputs to check the distribution of a single stripe
stripe = 150;           % Pedestrian quantity of the stripe
nBins = 30;             % Number of bins in histogram

% Serviceability criteria (delta)
delta = 0.1; % m/s2           % P(Delta >= delta | N = n)

% As we're working with yppN.txt, the serviceability criteria is an acceleration.
% In the case that we work with ypN.txt or yN.txt the criteria should be in m/s or m respectively

% Number of simulations to estimate the curves
nSims = 'all';             % integer number or 'all', this allow to estimate f.c. with less simulations than the number of sims in results

% Assumed distribution to plot in histogram
distribution = 'lognormal';   % 'normal' or 'lognormal'    % P(Delta>delta | N = n)  --> distribución del histograma por franja
alternative = 'pdf';

% Colors in plot
color1 = '#0072BD';
color2 = '#D95319';
color3 = '#EDB120';

%% get stripe figure
resultsForStripe = getStripe(fileFolder,fileName,stripe);
nsims = length(resultsForStripe);
figure
plot(stripe*ones(length(nsims),1),resultsForStripe,'o','color','#606060')
xlabel('N = n')
ylabel('$\Delta = \ddot{y} (N = n)[m/s^2]$','interpreter','latex')
legend(['nsims = ' convertStringsToChars(string(nsims)) '; np = ' convertStringsToChars(string(np))])
title('Maximum bridge acceleration for a single stripe')
grid on
xlim([0 200])

%% getStripeParams and figure
disParams = getStripeParams(fileFolder,fileName,stripe,nBins,distribution,alternative);
xlabel('\delta(N = n)')

%% get FragCurve
% Init vector
probs = zeros(np,1); 

% Get probabilities
for stripe_i = 1:np
    % get Stripe
    [disParams,nSims] = getStripeParams_2(fileFolder,fileName,stripe_i,nSims,nBins,distribution,'');

    % get Probs
    probs(stripe_i,1) = getProbs(delta,disParams,distribution);                % Cambiar norm_params o lognorm_params si se quiere distribución normal o lognormal respectivamente
end

colorToUse = color1;
% Figure with probs
figure
plot((1:1:np).',probs,'o','color',colorToUse)                        % Color2: #D95319  , Color3
xlabel('N = n')
ylabel('P(\Delta > \delta_{max} | N = n)')
if isequal(fileName,'yN.txt')
    title('Probability estimations')
    legend(['nsims = ' convertStringsToChars(string(nSims)) ' | \delta_{max} = ' convertStringsToChars(string(delta)) ' [m])'])
elseif isequal(fileName,'yppN.txt')
    title('Probability estimations')
    legend(['nsims = ' convertStringsToChars(string(nSims)) ' | \delta_{max} = ' convertStringsToChars(string(delta)) ' [m/s^2])'])
end
grid on
ylim([0 1])
xlim([0 200])


figure
plot(stripe,probs(stripe),'o','color',colorToUse,'linewidth',2)                        % Color2: #D95319  , Color3
xlabel('N = n')
ylabel('P(\Delta > \delta_{max} | N = n)')
if isequal(fileName,'yN.txt')
    title('Estimaciones de probabilidades')
    legend(['nsims = ' convertStringsToChars(string(nSims)) ' | \delta_{max} = ' convertStringsToChars(string(delta)) ' [m])'])
elseif isequal(fileName,'yppN.txt')
    title('Estimaciones de probabilidades')
    legend(['nsims = ' convertStringsToChars(string(nSims)) ' | \delta_{max} = ' convertStringsToChars(string(delta)) ' [m/s^2])'])
end
grid on
ylim([0 1])
xlim([0 200])

