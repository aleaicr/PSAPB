%% Fragility Curves
% EDITING
%
%
%
%
% COMENTARIOS:
% * Ya se verificó que distribuye como lognormal
%

%% Inicializar
clear variables
close all
clc

%% Inputs
% Folder y Result a analizar
fileFolder = 'Results\m1_w1_z1';
fileName = 'yppN.txt';

% Histfit Check  
stripe = 150;           % Single stripe to plot too
nBins = 30;             % Number of bins for histogram

% delta_max
delta = 0.1;

% Cantidad de simulaciones para estimar las curvas
nSims = 'all';                                  % usar número entero (ej: 1000), o usar 'all' para todas las simulaciones que estén en el archivo

% Distribución asumida de los resultados de la franja
distribution = 'lognormal';   % 'normal' o 'lognormal'                      % P(Delta>delta | N = n)  --> distribución del histograma por franja
alternative = 'pdf';

% Colores para plot
color1 = '#0072BD';
color2 = '#D95319';
color3 = '#EDB120';

%% Previous
% get np y nsims
% [np,nsims] = readReadme(fileFolder)
%%%%% Temporalmente, como ya se saben, solo se pondran
% nsims = 20;
np = 200;

%% getStripe figure  ### FOR SINGLE STRIPE
resultsForStripe = getStripe(fileFolder,fileName,stripe);
nsims = length(resultsForStripe);
figure
plot(stripe*ones(length(nsims),1),resultsForStripe,'o','color','#606060')
xlabel('N = n')
ylabel('$\Delta = \ddot{y} (N = n)[m/s^2]$','interpreter','latex')
legend(['nsims = ' convertStringsToChars(string(nsims)) '; np = ' convertStringsToChars(string(np))])
title('Aceleración máxima en puente para una franja')
grid on
xlim([0 200])

%% getStripeParams and figure
disParams = getStripeParams(fileFolder,fileName,stripe,nBins,distribution,alternative);
xlabel('\delta(N = n)')

%% get FragCurve
% Inicializar vectores
probs = zeros(np,1); 

% Obtener probabilidades
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
    title('Estimaciones de probabilidades')
    legend(['nsims = ' convertStringsToChars(string(nSims)) ' | \delta_{max} = ' convertStringsToChars(string(delta)) ' [m])'])
elseif isequal(fileName,'yppN.txt')
    title('Estimaciones de probabilidades')
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

