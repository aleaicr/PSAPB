%% Fragility Curves
% EDITING
%
%
%
%
% COMENTARIOS:
% * Ya se verificó que distribuye como lognormal
% * Ajustar el código para la cantidad de puentes a observar, este tiene 3


%% Inicializar
clear variables
close all
clc

%% Inputs
% Files to plot
fileFolder1 = 'Results\m1_w1_z1';      % Recomendable no cambiar folders
fileFolder2 = 'Results\m1_w1_z2';
fileFolder3 = 'Results\m1_w1_z3';
fileFolder4 = 'Results\m1_w2_z1';
fileFolder5 = 'Results\m1_w3_z1';
fileName = 'yppN.txt';
forHistFolder = fileFolder1;               % Folder to show on histogram

% Parámetro para comparar
parameterToCompare = 'frequency';             % 'frequency' o 'damping'

% Histfit Cehck  
stripe = 200;                                                               % Franja para observar histograma
nBins = 20;                                                                 % N° de cajas para el histograma

% delta_max
delta = 0.1;                                                               % delta_max para la curva de fragilidad para todos los puentes

% Distribución asumida de los resultados de la franja
distribution = 'lognormal';   % 'norm' o 'lognorm'                          % P(Delta>delta_{max} | N = n)  --> distribución del histograma por franja
alternative = 'cdf';                                                        % En histograma normalizar por pdf o por cdf

% Número de peatones en simulaciones
np = 200;

%% getStripe figure
% resultsForStripe = getStripe(forHistFolder,fileName,stripe);
% nsims = length(resultsForStripe);
% figure
% plot(np*ones(length(nsims),1),resultsForStripe,'o','color','#606060')
% xlabel('N = n')
% ylabel('$\Delta = \ddot{y} (N = n)[m/s2]$','interpreter','latex')
% legend(['nsims = ' convertStringsToChars(string(nsims)) '; np = ' convertStringsToChars(string(np))])
% title('Aceleración máxima en puente para una franja')
% grid on

%% Observe histogram & selected fit
% Observar histograma con ajuste seleccionado
% distParams = getStripeParams(forHistFolder,fileName,stripe,nBins,distribution,alternative);

%% get Fragility Curves
% Init Vectors
probs = zeros(np,1);
probs2 = zeros(np,1);
probs3 = zeros(np,1);

fprintf('comparando %s\n',parameterToCompare)
% get Probabilities
for stripe = 1:np
    fprintf('stripe =  %.0f\n',stripe)
    if isequal(parameterToCompare,'damping')
        % for Folder1
        % get Stripe
        distParams = getStripeParams(fileFolder1,fileName,stripe,nBins,distribution,'');
        % get Probs
        probs(stripe,1) = getProbs(delta,distParams,distribution);              % Cambiar norm_params o lognorm_params si se quiere distribución normal o lognormal respectivamente
        
        %for Folder 2
        % get Stripe
        distParams = getStripeParams(fileFolder2,fileName,stripe,nBins,distribution,'');
        % get Probs
        probs2(stripe,1) = getProbs(delta,distParams,distribution);             % Cambiar norm_params o lognorm_params si se quiere distribución normal o lognormal respectivamente
        
        %for Folder 3
        % get Stripe
        distParams = getStripeParams(fileFolder3,fileName,stripe,nBins,distribution,'');
        % get Probs
        probs3(stripe,1) = getProbs(delta,distParams,distribution);             % Cambiar norm_params o lognorm_params si se quiere distribución normal o lognormal respectivamente
    elseif isequal(parameterToCompare,'frequency')
        % for Folder1
        % get Stripe
        distParams = getStripeParams(fileFolder1,fileName,stripe,nBins,distribution,'');
        % get Probs
        probs(stripe,1) = getProbs(delta,distParams,distribution);              % Cambiar norm_params o lognorm_params si se quiere distribución normal o lognormal respectivamente
        
        %for Folder 2
        % get Stripe
        distParams = getStripeParams(fileFolder4,fileName,stripe,nBins,distribution,'');
        % get Probs
        probs2(stripe,1) = getProbs(delta,distParams,distribution);             % Cambiar norm_params o lognorm_params si se quiere distribución normal o lognormal respectivamente
        
        %for Folder 3
        % get Stripe
        distParams = getStripeParams(fileFolder5,fileName,stripe,nBins,distribution,'');
        % get Probs
        probs3(stripe,1) = getProbs(delta,distParams,distribution);             % Cambiar norm_params o lognorm_params si se quiere distribución normal o lognormal respectivamente
    else
        error('escribir parameterToCompare como "damping" o "frequency" en char')
    end
end
% Fragility Curva Figure
figure
plot((1:1:np).',probs,'o')
hold on
plot((1:1:np).',probs2,'o')
plot((1:1:np).',probs3,'o')
hold off
xlabel('N = n')
ylabel('P(\Delta > \delta_{max} | N = n)')
if isequal(fileName,'yN.txt')
    title('Estimaciones de probabilidades',['\delta_{max} = ' convertStringsToChars(string(delta)) '[m]'])
elseif isequal(fileName,'yppN.txt')
    title('Estimaciones de probabilidades',['\delta_{max} = ' convertStringsToChars(string(delta)) '[m/s2]'])
end
grid on
if isequal(parameterToCompare,'damping')
    legend('P1: \zeta_n = 0.7%','P2: \zeta_1 = 2%','P3: \zeta_3 = 5%')
elseif isequal(parameterToCompare,'frequency')
    legend('P1: f_1 = 0.5 [Hz]','P4: f_1 = 0.9 [Hz]','P5: f_1 = 1.3 [Hz]')
else
    error('escribir parameterToCompare como "damping" o "frequency" en char')
end
