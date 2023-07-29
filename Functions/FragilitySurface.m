%% Fragility Surface
% Generar una superficie de fragilidad con las probabilidades de
% excedencias dadas las cantidades de peatones para un barrido de
% criterios de servicio
%
% COMENTARIOS:
% * Ya se verificó que distribuye como lognormal
% * 
%% Inicializar
clear variables
close all
clc

%% Inputs
% Folder y Result a analizar
fileFolder = 'Results\m1_w1_z1';
fileName = 'yppN.txt';

% Histfit Check 
stripe = 200;
nBins = 20;

% Definir el barrido de deltas para la superficie                           % Siempre hay que tener una de las dos comentadas
% Desplazamiento
% delta_min = 0.001;
% delta_step = 0.001;
% delta_max = 0.01;
% deltas = delta_min:delta_step:delta_max;                                    % Vector del barrido de deltas a calcular

% Aceleración
delta_min = 0.02;
delta_step = 0.02;
delta_max = 0.25;
deltas = delta_min:delta_step:delta_max;

% Distribución asumida de los resultados de la franja
distribution = 'lognormal';   % 'norm' o 'lognorm'                    % P(Delta>delta | N = n)  --> distribución del histograma por franja
f = 'pdf';

%% Previous
% get np y nsims
% [np,nsims] = readReadme(fileFolder)
%%%%% Temporalmente, como ya se saben, solo se pondran
nsims = 480;
np = stripe;

%% getStripeParams
disParams = getStripeParams(fileFolder,fileName,stripe,nBins,distribution,f);

%% get FragCurve
% Inicializar vectores
probs = zeros(np,length(deltas)); 

% Obtener probabilidades
for index = 1:length(deltas)
    delta_val = deltas(index);
    for stripe = 1:np
        % get Stripe
        disParams = getStripeParams(fileFolder,fileName,stripe,nBins,distribution,'');
    
        % get Probs
        probs(stripe,index) = getProbs(delta_val,disParams,distribution);                % Cambiar norm_params o lognorm_params si se quiere distribución normal o lognormal respectivamente
    end
end

%% White surface
figure
surf((1:1:np),deltas,probs.','FaceColor','white')
xlabel('N = n')
if isequal(fileName,'yN.txt')
    ylabel('\delta_{max} [m]')
elseif isequal(fileName,'yppN.txt')
    ylabel('\delta_{max} [m/s2]')
end
zlabel('P(\Delta > \delta_{max} | N = n)')
title('Superficie de fragiilidad')
hold on

%% White surface and colored contour3
figure
surf((1:1:np),deltas,probs.','FaceColor','white')
xlabel('N = n')
if isequal(fileName,'yN.txt')
    ylabel('\delta_{max} [m]')
elseif isequal(fileName,'yppN.txt')
    ylabel('\delta_{max} [m/s2]')
end
zlabel('P(\Delta > \delta_{max} | N = n)')
title('Superficie de fragiilidad')
hold on
[~,h] = contour3((1:1:np),deltas,probs.');
h.LineWidth = 3;
colorbar

%% Contour
figure
[C,h] = contour((1:1:np),deltas,probs.');
h.LineWidth = 2;
xlabel('N = n')
if isequal(fileName,'yN.txt')
    ylabel('\delta_{max} [m]')
elseif isequal(fileName,'yppN.txt')
    ylabel('\delta_{max} [m/s^2]')
end
grid on
colorbar

