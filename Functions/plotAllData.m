%% Plot all Data

%% Inicializar
clear variables
close all
clc

%% Inputs
% Folder y Result a analizar
fileFolder = 'ResultsNewMod3\m1_w3_z1';
fileName = 'yppN.txt';

% Distribución asumida de los resultados de la franja
distribution = 'lognormal';   % 'norm' o 'lognorm'                    % P(Delta>delta | N = n)  --> distribución del histograma por franja
alternative = 'pdf';

% Cantidad de peatones en simulaciones (total)
np = 200;                                   

% Número de Simulaciones a considerar para plot
nsims = 'all';                % Número de simulaciones o 'all'                % Cantidad de simulaciones a graficar

%% getStripe & figure
median_ln = zeros(200,1);
sigma = zeros(200,1);
figure
hold on
for stripe = 1:200
    disp(stripe)
    resultsForStripe = getStripe(fileFolder,fileName,stripe);
    resultsForStripe = resultsForStripe(2:end,1);                           % El primero es un NaN
    if isequal(nsims,'all')
        % No hace nada, tenerlas todas
        nsims = length(resultsForStripe);
    else
        resultsForStripe = resultsForStripe(1:nsims,1);
    end
    plot(stripe*ones(length(nsims),1),resultsForStripe,'o','color','#606060','linewidth',1.3)
    median_ln(stripe,1) = geomean(resultsForStripe);
    sigma(stripe,1) = std(log(resultsForStripe));
end
plot((1:1:200).',median_ln,'k','linewidth',2)
plot((1:1:200).',exp(log(median_ln)+sigma),'.-','color','k','linewidth',2)
plot((1:1:200).',exp(log(median_ln)-sigma),'.-','color','k','LineWidth',2)
hold off
xlabel('N = n')
ylabel('$\Delta = \ddot{y} (N = n)[m/s2]$','interpreter','latex')
legend(['nsims = ' convertStringsToChars(string(nsims)) '; np = ' convertStringsToChars(string(np))])
grid on
xlim([0 200])
% ylim([0 0.6])
% ylim([0 0.5])
set(gcf,'Position',[100,100,800,600])
set(gcf,'Color','w')
set(gca,'FontSize',14)