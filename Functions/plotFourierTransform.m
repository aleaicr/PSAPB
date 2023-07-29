function [f_vect,data_fft] = plotFourierTransform(data,f)
%% Esta función grafica la transformada de fourier de la señal
% Alexis Contreras R.
% INPTUS
% data:         double vector, datos de la señal
% f:            double, frecuencia (constante) de muestreo de la señal
% 
% OUTPUTS
% -:            gráfico de la transformada de fourier 
%
% Comentarios
% *

% Previous calculations
N = length(data);                       % Número de muestras

% Vector de frecuencias
f_vect = (-N/2:(N/2-1))*(f/N);         % Vector de frecuencias

% Transformada de fourier
data_fft = fft(data);                   % Transformada de fourier
data_fft = fftshift(abs(data_fft).^2);

% Gráfico
figure
plot(f_vect,fftshift(abs(data_fft).^2))
xlabel('Frecuencia [Hz]')
ylabel('Amplitud')
title('Transformada de fourier')
grid on
end

