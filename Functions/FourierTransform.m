function [f_vect,data_fft,data_fft_nosq] = FourierTransform(data,f)
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
data_fft_nosq = fftshift(abs(fft(data)));                   % Transformada de fourier
data_fft = fftshift(fft(data).^2);

end

