function [response,max_response] = genToPhys_acc(q, psi_xvals)

% Generar matriz con todos los vectores de la respuesta en todo el
% tiempo para cada posición x_vals de la viga equivalente, es decir:

% y_bridge = [ y(x1) y(x2) y(x3) ... y(xi)] (matriz = vector fila de vectores columna
% donde: y(x1) = [y(x1(t1)); y(x1(t2)); ... y(x1(tf))] (vector)
% notar que y, puede ser yp o ypp si quiero velocidad o aceleración

% INPUTS
% psi_xvals:    Matriz con todos los valores de psi evaluados en todos los xvals para todos los modos (n_moods x length(x_vals))
% q:            Matriz con la respuesta de desplazamiento/velocidad/aceleracion en coordenadas generalizadas para todos los modos (t_length x n_modos)

% OUTPUT
% response: respuesta en coordenadas físicas (tiempo,posición)
% max_response: máxima respuesta en el puente para cada tiempo (tiempo,1)

% COMENTARIOS
% Notar que q, puede ser qp o qpp si se quiere velocidad o aceleración
% respectivamente, ya que esta función compone desde coordenadas
% generalizadas a coordenadas físicas.
% creo que no funcionaba porq en acc no hay demux ??
%%
response = permute(q,[3 1 2])*psi_xvals;                                    % q (t x n); psixvals (n x Xn); response (t x Xn)
max_response = max(abs(response),[],2);                                     % max_response (tx1)  --> máximo de abs(response)

end

