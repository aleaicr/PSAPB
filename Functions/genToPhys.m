function [response,max_response] = genToPhys(q, psi_xvals)
% Author: Alexis Contreras R.
% Generate matrix with all the response vectors over time for each x_vals of the equivalent beam, i.e.:
% y_bridge = [ y(x1) y(x2) y(x3) ... y(xi)] (matrix = row vector of column vectors
% where: y(x1) = [y(x1(t1)); y(x1(t2)); ... y(x1(tf))] (vector)
% note that y can be yp or ypp
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

%%
response = q*psi_xvals;                                                     % q (t x n); psixvals (n x Xn); response (t x Xn)
max_response = max(abs(response),[],2);                                     % max_response (tx1)  --> máximo de abs(response)
end

