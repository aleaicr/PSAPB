function response = genToPhys(q, psi_xvals)
% Generates a matrix with the response in physical coordinates for every location (x_vals) defined through the modal shapes (psi_xvals) of every location
% Where y_bridge = [y(x1) y(x2) y(x3) ... y(xn)] (matrix = vector of columns of every location)
% Where each column is a row vector: y(xi) = [y(xi(t1)); y(xi(t2)); ... y(xi(tf))]
%
% INPUTS
% psi_xvals:    Modal shapes (n_moods x length(x_vals))
% q:            Response in generalized coordinates (t_length x n_modos)
%
% OUTPUT
% response: Response in physical coordinates (t_length x length(x_vals))

response = permute(q,[3 1 2])*psi_xvals;                                    % q (t x n); psixvals (n x Xn); response (t x Xn)

end

