
function [response, max_response] = genToPhys_acc(q, psi_xvals)
% Convert acceleration from a generalized coordinates vector to a physical coordinates vector.
% This generates a matrix with all response vectors over time for each position x_vals of the equivalent beam, i.e.:
%
% y_bridge = [y(x1) y(x2) y(x3) ... y(xN)] (matrix)
% where each y(xi) = [y(xi(t1)); y(xi(t2)); ... y(xi(tf))] (vector)
% where each tj is a time step of the simulation and tf is the final time
% Note that y_bridge is a matrix, this is a row vector where each element is a column vectors)
%
% INPUTS
% psi_xvals:    Matrix with all values of psi evaluated at all xvals for all modes (n_modos x length(x_vals))
% q:            Matrix with the displacement/velocity/acceleration response in generalized coordinates for all modes (t_length x n_modos)
%
% OUTPUT
% response: response in physical coordinates (time,position)
% max_response: maximum response on the bridge for each time (time,1)
%
% NOTES:
% Note that q can be qp or qpp if velocity or acceleration is desired,
% respectively, since this function composes from generalized coordinates
% to physical coordinates.
% I think it didn't work because there is no demux in acc ??
%%

response = permute(q,[3 1 2])*psi_xvals; % q (t x n); psixvals (n x Xn); response (t x Xn)
max_response = max(abs(response),[],2);  % max_response (tx1)  --> maximum of abs(response)

end


