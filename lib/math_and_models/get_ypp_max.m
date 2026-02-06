function [y_max] = get_ypp_max(y_t)
% This function obtains the maximum value of the lateral movement of the bridge
%

% Obtain the maximum value in the matrix
y_max = max(max(abs(y_t)));

end