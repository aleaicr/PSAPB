function [val] = valueFixing(val,mu,sigma,maxval,minval)
% It is possible that some parameters that have been chosen with normal
% distribution are not in the range of the realistic possible values. This
% function corrects this problem by generating a new value with the normal
% distribution until it is in the range of the possible values.
%
% INPUTS
% val:                  double, Value to be corrected
% mu:                   double, mean of the normal distribution
% sigma:                double, standard deviation of the normal distribution
% maxval:               double, maximum possible value
% minval:               double, minimum possible value
%
% OUTPUTS
% val:                  Valor corregido                          

% While the value is not in the range of the possible values, generate a
% new value with the same normal distribution.
while or(val < minval, val > maxval)
    val = normrnd(mu,sigma);   
end
end