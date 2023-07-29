function [modelo,As,Bs,Cs,Ds] = SpaceState_bridge(n_modos,Me,Ke,Ce,Ge,alternative)
% Author: Alexis Contreras R.
% Returns the model of the dynamic system in space state representation.
% As the equivalent EOM with generalized coordinates is used, the representation
% will use generalized coordinates states.
%
%%
% Inputs
% n_modos           -> Cantidad de modos que se consideran
% Me                -> Matriz de masa (asociada a los modos)
% Ke                -> Matriz de rigidez
% Ce                -> Matriz de amortiguamiento
% Ge                -> Matriz (o vector) de participación de cada carga
% Alternative       -> Defines the response vector states, 
%                       'd' is displacement only
%                       'v' is velocity only
%                       'a' is acceleration only
%                       'dv' is displacement and velocity
%                       'da' is displacement and acceleration
%                       'va' is velocity and acceleration
%                       'dva' is displacement, velocity and acceleration
%
% Outputs
% modelo            -> Model of the dynamic system in space state representation
% As                -> State-Space Matrix A
% Bs                -> State-Space Matrix B
% Cs                -> State-Space Matrix C
% Ds                -> State-Space Matrix D

% State equation matrices
% x' = As*x + Bs*p(t)
As = [zeros(n_modos) eye(n_modos); -Me\Ke -Me\Ce];                      % State matrix
Bs = [zeros(n_modos); Me\Ge];                                           % Input matrix

% Response equation matrices
% y = Cs*x + Ds*p(t)                                                    % Cs is the output matrix and Ds is the feedthrough matrix
alternative = sort(alternative);
if isequal(alternative,'d')
    Cs = [eye(n_modos) zeros(n_modos)];
    Ds = zeros(n_modos);
elseif isequal(alternative,'v')
    Cs = [zeros(n_modos) eye(n_modos)];
    Ds = zeros(n_modos);
elseif isequal(alternative,'a')
    Cs = [-Me\Ke -Me\Ce];
    Ds = Me\Ge;
elseif isequal(alternative,'dv')
    Cs = [eye(n_modos) zeros(n_modos); zeros(n_modos) eye(n_modos)];
    Ds = [zeros(n_modos); zeros(n_modos)];
elseif isequal(alternative,'ad')
    Cs = [eye(n_modos) zeros(n_modos); -Me\Ke -Me\Ce];
    Ds = [zeros(n_modos); Me\Ge];
elseif isequal(alternative,'av')
    Cs = [zeros(n_modos) eye(n_modos); -Me\Ke -Me\Ce];
    Ds = [zeros(n_modos); Me\Ge];
elseif isequal(alternative,'adv')
    Cs = [eye(n_modos) zeros(n_modos); zeros(n_modos) eye(n_modos); -Me\Ke -Me\Ce];
    Ds = [zeros(n_modos); zeros(n_modos); Me\Ge];
end

modelo = ss(As,Bs,Cs,Ds);                                       % State-Space model, have all the information of the dynamic system
end

