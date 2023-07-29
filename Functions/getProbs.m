function[Pexc] = getProbs(delta,params,distribution)
% Obtener la probabilidad de excedencia de cierto criterio de servicio
% delta dada la cantidad de peatones
%
% INPUTS
%
%
% OUTPUTS
% P: Probabil
% 
% 
% COMENTARIOS
% 
% 
% 

if isequal(distribution,'normal')
    Pexc = 1 - normcdf(delta,params.mu,params.sigma);

elseif isequal(distribution,'lognormal')
    Pexc = 1 - logncdf(delta,params.mu,params.sigma);
end

end