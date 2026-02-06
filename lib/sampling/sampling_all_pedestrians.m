function [mi,vi,fi,wi,ai,lambdai,Ai,k1i,k2i,k3i,alpha_si,bi,side,ti,v0,v0p,loci] = sampling_all_pedestrians(pedParams,n_samples)
% Creates samples of all the properties of pedestrian properties
%
% 
%
% Previous was % function [mi,vi,fi,wi,ai,lambdai,Ai,k1i,k2i,k3i,alpha_si,bi,side,ti,tac,ppt,v0,v0p,loci] = sampling_all_pedestrians(simParams,pedParams,n_samples)


% Mass
mi = sampling_distributions('normal', n_samples, 'mu_normal', pedParams.mu_m, 'sigma_normal', pedParams.sigma_m, 'min_value', pedParams.m_min, 'max_value', pedParams.m_max);

% Walking speed and vertical gait frequency.
samples = sampling_distributions('multivariatenormal', n_samples, 'mu_multi', pedParams.mu_fv, 'covariance_multi', pedParams.covariance_fv, 'min_value', pedParams.min_fv, 'max_value', pedParams.max_fv);
fi = samples(:,1); % this is vertical gait frequency, to obtain the lateral gait frequency: f_lateral = f_vertical/2
vi = samples(:,2); 
wi = pi*fi; % this is lateral gait circular frequency (for the VdP oscillator)

% Van der Pol oscilator properties
ai = sampling_distributions('normal', n_samples, 'mu_normal', pedParams.mu_ai, 'sigma_normal', pedParams.sigma_ai, 'min_value', pedParams.ai_min, 'max_value', pedParams.ai_max);
lambdai = sampling_distributions('normal', n_samples, 'mu_normal', pedParams.mu_lambdai, 'sigma_normal', pedParams.sigma_lambdai, 'min_value', pedParams.lambdai_min, 'max_value', pedParams.lambdai_max);
bi = 2*pi*ones(n_samples,1);
Ai = 2.*1./bi.^2;
k1i = pedParams.k1*ones(n_samples,1);
k2i = pedParams.k2*ones(n_samples,1);
k3i = pedParams.k3*ones(n_samples,1);
alpha_si = pedParams.alpha_s*ones(n_samples,1);

% side where the pedestrian get on the bridge using bernoulli
p_side = pedParams.p_side;
% q_side = 1-p_side;
side = double(rand(n_samples, 1) <= p_side);% (1: enters from right (x=L); 0: enters from left (x=0))

% Time a pedestrian to be incorporated to the bridge (random between Tped_min
% and Tped_max)
ti = zeros(n_samples,1);
ti(1) = pedParams.Twaitadd;
ti(2:end) = sampling_distributions('uniform', n_samples-1, 'range_uniform', pedParams.range_ti);
% tac = cumsum(ti);

% % Number of pedestrians as a function of time (ppt)
% arrival_times = cumsum(ti);
% counts = histcounts(arrival_times, simParams.t_vect).';
% ppt_cumulative = cumsum(counts);
% ppt = [0; ppt_cumulative];
% if simParams.t_vect_oder_Tppt == 0 % only use the ti and the last pedestrian should last t_extra (another random)
%     t_extra = sampling_distributions('uniform', 1, 'range_uniform', pedParams.range_ti);
%     T_final = tac(end) + t_extra;
%     idx_time = find(simParams.t_vect >= T_final, 1, 'first');
%     if isempty(idx_time)
%         idx_time = length(simParams.t_vect);
%     end
%     % simParams.t_vect = simParams.t_vect(1:idx_time);   % out of the function
%     ppt = ppt(1:idx_time);
% end

% Initial conditions
loci = sampling_distributions('uniform', n_samples, 'range_uniform', [0, pedParams.loc_L]);
v0 = sampling_distributions('uniform', n_samples, 'range_uniform', pedParams.x0_range);
v0p = sampling_distributions('uniform', n_samples, 'range_uniform', pedParams.x0p_range);

end