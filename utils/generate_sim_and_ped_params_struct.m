% Use this script to generate sim and ped params
% Is better to run this script in the main script

% Init structus
simParams = struct();
pedParams = struct();

% simParams
simParams.t_init = t_init;
simParams.t_step = t_step;
simParams.t_end = t_end;
simParams.t_vect = t_vect;
simParams.t_vect_oder_Tppt = t_vect_oder_Tppt;

% PedParams
pedParams.P_ped = P_ped;
pedParams.mu_m = mu_m;
pedParams.sigma_m = sigma_m;
pedParams.m_min = m_min;
pedParams.m_max = m_max;
pedParams.mu_v = mu_v;
pedParams.sigma_v = sigma_v;
pedParams.v_min = v_min;
pedParams.v_max = v_max;
pedParams.mu_freq= mu_freq;
pedParams.sigma_freq = sigma_freq;
pedParams.freq_min = freq_min;
pedParams.freq_max = freq_max;
pedParams.rhofv = rhofv;
pedParams.mu_fv = mu_fv;
pedParams.sigma_fv = sigma_fv;
pedParams.min_fv = min_fv;
pedParams.max_fv = max_fv;
pedParams.covariance_fv = covariance_fv;
pedParams.mu_ai = mu_ai;
pedParams.sigma_ai = sigma_ai;
pedParams.ai_min = ai_min;
pedParams.ai_max = ai_max;
pedParams.mu_lambdai = mu_lambdai;
pedParams.sigma_lambdai = sigma_lambdai;
pedParams.lambdai_min = lambdai_min;
pedParams.lambdai_max = lambdai_max;
pedParams.b = b;
pedParams.A = A;
pedParams.k1 = k1;
pedParams.k2 = k2;
pedParams.k3 = k3;
pedParams.alpha_s = alpha_s;
pedParams.x0_range = x0_range;
pedParams.x0p_range = x0p_range;
pedParams.loc_L = L;
pedParams.range_ti= range_ti;
pedParams.p_side = p_side;
pedParams.Twaitadd = Twaitadd;
pedParams.Tped_min = Tped_min;
pedParams.Tped_max = Tped_max;

% print "simParams and pedParams structs are now in the workspace" 
disp('simParams and pedParams structs are now in the workspace');