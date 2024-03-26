function [pResults, gResults] = createParameters(pedParams, groupParams)

    %% Decompose params
    % Groups
    nGroups = groupParams.nGroups;                                  % Number of groups to be created
    npg_min = min(groupParams.npgDistr(:,1));                        % Minimum number of pedestrians per group
    npg_max = max(groupParams.npgDistr(:,1));                        % Maximum number of pedestrians per group
    npgDistr = groupParams.npgDistr;                                  % Probability that a group have this size
    sigma_v = groupParams.sigma_v;                                  % Standard deviation of walking speed

    % Pedestrians
    mu_m = pedParams.mass(1);                                      % Mean mass of pedestrians
    sigma_m = pedParams.mass(2);                                   % Standard deviation of mass of pedestrians
    m_min = pedParams.mass(3);                                     % Minimum mass of pedestrians
    m_max = pedParams.mass(4);                                     % Maximum mass of pedestrians
    mu_freq = pedParams.freq(1);                                   % Mean gait frequency of pedestrians
    sigma_freq = pedParams.freq(2);                                % Standard deviation of gait frequency of pedestrians
    freq_min = pedParams.freq(3);                                  % Minimum gait frequency of pedestrians
    freq_max = pedParams.freq(4);                                  % Maximum gait frequency of pedestrians
    mu_v = pedParams.speed(1);                                     % Mean walking speed of pedestrians
    sigma_v = pedParams.speed(2);                                  % Standard deviation of walking speed of pedestrians
    v_min = pedParams.speed(3);                                    % Minimum walking speed of pedestrians
    v_max = pedParams.speed(4);                                    % Maximum walking speed of pedestrians
    corr_vf = pedParams.corr_fv;                                   % Correlation between walking speed and gait frequency
    mu_ai = pedParams.ai(1);                                       % Mean factor a_i for each pedestrian
    sigma_ai = pedParams.ai(2);                                    % Standard deviation of factor a_i for each pedestrian
    ai_min = pedParams.ai(3);                                      % Minimum factor a_i for each pedestrian
    ai_max = pedParams.ai(4);                                      % Maximum factor a_i for each pedestrian
    mu_lambdai = pedParams.lambdai(1);                             % Mean factor lambda_i for each pedestrian
    sigma_lambdai = pedParams.lambdai(2);                          % Standard deviation of factor lambda_i for each pedestrian
    lambdai_min = pedParams.lambdai(3);                            % Minimum factor lambda_i for each pedestrian
    lambdai_max = pedParams.lambdai(4);                            % Maximum factor lambda_i for each pedestrian
    b = pedParams.b;                                               % Factor b_i for each pedestrian
    p_dir = pedParams.p_dir;                                       % Direction of pedestrian movement
    
    %% Groups
    % Group size for each group
    gSizes = zeros(nGroups,1);
    aux_gs = rand(nGroups,1);
    cumDistr = cumsum(npgDistr(:,2));
    for i = 1:npg_max   % Fill group sizes vector
        gSizes((aux_gs < cumDistr(i+1, 1)) & (aux_gs > cumDistr(i, 1))) = npgDist(i+1, 1);
    end

    % walking speed for each group
    vg = mu_v + sigma_v*randn(nGroups,1);
    vg = valuesFixing(vg, mu_v, sigma_v, v_min, v_max);

    % Side by which the group enters the bridge (Bernoulli)
    sides = rand(nGroups,1) < p_dir;

    % group results
    gResults = struct();
    gResults.gSizes = gSizes;
    gResults.vg = vg;
    gResults.sides = sides;

    %% Pedestrians
    n_Pedestrians = sum(gSizes);

    % gait frequency of each pedestrian
    f_peds = normrnd(mu_freq, sigma_freq,[n_Pedestrians,1]); % hz
    f_peds = valuesFixing(f_peds, mu_freq, sigma_freq, freq_min, freq_max);

    % walking speed of each pedestrian (extract from walking speed of each group and consider the correlation with f_peds)
    v_peds = zeros(n_Pedestrians,1);
    for i = 1:nGroups
        v_peds(sum(gSizes(1:i-1)) + 1:sum(gSizes(1:i))) = vg(i) + corr_vf*(f_peds(sum(gSizes(1:i-1))+1:sum(gSizes(1:i))) - mu_freq);
    end

    % Mass for each pedestrian
    m_peds = normrnd(mu_m, sigma_m, [n_Pedestrians,1]);
    m_peds = valuesFixing(m_peds, mu_m, sigma_m, m_min, m_max);

    % Factor a_i for each pedestrian
    ai_peds = normrnd(mu_ai,sigma_ai,[n_Pedestrians,1]);
    ai_peds = valuesFixing(ai_peds, mu_ai, sigma_ai, ai_min, ai_max);

    % Factor lambda_i for each pedestrian
    lambdai_peds = normrnd(mu_lambdai,sigma_lambdai,[n_Pedestrians, 1]); % 1/s
    lambdai_peds = valuesFixing(lambdai_peds, mu_lambdai, sigma_lambdai, lambdai_min, lambdai_max);

    % Factor b_i for each pedestrian 
    bi_peds = b*ones(n_Pedestrians,1);                                                   % Let this value be 2pi for speed up the waveform 

    %% Pedestrian Results
    pResults = struct();
    pResults.v_peds = v_peds;
    pResults.f_peds = f_peds;
    pResults.m_peds = m_peds;
    pResults.ai_peds = ai_peds;
    pResults.lambdai_peds = lambdai_peds;
    pResults.bi_peds = bi_peds;
end