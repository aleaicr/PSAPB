function [mi,vi,fi,wi,ai,lambdai,bi,side,tac,v0,v0p] = NewPedestrianProperties(simParams,pedParams,pq)
% This function creates random properties for pedestrians following their distribution
%
% INPUTS
% pedParams:    Struct, Parameters for make the selection of the pedestrian properties
% simParams:    Struct, Parameters for the simulation
% pq:           Number of pedestrians to generate the properties
% 
% OUTPUTS
% mi:                       [kg] Vector with the mass of each pedestrian
% vi:                       [m/s] Vector with the longitudinal walking speed of each pedestrian
% fi:                       [Hz] Vector with the mean frequency of the walking of each pedestrian
% wi:                       [rad/s] Vector with the natural frequency of the Van der Pol oscillator of each pedestrian
% ai:                       [-] Vector with the parameter of the Van der Pol oscillator of each pedestrian
% lambdai:                  [1/s] Vector with the parameter of the Van der Pol oscillator of each pedestrian
% bi:                       [-] Vector with the speed up factor of each pedestrian
% side:                     [-] Vector with the side of the bridge where each pedestrian is walking (1 or 2)
% tac:                      [s] Vector with the incorporation time of each pedestrian
% ppt:                      [s] Vector with the amount of pedestrian in function of time
%
% Comments
% * The used distributions are as follows:
% Mass                      - Normal distribution
% Speed                     - Normal distribution
% Frequency                 - Normal distribution
% Van der Pol parameters    - Normal distribution
% Incorporation time        - Uniform distribution
% Side                      - Random between 1 and 2 (side 1 and side 2) - Dichotomous

%% Read struct parameters
% tpq = simParams.tpq;
Tadd_min = simParams.Tadd_min;
Tadd_max_min = simParams.Tadd_max_min;
Tadd_max_max = simParams.Tadd_max_max;
Tadd_first = simParams.Tadd_first;

mu_m = pedParams.mu_m;
sigma_m = pedParams.sigma_m;
m_min = pedParams.m_min;
m_max = pedParams.m_max;
mu_v = pedParams.mu_v;
sigma_v = pedParams.sigma_v;
v_min = pedParams.v_min;
v_max = pedParams.v_max;
mu_freq = pedParams.mu_freq;
sigma_freq = pedParams.sigma_freq;
freq_min = pedParams.freq_min;
freq_max = pedParams.freq_max;
rhofv = pedParams.rhofv;
mu_ai = pedParams.mu_ai;
sigma_ai = pedParams.sigma_ai;
ai_min = pedParams.ai_min;
ai_max = pedParams.ai_max;
mu_lambdai = pedParams.mu_lambdai;
sigma_lambdai = pedParams.sigma_lambdai;
lambdai_min = pedParams.lambdai_min;
lambdai_max = pedParams.lambdai_max;
b = pedParams.b;
v0_range = pedParams.v0_range;
v0p_range = pedParams.v0p_range;

%% Data generation
% Mass
mi = normrnd(mu_m,sigma_m,[pq,1]);       % kg
% Gait frequency
fi = normrnd(mu_freq,sigma_freq,[pq,1]); % hz
% Walking speed
mu_v = (mu_v - rhofv*mu_freq)/sqrt(1-rhofv^2);                          % Median correction
sigma_v = sqrt(sigma_v^2 - (rhofv*sigma_freq)^2)/sqrt(1-rhofv^2);       % Standard deviation correction
vi = normrnd(mu_v,sigma_v,[pq,1]);       % m/s
% ai
ai = normrnd(mu_ai,sigma_ai,[pq,1]);
% lambdai
lambdai = normrnd(mu_lambdai,sigma_lambdai,[pq,1]); % 1/s
% bi 
bi = b*ones(n_max,1);                                                   % Let this value be 2pi for speed up the waveform 

% Incorporation times
Tadd_max = randi([Tadd_max_min, Tadd_max_max]);                         % Generate a random value for the maximum incorporation time
Taddvectprima = randi([Tadd_min,Tadd_max],[pq-1,1]);                    % Generate a vector with the wait time of the pedestrians to incorporate to the bridge
Tadd_vect = [Tadd_first;Taddvectprima];                                 % Add the first pedestrian wait time
tac = cumsum(Tadd_vect);                % s                             % Calculate the incorporation time of each pedestrian      

% Side to enter the bridge
side = randi([1,2],[pq,1]);                                            % Dichotomous variable to indicate the side of the bridge where the pedestrian is walking

%% Review the values (if they are out of the realistic limits, then change them)
for i = 1:pq                                                           % Pedestrians loop
    % mass
    if or(mi(i) < m_min, mi(i) > m_max)
        mi(i) = valueFixing(mi(i),mu_m,sigma_m,m_max,m_min);
    end

    % frequency
    if or(fi(i) < freq_min, fi(i) > freq_max)
        fi(i) = valueFixing(fi(i),mu_freq,sigma_freq,freq_max,freq_min);
    end

    % Walking speed
    if or(vi(i) < v_min, vi(i) > v_max)
        vi(i) = valueFixing(vi(i),mu_v,sigma_v,v_max,v_min);
    end

    % ai
    if ai(i) < ai_min || ai(i) > ai_max
        ai(i) = valueFixing(ai(i),mu_ai,sigma_ai, ai_max, ai_min);
    end

    %lambdai
    if lambdai(i) < lambdai_min || lambdai(i) > lambdai_max
        lambdai(i) = valueFixing(lambdai(i),mu_lambdai,sigma_lambdai, lambdai_max, lambdai_min);
    end
end

%% Missing data
% Circular frequency of lateral gait
wi = pi*fi;                         % rad/sec               (w_lateral = 2*pi*(f_vertical/2) = pi*f)

% Correlated walking speed in function of the frequency
vi = rhofv*fi + sqrt(1-rhofv^2)*vi;     % m/s               % This is why mu_v and sigma_v are corrected

% Initial conditions
v0 = randi(v0_range,[n_max 1])/1000;
v0p = randi(v0p_range,[n_max 1])/100;

end