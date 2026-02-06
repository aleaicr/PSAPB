% Obtain the path of the simulink file to use as a function of the bridge and the crowd size

% For bridge 1, crowd sizes < 75 do not trigger a numerical instability, so runge kutta ode4 fixed step can be used
% But for crowd sizes >= 75 triggers a numerical instability so the integration method must be changed to Backward Euler
if bridge_num == 1
    if P < 75 % Runge Kutta ode4
        simName = [SIMULINK_FOLDER_PATH, 'initialloc_acc_', num2str(P), '.slx'];
    elseif P == 75 % Dormand Prince ode45
        simName = [SIMULINK_FOLDER_PATH, 'dormandprince_initialloc_acc_75.slx'];
        t_step = 1/1000;
        t_step_min = 1/10000;
        t_step_max = t_step;
    else % Backward Euler
        simName = [SIMULINK_FOLDER_PATH, 'euler_initialloc_acc_', num2str(P), '.slx'];
    end
    % For bridge 2, crowd sizes < 175 do not trigger a numerical instability, so runge kutta ode4 fixed step can be used
    % But for crowd sizes >= 175 triggers a numerical instability so the integration method must be changed to Backward Euler
elseif bridge_num == 2
    if P < 175 % Runge Kutta ode 4
        simName = [SIMULINK_FOLDER_PATH, 'initialloc_acc_', num2str(P), '.slx'];
    else % Backward Euler (I should test if Dormand prince works for P = 200, if not, then B.Euler)
        simName = [SIMULINK_FOLDER_PATH, 'euler_initialloc_acc_', num2str(P), '.slx'];
    end
    % For the other bridges, the crowd size does not trigger a numerical instability, so runge kutta ode4 fixed step can be used
else
    simName = [SIMULINK_FOLDER_PATH, 'initialloc_acc_', num2str(P), '.slx'];
end