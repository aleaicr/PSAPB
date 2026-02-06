function max_response = getMaxResponse_t(response)
% Compute the maximum response as a function of time
%
% Input:
% response: Response as a function of time in multiple locations
%           dimensions: [time rows, locations columns]
% Output:
% max_response: Absolute maximum response as a function of time
%           dimensions: [time rows]
%

% Compute maximum response
max_response = max(abs(response), [], 2);

end
