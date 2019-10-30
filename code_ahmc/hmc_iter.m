function [next_q, accept, q, mr, RandomStep, energy] = hmc_iter(l, epsilon, q, ...
    gradient, func, sigmaCholR)

% Input Arguments:
% l:            The maximum number of leapfrogs.
% q:            The previous sample.
% epsilon:      The step size.
% func:         A function handle that returns the NEGATIVE log density.
% gradient:     A function handle that returns the NEGATIVE log gradient of the
%               target density
% sigmaCholR:   the lower triangular Cholesky factor of the inverse 
%               of the mass matrix, i.e. sigmaCholR*sigmaCholR' = inv(Mass).
%               If sigmaCholR is not assigned, it is taken to be I.


% Outputs:
% next_q:       Next sample.
% accept:       Indicates whether the proposal was accepted.
% q:            The proposal.
% mr:           The Metropolis-Hastings acceptence ratio.
% RandomStep:   The number of leapfrog steps actually taken.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


ignoreSigma = nargin < 6;       % Take the mass matrix to be the identity.

% Generate gaussian random variables
old_p = randn(size(q, 1), 1);
if ~ignoreSigma
    old_p = sigmaCholR'\old_p;
end

p = old_p; old_q = q;

RandomStep = ceil(rand*l);      % Randomize the number of leapforgs taken.

% The leapfrog steps
p = p - epsilon*gradient(q)/2;  % Take a half leap
for i = 1:RandomStep
    if sum(isnan(p)) > 0
        break
    end
    
    if ignoreSigma
        q = q + epsilon*p;
    else
        q = q + epsilon*(sigmaCholR*sigmaCholR'*p);
    end
    p = p - epsilon*gradient(q);
    
end
p = p + epsilon*gradient(q)/2;

% To accept or reject the proposed values
proposed_E = func(q); original_E = func(old_q);

if ignoreSigma
    proposed_K = p'*p/2; original_K = old_p'*old_p/2;
else
    proposed_K = norm(sigmaCholR'*p)^2/2;
    original_K = norm(sigmaCholR'*old_p)^2/2;
end

% Note: Some modifications made by Wolfgang Roth:
% We observed that proposed_E was infinity and proposed_K was NaN.
% This resulted in mr=1, the sample was accepted although it should have a
% probability of 0 being accepted.
if isnan(proposed_E) || isnan(original_E) || isnan(original_K) || isnan(proposed_K)
  mr = 0;
else
  mr = min(exp(-proposed_E + original_E + original_K - proposed_K), 1);
end

next_q = old_q; accept = 0; energy = -original_E;
if rand < mr
    next_q = q; accept = 1; energy = -proposed_E;
end