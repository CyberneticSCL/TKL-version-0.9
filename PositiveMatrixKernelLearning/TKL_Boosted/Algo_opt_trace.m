function [X] = Algo_opt_trace(A, q, tol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [X] = Algo_opt_trace(A, q, tol) function takes a Matrix A, a trace
% constraint q and tolerance and finds the optimal P>0
% P = argmin||X-A||_F.
%     s.t.   X > 0
% 
% INPUT:  A - symmetric matrix nxn
%         q - trace
%        tol- tolerance
% OUTPUT: X - PSD matrix nxn: minimize||A - X||_F s.t. traceX = q and X>=0 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL -   Algo_opt_trace 
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    eps = tol; 
    n = size(A, 1);
    [V, E] = eig(A);
    diag_E = real(diag(E));
    y = 0;
    y_l = -max(abs(diag_E))-q;
    y_r =  max(abs(diag_E))+q;
    r = 1000;
%     if sum(diag_E(diag_E > 0)) < q
%        y = 0;
%     else
    while abs(r) > eps
        y = 0.5*(y_l + y_r);
        r = sum( diag_E(diag_E - y > 0) - y) - q;
        if r < 0 
            y_r = y;
        else
            y_l = y;
        end
    end 
    indexes = diag_E - y > 0;

    X = zeros(size(A));
    for i=1:length(diag_E)
       if indexes(i) == 0
           continue
       end
       v_i = V(:,i);
       X = X + (diag_E(i) - y)*real(v_i*v_i'); 
    end
end