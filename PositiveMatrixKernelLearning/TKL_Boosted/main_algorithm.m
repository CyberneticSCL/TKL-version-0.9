function [SVM] = main_algorithm(SVM, Kernel, mu, tau0, sigma0, maxit, print_index)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM = main_algorithm(SVM, Kernel, mu, tau0, sigma0, maxit, print_index) function
% finds optimal Kernel and alpha using APD.
% 
% INPUT
% SVM:                SVM object.
% Kernel:             Kernel.
% mu, tau0, sigma0:   Parameters for APD.
% maxiter:            Maximum number of iterations
% print_index:        The integer, that The script print results every print_index iterations.
%
% OUTPUT
% SVM: A fit SVM with an optimal Positive Matrix Kernel function.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - main_algorithm
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% INITIAL PARAMETRIZATION
    iter = 0;
    gamma_k = sigma0/tau0;
    tau_k = tau0;
    sigma_k = sigma0;
    go = true;  
    tol = SVM.Params.tol;
    while (iter < maxit) && go
        %% MAIN ALGORITHM
        iter = iter+1;
        
        tic;
        sigma_prev= sigma_k;
        sigma_k = gamma_k*tau_k;
        theta_k = sigma_prev/sigma_k;
        P_current     = SVM.Params.P; 
        %% UPDATE VARIABLES
        SVM = findP_2(SVM, Kernel, tau_k, sigma_k,  theta_k, iter);
        if iter > maxit | min(SVM.Opt.dualGap(end),SVM.Opt.dualGap2(end)) < tol % Quits if maxit > current iteration or if the objective function change was too small.
            go = false;
        end 

        gamma_next = gamma_k*(1 + mu*tau_k);
        tau_next   = tau_k*sqrt(gamma_k/gamma_next);

        SVM.Opt.T(end+1) = toc;
        
        gamma_k = gamma_next;
        tau_k = tau_next;
        if mod(iter, print_index)==0
                fprintf('%10d  |  %1.4e  |  %1.4e \n', iter, SVM.Opt.Obj(end), min(SVM.Opt.dualGap(end),SVM.Opt.dualGap2(end)));
        end
    end
end