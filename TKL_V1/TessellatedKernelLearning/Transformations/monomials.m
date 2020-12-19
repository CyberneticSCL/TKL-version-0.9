function [z] = monomials(x, d)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [z] = monomials(x,d) function takes a matrix of inputs (x) and computes
% the monomial basis of degree d (z).
%
% INPUT
% x: Input data.
% d: Maximum monomial degree.
%
% OUTPUT
% z: Monomial basis of the input data.  
% 
% NOTES:
% For support, contact M. Peet, Arizona State University at mpeet@asu.edu
% or B.K. Colbert at brendon.colbert@asu.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TKL - monomials
%
% Copyright (C)2019  M. Peet, B.K. Colbert
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% If you modify this code, document all changes carefully and include date
% authorship, and a brief description of modifications
%
% Initial coding MMP, BKC  - 12_15_2020
%


dim = size(x,1); index = []; vec = []; % Initialize variables
%% Add degree 1 monomials
if d > 0
    for n = 1:dim
        vecTemp(n,:) = x(n,:); % Store current monomials of highest degree
        vec(n,:) = x(n,:); % Store monomial basis of all degree one monomials
        index(end+1) = n; % Store the monomial that was added to vecTemp
    end
end

%% Add degree > 1 monomials
for k = 1:d-1
    vecNew = []; % Initialize vector to store future monomials of highest degree
    indexNew = []; % Initialize vector that store the monomial that was added to vecNew
    for n = 1:size(vecTemp,1)
        for l = index(n):dim
            vecNew(end+1,:) = vecTemp(n,:).*x(l,:); % Update the vector of current highest degree monomials
            vec(end+1,:)  = vecNew(end,:); % Update the vector of all monomials
            indexNew(end+1) = l; % Update the index of the current highest degree monomials
        end
    end
    vecTemp = vecNew; % Initialize vecTemp as vecNew for the next loop
    index = indexNew; % Initialize index as indexNew for the next loop
end
z = [ones(1,size(x,2));vec]; % Add degree 0 monomial and return the monomial basis
end

