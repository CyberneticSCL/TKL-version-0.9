function [val] = tTK(a,b,c)
%T Summary of this function goes here
%   Detailed explanation goes here

val = prod( (b.^c)./c-(a.^c)./c , 2);

% Old for loop version
% val = 1;
% for n = 1:length(a)
%     val = val.*((b(n).^c(n))./c(n) - (a(n).^c(n))./c(n));
% end


end

