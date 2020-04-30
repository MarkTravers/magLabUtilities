function [P, dP] = LegPoly_d(N,x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Recursive function to compute the Legendre
%%  polynomials of order 0:N
%
if(N < 0) 
    return;
end
P = zeros(N+1,1);
dP = zeros(N+1,1);
P(1) = 1.;         % P0 = 1;
dP(1)= 0.;
if(N < 1) 
    return;
end

P(2) = x;          %P1 = x;
dP(2)= 1.0;
sgn = -1.0;
for n = 2:N      % loop over the remaining Legendre orders
    m = n+1;             %
    P(m)= ((2.0*n-1.)*x*P(n) - (n-1.0)*P(n-1))/n;
    if(abs(x) ~= 1.0)
        dP(m) = (x*P(m)-P(n))*n/(x*x-1.0);
    else
        dP(m) = (n+1)*n/2;
        if(x < 0.0)
            dP(m) = dP(m)*sgn;
            sgn = -sgn;
        end
    end
end
