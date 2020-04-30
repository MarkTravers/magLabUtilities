function P = LegPoly(N,x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Recursive function to compute the Legendre
%%  polynomials of order 0:N
%
if(N < 0) 
    return;
end
P = zeros(N+1,1);   
P(1) = 1;          % P0 = 1;
if(N < 1) 
    return;
end

P(2) = x;          %P1 = x;

for n = 2:N      % loop over the remaining Legendre orders
    m = n+1;             %
    P(m)= ((2.0*n-1.)*x*P(n) - (n-1.0)*P(n-1))/n;
end
