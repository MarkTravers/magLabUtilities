function [Lp,dLp,t] = Legendre_Interpolation(f,dt,nWin,nStep,p)
% Compute a Legendre Polynomial Interpolation
% nWin = 500;                 % number of samples in the integration window
% nStep = 50;                 % step size between post-processed points

nWinO2 = floor(nWin/2);     % 
[npts, m] = size(f);     % number of points in the raw data
nSize = floor((npts-nWin)/nStep);% % size of the interpolated data array
alphaH = zeros(nSize,p+1);  % coeficients of interpolation
t_Lp = zeros(nSize,1);
H_Lp = zeros(nSize,1);
dH_Lp = zeros(nSize,1);

ict = 0;
for n=1:nStep:npts
    ict = ict+1;
    n1 = n-nWinO2;
    n2 = n+nWinO2;
    if (n1 < 1)
        n1 = 1;
        n2 = nWin+1;
    elseif(n2 > npts)
        n2 = npts;
        n1 = npts-nWin;
    end
    
    % Inner product of Legendre Polynomials (use a simple
    % trapezoidal rule for integration)
    intSum = zeros(p+1,1);
    for i = n1:n2
       x = (2.0*i-(n1+n2))/(n2-n1);
       LegP = LegPoly(p,x);
       mult = 1.0;
       if(i == n1 || i == n2)
           mult = 0.5;
       end
       for j = 1:p+1
           intSum(j) = intSum(j) + f(i)*LegP(j)*mult;
       end
    end
    intSum = intSum*2.0/(n2-n1);
    
    %%%  Divide by Int(-1,1) Pm(x) Pn(x) = 2/(2n+1) *deltan,m
    for j = 1:p+1
        alphaH(ict,j) = intSum(j)*(2.0*j-1.0)*0.5;
    end
    
    % Compute the interpolated field at the sample point:
    t_Lp(ict) = (n-1)*dt;
    x = (2.0*n-(n1+n2))/(n2-n1);
    % Compute the Legendre polynomial and its derivative:
    [LegP, dLegP] = LegPoly_d(p,x);
    sum = 0.0;
    sumd = 0.0;
    for j = 1:p+1
        sum = sum + alphaH(ict,j)*LegP(j);
        sumd = sumd + alphaH(ict,j)*dLegP(j);
    end
    H_Lp(ict) = sum;
    dH_Lp(ict) = sumd*2.0/(dt*(n2-n1));

end
Lp = H_Lp;
dLp = dH_Lp;
t = t_Lp;
end


