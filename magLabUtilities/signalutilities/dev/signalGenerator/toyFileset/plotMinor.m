clear all; close all;
dataH = load("ring_Minor_prb_DeMagCellsh.txt");
dataM = load("ring_Minor_prb_grp_DeMagCells_0.txt");

H = dataH(1:20:end,6);
M = dataM(1:20:end,6);

plot(H,M);

data(1,:) = H;
data(2,:) = M;

csvwrite('magstromMinor.csv',data');