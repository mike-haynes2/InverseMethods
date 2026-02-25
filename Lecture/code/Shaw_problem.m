clear
close all

set(groot, 'DefaultLineLineWidth', 1.0);
set(gca,'FontSize',18,'LineWidth',1.0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Reproducing Example 3.3 in Parameter Estimation & Inverse Problems     
%%% by Aster et al. The Shaw problem.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load shaw20.mat

% Compute the SVD.
[U,S,V] = svd(G);

% Plot the singular values, notice large dynamic range
figure(1)
semilogy(diag(S),'ko');
ylim([10^-18 10^3])
xlabel('i');
ylabel('s_i');

% Create a spike model
spike = zeros(20,1);
spike(10) = 1;

% Plot the spike model
figure(2);
plotconst(spike,-pi/2,pi/2,'k'); % Plots a model in piecewise constant form over n subintervals
axis([-2 2 -0.5 1.5]);
ylabel('Intensity')
xlabel('\theta (radians)')

% Use spike model to produce noise-free synthetic data (dspike).
dspike = G*spike;

% Plot the synthtic data
figure(3)
plotconst(dspike,-pi/2,pi/2,'k'); % Plots a model in piecewise constant form over n subintervals
axis([-2 2 -0.25 .75]);
ylabel('Intensity')
xlabel('s (radians)')

% Generalized solution for noise-free data
spikemod = G\dspike;

% Plot the inverse model solution for the noise-free spike data
figure(4)
plotconst(spikemod,-pi/2,pi/2,'k'); % Plots a model in piecewise constant form over n subintervals
axis([-2 2 -0.5 1.5]);
ylabel('Intensity')
xlabel('\theta (radians)')

% Create slightly noisy data (dspiken) and see what happens.
dspiken = dspike + 1.0e-6*randn(size(dspike));

% Find the pseudoinverse solution with noisy data for p=18.
p = 18;
Up = U(:,1:p);
Vp = V(:,1:p);
Sp = S(1:p,1:p);
spikemod18n = Vp * inv(Sp) * Up' * dspiken;

% Plot pseduoinverse solution 
figure(5)
plotconst(spikemod18n,-pi/2,pi/2,'k'); % Plots a model in piecewise constant form over n subintervals
ylabel('Intensity')
xlabel('\theta (radians)')

% Find the pseudoinverse solution with data for p=10.
p = 10;
Up = U(:,1:p);
Vp = V(:,1:p);
Sp = S(1:p,1:p);

% recover the noise-free model
spikemod10 = Vp * inv(Sp) * Up' * dspike;

% recover the noisy model
spikemod10n = Vp * inv(Sp) * Up' * dspiken;

% Plot comparison of models for noise-free and noisy data, for p=10
figure(6)
plotconst(spikemod10,-pi/2,pi/2,'k-'); % Plots a model in piecewise constant form over n subintervals
hold on
plotconst(spikemod10n,-pi/2,pi/2,'r--'); 
axis([-2 2 -0.2 0.5]);
ylabel('Intensity')
xlabel('\theta (radians)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Reproducing Example 4.1 in Parameter Estimation & Inverse Problems     
%%% by Aster et al. The Shaw problem.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

s=diag(S);

% calculate and plot the L-curve
[rho,eta,reg_param]=l_curve_tikh_svd(U,s,dspiken,1000); % see function

% Plot the L-curve
figure(7)
loglog(rho,eta,'k-');
xlabel('Residual Norm ||Gm - d||_2')
ylabel('Solution Norm ||m||_2')
axis tight

% get the spike solution corresponding to the L-curve corner
alpha_tikh = 6.4e-6;

m_tikh=(G'*G+alpha_tikh^2*eye(20))\G'*dspiken;

% residual using the L-curve solution
r_spike=norm(G*m_tikh-dspiken);

figure(8)
plotconst(m_tikh,-pi/2,pi/2,'k');
xlabel('\theta');
ylabel('Intensity');
ylim([-.2 0.5])

% Use the discrepancy principle to get a second solution.
% find the regularization value, alpha_disc, for rho=discrep by interpolation 
% of the L-curve
alpha_disc=4.29e-5;

% get the model and residual
m_disc=(G'*G+alpha_disc^2*eye(20))\G'*dspiken;
r_spike_disc=norm(G*m_disc-dspiken);

figure(9)
H1=plotconst(spike,-pi/2,pi/2,'k-');
hold on
H2=plotconst(m_disc,-pi/2,pi/2,'k--');
xlabel('\theta');
ylabel('Intensity');

%%%

function [rho,eta,reg_param] = l_curve_tikh_svd(U,s,d,npoints,varargin)
 
% Initialization. 
[m,n] = size(U);
[p] = length(s); 
 
% compute the projection, and residual error introduced by the projection
d_proj = U'*d;
dr = norm(d)^2 - norm(d_proj)^2;

%data projections
d_proj = d_proj(1:p); 

%scale series terms by singular values
d_proj_scale = d_proj./s; 
 
% initialize storage space
eta = zeros(npoints,1);
rho = eta;
reg_param = eta;
s2 = s.^2; 

if size(varargin,2)==0
% set the smallest regularization parameter that will be used
smin_ratio = 16*eps;
reg_param(npoints) = max([s(p),s(1)*smin_ratio]); 

% ratio so that reg_param(1) will be s(1)
ratio = (s(1)/reg_param(npoints))^(1/(npoints-1));
end

if size(varargin,2)==2
    alpharange=cell2mat(varargin);
    reg_param(npoints)=alpharange(2);
    ratio=(alpharange(1)/alpharange(2))^(1/(npoints-1));
end
    

% calculate all the regularization parameters
for i=npoints-1:-1:1
    reg_param(i) = ratio*reg_param(i+1);
end 

% determine the fit for each parameter
for i=1:npoints
  %GSVD filter factors
  f = s2./(s2 + reg_param(i)^2); 
  eta(i) = norm(f.*d_proj_scale); 
  rho(i) = norm((1-f).*d_proj); 
end 

% if we couldn't match the data exactly add the projection induced misfit
if (m > n && dr > 0)
  rho = sqrt(rho.^2 + dr);
end

end

function H=plotconst(x,l,r,c)

% Find length of model
n=length(x);
% Find size of each interval
delta=(r-l)/n;
% Dummy values at beginning of vectors to allow concatination
myx=[0];
myy=[0];
% Iteratively fill vector of x and y values for steps for plot
% by concatinating onto dummy vectors
for i=1:n
  myx=[myx ((i-1)*delta+l:(delta/20):i*delta+l)];
  myy=[myy (ones(1,21)*x(i))];
end

% Find length of resulting vector of x values
l2=length(myx);
% Truncate vectors to remove dummy values used in concatination
myx=myx(2:l2);
myy=myy(2:l2);

% Plot piecewise constant graph
plot(myx,myy,c);
H=gca;
set(H,'FontSize',18);
set(H,'LineWidth',1.0);
end

