%% %%% AWA2 TEST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Semi-supervised for Zero-shot Learning. 
% 
% By XBR.
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

warning off

clc, clear all, close all
addpath('data_zsl')
addpath('library')

load('data_zsl/AWA2_res101_picture.mat');%AWA2_res101_ms
load('ale_AWA_1e-3_50.mat');


A = att;

%% initialize
X = [X_tr];  
[p, n] = size(X);
[c, m] = size(A);


S = eye(c);
C = rand(c);
AF = att*F_new;
F_ori = F;
F = F_new;

W = W';


% Construct the K-NN Graph
options.NeighborMode= 'KNN';
options.k = 0;
W1 = constructW(X',options);
Di = sum(W1,2);
Di = diag(Di);
L = (Di-W1);



alpha = 0.0001;
mu = 1e-6;
Y2 = zeros(c);
lambda =0.001;% [0.01 0.001 0.0001 0.00001 0.000001];
rho = 1.2;
max_mu = 1e6;
lam = 1000;

O1 = ones(1,m);
O2 = ones(1,n);

loss = 0;
for iter=1:100
%% classification
 
dist     =  1 - (pdist2(X_te'*W', att_new', 'cosine'));
dist     = zscore(dist);
HITK     = 1;
Y_hit5   = zeros(size(dist,1),HITK);
for i    = 1:size(dist,1)
    [sort_dist_i, I] = sort(dist(i,:),'descend');
    Y_hit5(i,:) = te_cl_id(I(1:HITK));
end

n=0;
for i  = 1:size(dist,1)
    if ismember(test_labels_cub(i),Y_hit5(i,:))
        n = n + 1;
    end
end
zsl_accuracy = n/size(dist,1);
fprintf('AWA2 ZSL accuracy: %.1f%%\n', zsl_accuracy*100);


%% Update C
temp = S + Y2/mu;
[U,sigma,V] = svd(temp,'econ');
sigma = diag(sigma);
svp = length(find(sigma>lambda/mu));
if svp>=1
    sigma = sigma(1:svp)-lambda/mu;
else
    svp = 1;
    sigma = 0;
end
C = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    
%% update S
Wx = W*X;
inv_a = inv(Wx*Wx'+(0.1+mu)*eye(size(Wx,1)));
S = (AF*Wx'+mu*C-Y2)*inv_a;


%% Update W
%with ||W||_F^2
A1 = S'*S;
B1 = lam*inv(X*X');
C1 = (S'*AF*X'+lam*W)*inv(X*X'); 
W = sylvester(A1,B1,C1);

%% with graph
% A1 = S'*S;
% B1 = 1e2*X*L*X'*inv(X*X');
% C1 = S'*AF*X'*inv(X*X'); 
% W = sylvester(A1,B1,C1);


%% Update F
WA = A;
PX = S*W*X;
AX = WA'*PX;
AAF = WA'*WA*F;
AXP = (abs(AX)+AX)/2;
AXN = (abs(AX)-AX)/2;
AAFP = (abs(AAF)+AAF)/2;
AAFN = (abs(AAF)-AAF)/2;
las = 1e6;
F1 = (AXP+alpha*F*W1+AAFN+las*O1'*O2);%   
F2 = (AXN+alpha*F*Di+AAFP+las*O1'*O1*F);%
F = F.*sqrt(F1./max(F2,1e-10));


AF= A*F;

%% loss

loss1(iter) = norm(S*W*X-att*F,'fro');
loss2 = lambda*rank(S);%

% Update Y2
leq2 = S-C;
Y2 = Y2 + mu*leq2;
mu = min(max_mu,mu*rho);



% lossn = loss1+loss2;
% disp(norm(S-C,'inf'))
% disp(abs(lossn-loss))
% loss = lossn;
        

end


dist     =  1 - (pdist2(X_te'*W', att_new', 'cosine'));
dist     = zscore(dist);
HITK     = 1;
Y_hit5   = zeros(size(dist,1),HITK);
for i    = 1:size(dist,1)
    [sort_dist_i, I] = sort(dist(i,:),'descend');
    Y_hit5(i,:) = te_cl_id(I(1:HITK));
end

n=0;
for i  = 1:size(dist,1)
    if ismember(test_labels_cub(i),Y_hit5(i,:))
        n = n + 1;
    end
end
zsl_accuracy_unseen = n/size(dist,1);

seenclasses = unique(test_seen_labels,'stable');
A = att(:, seenclasses);
X_tes = test_seen_X;
tests_labels_cub = [test_seen_labels];%; test_seen_labels
tes_cl_id = unique(tests_labels_cub,'stable');

dist2     =  1 - (pdist2(X_tes*W', A', 'cosine'));
dist2     = zscore(dist2);
HITK     = 1;
Y_hit5   = zeros(size(dist2,1),HITK);
for i    = 1:size(dist2,1)
    [sort_dist_i, I] = sort(dist2(i,:),'descend');
    Y_hit5(i,:) = tes_cl_id(I(1:HITK));
end

n=0;
for i  = 1:size(dist2,1)
    if ismember(test_seen_labels(i),Y_hit5(i,:))
        n = n + 1;
    end
end
zsl_accuracy_seen = n/size(dist2,1);

fprintf('AWA GZSL_U accuracy: %.1f%%\n', zsl_accuracy_unseen*100);
fprintf('AWA GZSL_S accuracy: %.1f%%\n', zsl_accuracy_seen*100);
h=2*zsl_accuracy_seen*zsl_accuracy_unseen/(zsl_accuracy_seen+zsl_accuracy_unseen)



