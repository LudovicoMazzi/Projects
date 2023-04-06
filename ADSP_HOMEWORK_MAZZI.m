%% 1.1 - Noise Generation
clear all;
close all;
clc;
M = 4;
MC = 10; % Montecarlo Iterations
MSE_H = zeros(10,10,MC);

for iter = 1:MC % iterate through Montecarlo
    i = 1; 
    u = randn([M,200]); % returns a Mx200 matrix containing "random" values from normal distrib
    for L = floor(linspace(10,200,10)) % iterate through L (it is the K, i call it L because K is the memory)
        k = 1;
        for rho = linspace(0, 0.99,10) %iterate through values of rho
            Cw_true = toeplitz([1,rho,rho,rho]); % true Covariance Matrix, it makes like sum of diags in a toeplitz form
            w = chol(Cw_true, 'lower')*u(:,1:L); % noise, chol gives a lower triangular matrix drawn from Cw_true, rest will be no used 0
            Cw_est = cov(w.'); % estimate of the Covariance Matrix (we put the transpose since for cov each column is a variable, each row observation
            MSE_H(k, i, iter) = mean(mean(Cw_true - Cw_est).^2); % .^ since element wise, mean of mean compute the mean of the vector of the mean
            k = k + 1; % to increment index associated to rho
        end
        i = i + 1; % to increment index associated to L 
    end
end
MSE_H = mean(MSE_H,3); % mean for each combination of x and y of MSE
figure()
bar3(MSE_H);xlabel('L');ylabel('\rho');zlabel(' MSE') % 3D PLOT


figure() 
for l = 1:10 % iterate through L, to have different lines of MSE v. rho wrt L
    plot(linspace(0, 0.99, 10), MSE_H(:,l))
    hold on
end
title('MSE v \rho') % increasing rho = increasing MSE
legend({'L = 10', 'L = 31', 'L = 52', 'L = 73', 'L = 94', 'L = 115', 'L = 136', 'L = 157', 'L = 178', 'L = 200'}, 'Location', 'best', 'NumColumns', 4)
xlabel('\rho');
ylabel('MSE')


figure()
for rho = 1:10 % iterate through rho, to have different lines of MSE v. L wrt rho
plot(floor(linspace(10,200,10)), MSE_H(rho, :))
hold on;
end
title('MSE v L') % increasing L = lowering MSE
legend({'\rho = 0', '\rho = 0.11', '\rho = 0.22', '\rho = 0.33', '\rho = 0.44', '\rho = 0.55', '\rho = 0.66', '\rho = 0.77', '\rho = 0.88', '\rho = 0.99'}, 'Location', 'best', 'NumColumns', 4)
xlabel('L');
ylabel('MSE');

%% 1.2(i) - MIMO Estimation: Memoryless Filter
MC = 25; % montecarlo 
K = 1; % memoryless
M = 4; % first dimensions
N = 4; % second dimensions
SNR = -10:2:30; % snr in db
SNR_lin = 10.^(SNR/10); % snr linear
sigma_x = 1;
rho = 0.1;
C_true = toeplitz([1, rho, rho, rho]);
MLE = zeros(21, 5, 5, MC); % 5 and 5 is for the iterations on alpha and q, 21 for snr
CRB = zeros(21,5,5,MC);

for iter = 1:MC %Montecarlo iterations
    sigma_index = 1;
    for sigma_w = 1./SNR_lin 
        C = sigma_w*C_true; 
        u_all = randn(M,200); 
        w_all = chol(C, 'lower')*u_all;
        alpha_index = 1;
        for alpha = linspace(0, 0.99, 5)
            x_all = sigma_x*randn(N,200); 
            h = toeplitz([alpha^0, alpha^1, alpha^2, alpha^3]);
            h_reshape = reshape(h,1,[])'; %MNx1 reshaped in vector form
            i_q = 1;
            for Q = floor(linspace(5,50,5))
                w = w_all(:,1:Q);
                x = x_all(:,1:Q);
                x_kron = kron(eye(N), x'); %eye(N) since M=N, it returns (MQxMN) 
                C_kron = kron(eye(Q), C); 
                w_kron = reshape(w.', 1, [])'; %vector shape MQx1
                y_kron = x_kron*h_reshape + w_kron; %MQx1 obviously
                h_est_mle = inv(x_kron.'*inv(C_kron)*x_kron)*x_kron.'*inv(C_kron)*y_kron; 
                CRB(sigma_index, i_q, alpha_index, iter) = trace(inv(x_kron.'*inv(C_kron)*x_kron))/16; 
                MLE(sigma_index, i_q, alpha_index, iter) = mean((h_reshape - h_est_mle).^2);
                i_q = i_q + 1;
            end
            alpha_index = alpha_index + 1;
        end
        sigma_index = sigma_index + 1;
    end
end
MLE = mean(MLE, 4);
CRB = mean(CRB, 4);

%PLOTS
figure()
alpha_plot = 5;
cmap = hsv(5);
for i = 1:5
    loglog(SNR_lin, (MLE(:, i, alpha_plot)), 'color', cmap(i,:))
    hold on;
    loglog(SNR_lin, (CRB(:, i , alpha_plot)), '--', 'color', cmap(i,:))
end
hold off;
title('MSE v. SNR (\alpha=0.99, Q varies)')
grid on;
xlabel('SNR(dB)')
ylabel('MSE(dB)')
legend('MSE(Q=5)', 'CRB(Q=5)', 'MSE(Q=16)', 'CRB(Q=16)', 'MSE(Q=27)', 'CRB(Q=27)', 'MSE(Q=38)', 'CRB(Q=38)', 'MSE(Q=50)', 'CRB(Q=50)')



figure()
Q_plot = 5;
alpha_str = linspace(0, 0.99, 5);
for i = 1:5
    loglog(SNR_lin, (MLE(:,Q_plot, i)), 'color', cmap(i,:))
    hold on;
    loglog(SNR_lin, CRB(:, Q_plot, i), '--', 'color', cmap(i,:))
end
hold off;
title('MSE v. SNR (Q=50, \alpha varies)')
grid on;
xlabel('SNR(dB)')
ylabel('MSE(dB)')
legend('MLE (\alpha=0)', 'CRB (\alpha=0)', 'MLE (\alpha=0.2)', 'CRB (\alpha=0.2)', 'MLE (\alpha=0.5)', 'CRB (\alpha=0.5)', 'MLE (\alpha=0.75)', 'CRB (\alpha=0.75)', 'MLE (\alpha=0.99)', 'CRB (\alpha=0.99)')


%% 1.2(ii) - MIMO Estimation: Memory Filter

MC = 25; % Montecarlo Iterations
K = 4;
M = 4;
N = 4;
SNR = -10:2:30;
SNR_lin = 10.^(SNR/10);
rho = 0.1; 
beta = [0.9, 0.5, 0.1];
sigma_x = 1;
MLE = zeros(21,5,5,3,MC); % 3 for beta
CRB = zeros(21,5,5,3,MC);

for iter = 1:MC
    sigma_i = 1; % index for sigma
    for sigma_w = 1./SNR_lin
        alpha_i = 1; % index for alpha
        C_true = toeplitz([1, rho, rho, rho]);
        C = sigma_w*C_true; 
        for alpha = linspace(0, .99, 5)
            for beta_i = 1:3
                h = zeros(M,N,K);
                for i = 1:K
                    h(:,:,i) = (beta(beta_i)^(i-1))*toeplitz([alpha^0, alpha^1, alpha^2, alpha^3]);
                end
                h_reshape = reshape(h, 1, [])'; % 64x1 because (MxNxK)x1 in vector form
                q = 1; 
                for Q = floor(linspace(15,50,5))
                    u = randn(M*(Q+K-1),1);
                    C_kron = kron(C,eye(Q+K-1));
                    w_kron = chol(C_kron, 'lower')*u; %now u changes from time to time
                    x_mem = zeros(Q+K-1, N*K); % (Q+K-1)x(NxK)
                    for n = 1:N
                        x = sigma_x*randn(Q,N);
                        for i = 1:K
                            x_mem(i:Q+i-1, i+N*(n-1)) = x(:,n);
                        end
                    end
                    x_kron = kron(eye(4), x_mem); % (Mx(Q+K-1))x(MxNxK) = (212x64)
                    y = x_kron*h_reshape + w_kron; %(Mx(Q+K-1))x1
                    h_est_mle = inv(x_kron'*inv(C_kron)*x_kron)*x_kron'*inv(C_kron)*y; % MLE est
                    MLE(sigma_i, q, alpha_i, beta_i,iter) = mean((h_reshape - h_est_mle).^2); % 21 because of sigma which counts for SNR
                    CRB(sigma_i, q, alpha_i, beta_i,iter) = trace(inv(x_kron.'*inv(C_kron)*x_kron))/64;
                    q = q + 1;
                end
            end
            alpha_i = alpha_i + 1;
        end
        sigma_i = sigma_i + 1;
    end
end
MLE = mean(MLE, 5);
CRB = mean(CRB, 5);

%PLOTS
figure()
Q_plot = 5;
alpha_plot = 5;
cmap = hsv(3);
for i = 1:3
    loglog(SNR_lin, (MLE(:, Q_plot, alpha_plot, i)), 'color', cmap(i,:))
    hold on;
    loglog(SNR_lin, (CRB(:, Q_plot, alpha_plot, i)),'--', 'color', cmap(i,:))
end
hold off;
title('MSE v SNR -- Q=50, \alpha=0.99')
legend('MLE (\beta=0.9)', 'CRB (\beta=0.9)', 'MLE (\beta=0.5)', 'CRB (\beta=0.5)', 'MLE (\beta=0.1)', 'CRB (\beta=0.1)')
grid on;

figure()
Q_plot = 5;
beta_plot = 1;
cmap = hsv(5);
for i = 1:5
    loglog(SNR_lin, (MLE(:,Q_plot, i, beta_plot)), 'color', cmap(i, :))
    hold on;
    loglog(SNR_lin, (CRB(:,Q_plot, i, beta_plot)), '--', 'color', cmap(i, :))
end
hold off;
title('MSE v SNR -- Q=50, \beta=0.9')
grid on;
xlabel('SNR(dB)')
ylabel('MSE(dB)')
legend('MLE (\alpha=0)', 'CRB (\alpha=0)', 'MLE (\alpha=0.2)', 'CRB (\alpha=0.2)', 'MLE (\alpha=0.5)', 'CRB (\alpha=0.5)', 'MLE (\alpha=0.75)', 'CRB (\alpha=0.75)', 'MLE (\alpha=0.99)', 'CRB (\alpha=0.99)')

figure()
alpha_plot = 5;
beta_plot = 1;
cmap = hsv(5);
for i = 1:5
    loglog(SNR_lin, (MLE(:, i, alpha_plot, beta_plot)), 'color', cmap(i,:))
    hold on;
    loglog(SNR_lin, (CRB(:, i, alpha_plot, beta_plot)), '--', 'color', cmap(i, :))
end
hold off;
title('MSE v SNR -- \alpha=0.99, \beta=0.9')
grid on;
xlabel('SNR(dB)')
ylabel('MSE(dB)')
legend('MLE (Q=15)', 'CRB (Q=15)', 'MLE (Q=23)', 'CRB (Q=23)', 'MLE (Q=32)', 'CRB (Q=32)', 'MLE (Q=41)', 'CRB (Q=41)', 'MLE (Q=50)', 'CRB (Q=50)')

%% 1.3 - MIMO Deconvolution
MC = 25; % Montecarlo Iterations
K = 1; % Memoryless (it's like the third dimensions)
M = 4; % first dimensions of filter response matrix
N = 4; % second dimension of filter response matrix
SNR = -10:2:30; % SNR in dB (21 values)
SNR_lin = 10.^(SNR/10); % SNR in linear
sigma_x = 1; % we put it to 1 for simplicity
rho = 0.1; % value wanted by professor
alpha = 0.5; % value wanted by professor (coupling effect)
h = toeplitz([alpha^0, alpha^1, alpha^2, alpha^3]);
h_reshape = reshape(h, 1, [])'; % it puts the matrix h in a vector, column vector since it is transpose
C_true = toeplitz([1, rho, rho, rho]);
MLE = zeros(21,20, MC);
MMSE = zeros(21,20,MC);
MLE_H = zeros(21,20, MC);
MSE_H = zeros(21,20,MC);
for iter = 1:MC
    sigma_index = 1;
    for sigma_w = 1./SNR_lin % by definition
        C = sigma_w*C_true;
        u_al = randn(M,200);
        w_al = chol(C,'lower')*u_al;
        x_al = sigma_x*randn(N,200);
        y_al = h*x_al + w_al;

        q_index = 1;
        for Q = floor(linspace(5, 199, 20))  %number of pilots samples known both at the receiver and the transmitter
            w = w_al(:, 1:Q);
            x = x_al(:, 1:Q); % used as pilot
            x_rest = x_al(:, Q+1:end); % in the end there is always at least one information sample
            x_kron = kron(eye(N), x'); % to reshape it 
            C_kron = kron(eye(Q),C); % reshape of sigma_w*C_true
            C_eq_h = kron(eye(N), C);
            w_kron = reshape(w.', 1, [])';
            y_kron = x_kron*h_reshape + w_kron;
            C_x = cov(x');
            h_est_mle = inv(x_kron'*inv(C_kron)*x_kron)*x_kron'*inv(C_kron)*y_kron;
            y_rest = y_al(:,Q+1:end);
            h_rest = reshape(h_est_mle, 4, 4);
            x_est_mle = inv(h_rest'*inv(C)*h_rest)*h_rest'*inv(C)*y_rest; % mle formula
            x_est_mmse = inv(h_rest'*inv(C)*h_rest + inv(C_x))*h_rest'*inv(C)*y_rest; % mmse formula

            MLE(sigma_index, q_index, iter) = mean(mean((x_rest - x_est_mle).^2));
            MMSE(sigma_index, q_index, iter) = mean(mean((x_rest - x_est_mmse).^2));
            MLE_H(sigma_index, q_index, iter) = mean((h_est_mle - h_reshape).^2);
            q_index = q_index + 1; % iterate through q pilots
        end
        sigma_index = sigma_index + 1; %iterate in 1/SNR 
    end

end
%to mean through iter
MLE = mean(MLE, 3);
MMSE = mean(MMSE, 3);
MLE_H = mean(MLE_H, 3);

%METRIC FOR Q
cost = zeros(21,20); % cost function dimension 21 for snr, 20 for q
P = 200; %length of input signal, when P = Q we are transmitting only pilot samples
sigma_index = 1; % index for sigma
MSE_dB = 10*log10(MMSE);
normal = mean(floor(linspace(5, P, 20)))/(mean(mean(MMSE))); % normalization
for sigma_tx = SNR_lin
    q_index = 1;
    signal_p = sigma_tx * randn(4,P);
    power_p = 10*log10(mean(mean(signal_p.^2)));
    for Q = floor(linspace(5, P, 20))
        signal_q = signal_p(:,1:Q); 
        signal_info = signal_p(:,Q+1:end);
        power_q = mean(mean(signal_q.^2)); % mean over mean to obtain a value
        power_info = mean(mean(signal_info.^2)); 
        cost(sigma_index, q_index) = normal*MMSE(sigma_index, q_index) + 0.5*Q; % it has to be put in dB to be confronted with the SNR
        q_index = q_index + 1;
    end
    sigma_index = sigma_index + 1; %increment of index of sigma
end

Q = floor(linspace(5, P, 20));

%PLOTS
figure()
for i = [10,13,15,20]
    plot(Q, 10*log10((cost(i,:))))
    hold on;
end
title('Loss Function v. Q')
xlabel('Q');
ylabel('Loss (dB)')
legend('SNR = 8dB', 'SNR = 14dB', 'SNR = 18dB', 'SNR = 28dB')

figure()
color_i = 1;
color = 'rbgm';
for i = floor(linspace(1,20,4))
    loglog(SNR_lin, (MLE(:,i)), 'color', color(color_i))
    hold on;
    loglog(SNR_lin, (MMSE(:,i)), '--', 'color', color(color_i))
    color_i = color_i + 1;
end
hold off;
title('MMSE v. MLE')
grid on
xlabel('SNR(dB)')
ylabel('MSE(dB)')
legend('MLE (Q=5)', 'MMSE (Q=5)', 'MLE (Q=66)', 'MMSE (Q=66)', 'MLE (Q=127)', 'MMSE (Q=127)', 'MLE (Q=199)', 'MMSE (Q=199)')

