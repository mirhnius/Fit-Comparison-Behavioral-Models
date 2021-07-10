T = table2array(readtable('rawdata.csv'));
itr = 20;
min_log = inf* ones(20,itr);
x_ = zeros(20,1);

for i=1:20
    args = T(100*(i-1)+1:100*i,3:4);
    min_ =  inf;
    for j = 1:itr
        x0 = rand;
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        lb = 0;
        ub = 1;
        fun = @(x)bernoli(x,args);
        x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
        this_turn = fun(x);
         if min_ > this_turn
            min_log(i,j) = this_turn;
            x_(i,:) = x;
        else
            min_log(i,j) = min_;
        end
        min_ = min(min_log(i,:));
    end
    
end

plot_Log_Likelihood(-min_log)

figure()
histogram(x_(:,1),5)
title('Distribution of b')

b_mean = mean(x_);
b_sd = std(x_);
minlog_random = -min_log(:,end);
%%
T = table2array(readtable('rawdata.csv'));
min_log = inf* ones(20,itr);
x_ = zeros(20,2);

for i=1:20
    args = T(100*(i-1)+1:100*i,3:4);
    min_ = inf;
    best_x = inf;
    for j = 1:itr
        x0 = [rand, 5 * rand];
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        lb = [0, 0];
        ub = [1, 5];
        fun = @(x)llike_td(x,args);
        x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
        this_turn = fun(x);
        if min_ >  this_turn
            min_log(i,j) = this_turn;
            x_(i,:) = x;
        else
            min_log(i,j) = min_;
        end
        min_ = min(min_log(i,:));
        
    end
end

plot_Log_Likelihood(-min_log)

titles = {'alpha', 'beta'};
for i=1:2
    figure()
    histogram(x_(:,i),5)
    title(['Distribution of ' titles(i)]);
end

alpha_mean = mean(x_(:,1));
alpha_sd = std(x_(:,1));

beta_mean = mean(x_(:,2));
beta_sd = std(x_(:,2));
minlog_td = -min_log(:,end);
%%
T = table2array(readtable('rawdata.csv'));
min_log = inf* ones(20,itr);
x_ = zeros(20,1);

for i=1:20
    args = T(100*(i-1)+1:100*i,3:4);
    min_ = inf;
    best_x = inf;
    for j = 1:itr
        x0 = rand;
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        lb = 0;
        ub = 1;
        fun = @(x)noisy_win_stay_lose_shift(x,args);
        x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
        this_turn = fun(x);
        if min_ > this_turn
            min_log(i,j) = this_turn;
            x_(i,:) = x;
        else
            min_log(i,j) = min_;
        end
        min_ = min(min_log(i,:));
        
    end
end

plot_Log_Likelihood(-min_log)

figure()
histogram(x_(:,1),5)
title('Distribution of epsilon')

epsilon_mean = mean(x_(:,1));
epsilon_sd = std(x_(:,1));

minlog_win_lose = -min_log(:,end);

%%
T = table2array(readtable('rawdata.csv'));
min_log = inf* ones(20,itr);
x_ = zeros(20,4);

for i=1:20
    args = T(100*(i-1)+1:100*i,3:4);
    min_ = inf;
    best_x = inf;
    for j = 1:itr
        x0 = [rand, 5 * rand, rand, rand];
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        lb = [0, 0, 0, 0];
        ub = [1, 5, 1, 5];
        fun = @(x)extended(x,args);
        x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
        this_turn = fun(x);
        if min_ > this_turn
            min_log(i,j) = this_turn;
            x_(i,:) = x;
        else
            min_log(i,j) = min_;
        end
        min_ = min(min_log(i,:));
        
    end
end

plot_Log_Likelihood(-min_log)

titles = {'alpha', 'beta', 'alpha_k', 'beta_k'};
for i=1:4
    figure()
    histogram(x_(:,i),5)
    title(['Distribution of ' titles{i}]);
end

alpha_exd_mean = mean(x_(:,1));
alpha_ex_sd = std(x_(:,1));

beta_ex_mean = mean(x_(:,2));
beta_ex_sd = std(x_(:,2));

alpha_exd_k_mean = mean(x_(:,3));
alpha_ex_k_sd = std(x_(:,3));

beta_ex_k_mean = mean(x_(:,4));
beta_ex_k_sd = std(x_(:,4));

minlog_td_ex = -min_log(:,end);
%%
T = table2array(readtable('rawdata.csv'));
min_log = inf* ones(20,itr);
x_ = zeros(20,2);

for i=1:20
    args = T(100*(i-1)+1:100*i,3:4);
    min_ = inf;
    best_x = inf;
    for j = 1:itr
        x0 = [rand, rand];
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        lb = [0, 0];
        ub = [1, 1];
        fun = @(x)llike_eg(x,args);
        x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub);
        this_turn = fun(x);
        if min_ > this_turn
            min_log(i,j) = this_turn;
            x_(i,:) = x;
        else
            min_log(i,j) = min_;
        end
        min_ = min(min_log(i,:));
        
    end
end

plot_Log_Likelihood(-min_log)

titles = {'alpha', 'epsilon'};
for i=1:2
    figure()
    histogram(x_(:,i),5)
    title(['Distribution of ' titles{i}]);
end

alpha_mean_eg = mean(x_(:,1));
alpha_sd_eg = std(x_(:,1));

e_mean = mean(x_(:,2));
e_sd = std(x_(:,2));
minlog_eg = -min_log(:,end);

%%
logL = [minlog_random, minlog_win_lose, minlog_td, minlog_eg, minlog_td_ex];
AIC = zeros(20,5);
BIC = zeros(20,5);
for i=1:20
    [aic,bic] = aicbic(logL(i,:), [1; 2; 2; 2; 4], 20*ones(5,1));
    AIC(i,:) = aic;
    BIC(i,:) = bic;
end

[~,idx_aic] = sort(AIC,2);
[~,idx_bic] = sort(BIC,2);

best_aic = unique(idx_aic(:,1));
best_bic = unique(idx_bic(:,1));

ratio_aic = zeros(length(best_aic),1);
j = 1;
for i=best_aic'
    ratio_aic(j) = length(find(idx_aic(:,1),i))/20;
    j = j + 1;
end
aic_ = [best_aic, ratio_aic];

ratio_bic = zeros(length(best_bic),1);
j = 1;
for i=best_aic'
    ratio_bic(j) = length(find(idx_bic(:,1),i))/20;
    j = j + 1;
end
bic_ = [best_bic, ratio_bic];
