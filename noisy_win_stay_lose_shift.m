function log_prob = noisy_win_stay_lose_shift(epsilon,args)
    last_reward = nan;
    last_action = nan;
    
    actions = args(:,1);
    rewards = args(:,2);
    
    log_prob_actions = zeros(length(actions),1);
    for i = 1:length(actions)
        if isnan(last_action)
            p = [0.5, 0.5];
        else 
            if last_reward == 1
                p = (epsilon/2) * [1, 1];
                p(last_action) = 1 - epsilon/2;
            else
                p = (1-epsilon/2) * [1, 1];
                p(last_action) = epsilon/2;
            end
            
        end
        log_prob_actions(i,1) = log(p(actions(i)));
        
        last_reward = rewards;
        last_action = actions(i);
    end
    log_prob = -sum(log_prob_actions);
end







% def llik_td(x, *args):
%     # Extract the arguments as they are passed by scipy.optimize.minimize
%     alpha, beta = x
%     actions, rewards = args
% 
%     # Initialize values
%     Q = np.array([.5, .5])
%     log_prob_actions = np.zeros(len(actions))
% 
%     for t, (a, r) in enumerate(zip(actions,rewards)):
%         # Apply the softmax transformation
%         Q_ = Q * beta
%         log_prob_action = Q_ - scipy.special.logsumexp(Q_)
% 
%         # Store the log probability of the observed action
%         log_prob_actions[t] = log_prob_action[a]
% 
%         # Update the Q values for the next trial
%         Q[a] = Q[a] + alpha * (r - Q[a])
% 
%     # Return the negative log likelihood of all observed actions
%     return -np.sum(log_prob_actions[1:])