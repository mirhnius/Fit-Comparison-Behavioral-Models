function log_prob = llike_td(x, args)
    alpha = x(1,1);
    beta = x(1,2);
    
    actions = args(:,1);
    rewards = args(:,2);
    
    Q = [0,0];
    log_prob_actions = zeros(length(actions),1);
    
    for i = 1:length(actions) 
        Q_ = Q * beta;
        log_prob_action = Q_ - logsumexp(Q_,2);
        log_prob_actions(i, 1) = log_prob_action(actions(i));
        Q(actions(i)) = Q(actions(i)) + alpha * (rewards(i) - Q(actions(i)));     
    end
    
  log_prob = -sum(log_prob_actions);          
end