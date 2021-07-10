function log_prob = extended(x, args)
    alpha = x(1,1);
    beta = x(1,2);
    alpha_k = x(1,3);
    beta_k = x(1,4);
    
    actions = args(:,1);
    rewards = args(:,2);
    
    Q = [0,0];
    CK = [0,0];
    log_prob_actions = zeros(length(actions),1);
    
    for i = 1:length(actions) 
        Q_ = Q * beta + beta_k * CK;
        log_prob_action = Q_ - logsumexp(Q_,2);
        log_prob_actions(i, 1) = log_prob_action(actions(i));
        
        Q(actions(i)) = Q(actions(i)) + alpha * (rewards(i) - Q(actions(i)));
        
        CK = (1-alpha_k) * CK;
        CK(actions(i)) = CK(actions(i)) + alpha_k;
    end
    
  log_prob = -sum(log_prob_actions);   
end