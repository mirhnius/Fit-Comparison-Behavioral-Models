function log_prob = llike_eg(x, args)
    alpha = x(1,1);
    e = x(1,2);
    
    actions = args(:,1);
    rewards = args(:,2);
    
    Q = [0,0];
    log_prob_actions = zeros(length(actions),1);
    
    for i = 1:length(actions) 
        [~,idx] = max(Q);
        if idx == actions(i)
            log_prob_actions(i, 1) = log(1-e/2);
        else
           log_prob_actions(i, 1) = log(e/2); 
        end
        Q(actions(i)) = Q(actions(i)) + alpha * (rewards(i) - Q(actions(i)));     
    end
    
  log_prob = -sum(log_prob_actions);          
end