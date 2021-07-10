function log_prob = bernoli(x,args)
    
    actions = args(:,1);

    log_prob_actions = zeros(length(actions),1);
    
    for i = 1:length(actions)
        if actions(i) == 1
            log_prob_actions(i, 1) = log(1-x);
        else
            log_prob_actions(i, 1) = log(x);
        end
    end
    log_prob = -sum(log_prob_actions);
end