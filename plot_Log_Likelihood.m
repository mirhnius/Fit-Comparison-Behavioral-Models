function plot_Log_Likelihood(max_log)
    n = size(max_log(:,1));
    n = n(1,1);
    figure()
    for i=1:n
        subplot(floor(n/5),floor(n/4),i);
        plot(1:n,max_log(i,:))
        title(num2str(i))
        %title(['max log_likelihood for subject ' num2str(i)])
    end
end