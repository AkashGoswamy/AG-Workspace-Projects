% ============================================================
% COMPARISON: Big-M vs Two-Phase Simplex Method
% Problem: Max Z = x1 + 2x2 + 3x3 - x4
% s.t.  x1 + 2x2 + 3x3       = 15
%       2x1 + x2 + 5x3       = 20
%       x1 + 2x2 +  x3 + x4  = 10
%       x1,x2,x3,x4 >= 0
% ============================================================

function compare_methods()
    fprintf('\n========================================\n');
    fprintf('  Big-M vs Two-Phase Method Comparison\n');
    fprintf('========================================\n\n');

    N = 1000;

    t_bigm = zeros(N,1);
    for k = 1:N
        tic;
        [sol_bigm, Z_bigm, iters_bigm, pivots_bigm, tableaux_bigm] = bigM_method();
        t_bigm(k) = toc;
    end

    t_tp = zeros(N,1);
    for k = 1:N
        tic;
        [sol_tp, Z_tp, iters_tp, pivots_tp, tableaux_tp] = twophase_method();
        t_tp(k) = toc;
    end

    fprintf('%-30s %-15s %-15s\n', 'Metric', 'Big-M', 'Two-Phase');
    fprintf('%s\n', repmat('-',1,62));
    fprintf('%-30s %-15.6f %-15.6f\n', 'Avg Runtime (seconds)',  mean(t_bigm), mean(t_tp));
    fprintf('%-30s %-15.6f %-15.6f\n', 'Min Runtime (seconds)',  min(t_bigm),  min(t_tp));
    fprintf('%-30s %-15.6f %-15.6f\n', 'Max Runtime (seconds)',  max(t_bigm),  max(t_tp));
    fprintf('%-30s %-15d %-15d\n',     'Total Iterations',       iters_bigm,   iters_tp);
    fprintf('%-30s %-15d %-15d\n',     'Total Pivot Operations', pivots_bigm,  pivots_tp);
    fprintf('%-30s %-15d %-15d\n',     'No. of Tableaux',        tableaux_bigm,tableaux_tp);
    fprintf('%-30s %-15.4f %-15.4f\n', 'Optimal Z',              Z_bigm,       Z_tp);
    fprintf('%-30s %-15.4f %-15.4f\n', 'x1', sol_bigm(1), sol_tp(1));
    fprintf('%-30s %-15.4f %-15.4f\n', 'x2', sol_bigm(2), sol_tp(2));
    fprintf('%-30s %-15.4f %-15.4f\n', 'x3', sol_bigm(3), sol_tp(3));
    fprintf('%-30s %-15.4f %-15.4f\n', 'x4', sol_bigm(4), sol_tp(4));
    fprintf('%s\n', repmat('-',1,62));

    figure('Name','Big-M vs Two-Phase','Position',[100 100 1100 700]);

    subplot(2,3,1);
    boxplot([t_bigm*1e6, t_tp*1e6], {'Big-M','Two-Phase'});
    ylabel('Time (us)'); title('Runtime Distribution'); grid on;

    subplot(2,3,2);
    b = bar([mean(t_bigm)*1e6, mean(t_tp)*1e6],'FaceColor','flat');
    b.CData = [0.2 0.5 0.8; 0.9 0.4 0.2];
    set(gca,'XTickLabel',{'Big-M','Two-Phase'});
    ylabel('Mean Time (us)'); title('Mean Runtime'); grid on;

    subplot(2,3,3);
    cats = categorical({'Iterations','Pivots','Tableaux'});
    data = [iters_bigm, pivots_bigm, tableaux_bigm; iters_tp, pivots_tp, tableaux_tp];
    br = bar(cats, data');
    br(1).FaceColor = [0.2 0.5 0.8];
    br(2).FaceColor = [0.9 0.4 0.2];
    legend('Big-M','Two-Phase','Location','northwest');
    title('Iterations, Pivots & Tableaux'); grid on;

    subplot(2,3,4);
    vars = categorical({'x1','x2','x3','x4'});
    b2 = bar(vars, [sol_bigm'; sol_tp(1:4)']');
    b2(1).FaceColor = [0.2 0.5 0.8];
    b2(2).FaceColor = [0.9 0.4 0.2];
    legend('Big-M','Two-Phase'); title('Solution Variables');
    ylabel('Value'); grid on;

    subplot(2,3,5);
    histogram(t_bigm*1e6,30,'FaceColor',[0.2 0.5 0.8],'FaceAlpha',0.6); hold on;
    histogram(t_tp*1e6,  30,'FaceColor',[0.9 0.4 0.2],'FaceAlpha',0.6);
    xlabel('Time (us)'); ylabel('Frequency');
    title('Runtime Histogram'); legend('Big-M','Two-Phase'); grid on;

    subplot(2,3,6);
    plot(cumsum(sort(t_bigm))*1e6,'b-','LineWidth',2); hold on;
    plot(cumsum(sort(t_tp))*1e6,  'r-','LineWidth',2);
    xlabel('Run #'); ylabel('Cumulative Time (us)');
    title('Cumulative Runtime (sorted)');
    legend('Big-M','Two-Phase'); grid on;

    sgtitle('Big-M vs Two-Phase: Performance Comparison (1000 runs)','FontSize',13,'FontWeight','bold');
    saveas(gcf,'comparison_plot.png');
    fprintf('\nPlot saved.\n');
end

function [sol, Zopt, total_iters, total_pivots, n_tableaux] = bigM_method()
    M_val = 1e6;
    T = [1 2 3 0 1 0 15;
         2 1 5 0 0 1 20;
         1 2 1 1 0 0 10;
         0 0 0 0 0 0  0];
    cobj = [-1,-2,-3,1,M_val,M_val,0];
    T(4,:) = cobj + M_val*T(1,:) + M_val*T(2,:);
    basis = [5,6,4];
    total_iters=0; total_pivots=0; n_tableaux=1;
    for iter=1:20
        [minval,pcol] = min(T(4,1:6));
        if minval >= -1e-8; break; end
        ratios = inf(3,1);
        for i=1:3
            if T(i,pcol)>1e-9; ratios(i)=T(i,end)/T(i,pcol); end
        end
        [~,prow]=min(ratios);
        T(prow,:)=T(prow,:)/T(prow,pcol);
        for i=1:4
            if i~=prow; T(i,:)=T(i,:)-T(i,pcol)*T(prow,:); end
        end
        basis(prow)=pcol;
        total_iters=total_iters+1;
        total_pivots=total_pivots+1;
        n_tableaux=n_tableaux+1;
    end
    sol=zeros(4,1);
    for v=1:4
        for r=1:3
            if basis(r)==v; sol(v)=T(r,end); end
        end
    end
    Zopt=-T(4,end);
end

function [sol, Zopt, total_iters, total_pivots, n_tableaux] = twophase_method()
    total_iters=0; total_pivots=0; n_tableaux=1;
    T=[1 2 3 0 1 0 15;
       2 1 5 0 0 1 20;
       1 2 1 1 0 0 10;
       0 0 0 0 0 0  0];
    T(4,:)=[0,0,0,0,1,1,0]+T(1,:)+T(2,:);
    basis=[5,6,4];
    for iter=1:20
        [minval,pcol]=min(T(4,1:6));
        if minval>=-1e-8; break; end
        ratios=inf(3,1);
        for i=1:3
            if T(i,pcol)>1e-9; ratios(i)=T(i,end)/T(i,pcol); end
        end
        [~,prow]=min(ratios);
        T(prow,:)=T(prow,:)/T(prow,pcol);
        for i=1:4
            if i~=prow; T(i,:)=T(i,:)-T(i,pcol)*T(prow,:); end
        end
        basis(prow)=pcol;
        total_iters=total_iters+1; total_pivots=total_pivots+1; n_tableaux=n_tableaux+1;
    end
    T2=T(1:3,[1,2,3,4,7]);
    basis2=basis;
    cZ=[-1,-2,-3,1,0];
    zrow2=cZ;
    for r=1:3
        b=basis2(r);
        if b<=4; zrow2=zrow2-cZ(b)*T2(r,:); end
    end
    T2=[T2;zrow2];
    n_tableaux=n_tableaux+1;
    for iter=1:20
        [minval,pcol]=min(T2(4,1:4));
        if minval>=-1e-8; break; end
        ratios=inf(3,1);
        for i=1:3
            if T2(i,pcol)>1e-9; ratios(i)=T2(i,end)/T2(i,pcol); end
        end
        [~,prow]=min(ratios);
        T2(prow,:)=T2(prow,:)/T2(prow,pcol);
        for i=1:4
            if i~=prow; T2(i,:)=T2(i,:)-T2(i,pcol)*T2(prow,:); end
        end
        basis2(prow)=pcol;
        total_iters=total_iters+1; total_pivots=total_pivots+1; n_tableaux=n_tableaux+1;
    end
    sol=zeros(4,1);
    for v=1:4
        for r=1:3
            if basis2(r)==v; sol(v)=T2(r,end); end
        end
    end
    Zopt=-T2(4,end);
end