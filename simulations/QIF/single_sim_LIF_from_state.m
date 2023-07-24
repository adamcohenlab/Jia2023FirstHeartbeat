function single_sim_LIF_from_state(input_path, t_end, n_repeats, output_path)
    tic
    global dt rest sigma lambda
    function y = func(x, t)
       y = 0;
    end
    
    r = [];
    s = [];
    y_end = [];
    t_total = [];
    last_peak = [];
    n_peaks = [];
    isi_mu = [];
    isi_std = [];
    dt_set = [];

    load(input_path, "r", "s", "y_end", "t_total", "dt_set", "last_peak", "n_peaks", "isi_mu", "isi_std");
    lambda = 1;
    l = 1;
    rest = r;
    sigma = s;
    dt = dt_set;
    disp(last_peak);
    disp(isi_mu);
    disp(isi_std);
    disp(n_peaks);
    
    y0 = y_end;
    for i = 1:n_repeats
        [t,y] = onecell_euler(@onecell_de_white, [0, t_end], dt, y0, @func);
        locs = find(y > 5.5);
        all_peaks = [t(locs)+(i-1)*t_end+t_total];
        
        if last_peak == -1
            isi = diff(all_peaks);
        else
            isi = diff([last_peak all_peaks]);
        end
        isi_mu = (isi_mu*(n_peaks-1) + sum(isi))/(max(n_peaks-1,0) + length(isi));
        isi_std = ((isi_std^2*max(n_peaks-1, 0) + sum((isi-isi_mu).^2, 'all'))/(max(n_peaks-1,0) + length(isi)))^0.5;
        
        n_peaks = n_peaks + length(all_peaks);
        if length(all_peaks) > 0
            last_peak = all_peaks(end);
        end
        y0 = y(end);
    end
    disp(isi_mu);
    disp(isi_std);
    t_total = t_end*n_repeats + t_total;
    y_end = y(end);
    f = n_peaks/t_total;
    save(fullfile(output_path, sprintf('sigma_%f_r_%f.mat', s, r)), "s", "l", "r", "isi_mu", "isi_std", "f", "y_end", "dt_set", "t_total", "last_peak", "n_peaks");
    disp(toc)
end