function single_sim_LIF_count_isis(s, r, reset, t_end, dt_set, n_repeats, output_path)
    tic
    global dt rest sigma x_reset
    function y = func(x, t)
       y = 0;
    end
    
    rest = r;
    sigma = s;
    dt = dt_set;
    x_reset = reset;
    
    n_peaks = 0;
    isi_std = 0;
    isi_mu = 0;
    last_peak = -1;
    disp(x_reset);
    disp(t_end);
    y0 = min(rest, 1) + randn*sigma;
    for i = 1:n_repeats
        [t,y] = onecell_euler(@onecell_de_white, [0, t_end], dt, y0, @func);
        locs = find(y > 5.5);
        all_peaks = [t(locs)+(i-1)*t_end];
        
        if last_peak == -1
            isi = diff(all_peaks);
        else
            isi = diff([last_peak all_peaks]);
        end
        
        isi_mu = (isi_mu*(n_peaks-1) + sum(isi))/(max(n_peaks-1,0) + length(isi));
        isi_std = ((isi_std^2*max(n_peaks-1, 0) + sum((isi-isi_mu).^2, 'all'))/(max(n_peaks-1,0) + length(isi)))^0.5;
        
        if length(all_peaks) > 0
            last_peak = all_peaks(end);
            n_peaks = n_peaks + length(all_peaks);

        end
        y0 = y(end);
    end
    
    f = n_peaks/(t_end*n_repeats);
    y_end = y(end);
    t_total = t_end*n_repeats;
    disp(n_peaks);
    save(fullfile(output_path, sprintf('sigma_%f_r_%f.mat', s, r)), "s", "r", "isi_mu", "isi_std", "isi", "f", "y_end", "dt_set", "t_total", "n_peaks", "last_peak");
    disp(toc)
end
