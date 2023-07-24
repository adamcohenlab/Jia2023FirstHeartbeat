function single_sim_LIF_twovar_2D(s, r, sigma_r, D, nx, ny, t_end, dt_set, output_path)
    tic
    global dt rest sigma nx ny alpha beta
    alpha = 1;
    beta = 1;
    rest = r*ones(1,nx*ny) + sigma_r*randn(1,nx*ny);
    sigma = s;
    dt = dt_set;
    y0 = r*ones(1, 2*ny*nx);
    phi_field = zeros(ny, nx);
    phi_field(:,1:end-1,1) = D;
    phi_field(1:end-1,:,2) = D;

    [t,y] = onecell_euler(@(t,y,f) twovar2D(t,y,phi_field,f), [0, t_end], dt, y0, 0);
    y_mean = mean(y(:, nx*ny+1:end), 2);
    
    [~, locs] = findpeaks(1-y_mean, 'MinPeakHeight', 0.6, 'MinPeakProminence', 0.45);
    n_peaks = length(locs);
    isi = diff(t(locs));
    isi_mu = mean(isi);
    isi_std = std(isi);
    f = n_peaks/t_end;
    [ps, f_spect] = pspectrum(y_mean-mean(y_mean), 1/dt);
    save(fullfile(output_path, sprintf('sigma_%f_r_%f.mat', s, r)), "s", "r", "isi_mu", "isi_std", "f", "ps", "f_spect");
    disp(toc)
end