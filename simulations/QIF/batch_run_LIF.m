function batch_run_LIF(s, r_min, r_max, n_steps, t_end, dt_set, n_repeats, output_path, reset)
    rs = linspace(r_min, r_max, n_steps+1);
    rs = rs(1:end-1);
    for r = rs
        single_sim_LIF(s, r, reset, t_end, dt_set, n_repeats, output_path);
    end
end
