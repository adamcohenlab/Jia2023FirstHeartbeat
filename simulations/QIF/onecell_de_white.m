function dxdt = onecell_de_white(t, x, f)
    global rest dt sigma x_reset
    if x > 5.5
        dxdt = (-x + x_reset)/dt;
    else
        dxdt = x^2 + rest + sigma*randn + f(x, t);
    end
end