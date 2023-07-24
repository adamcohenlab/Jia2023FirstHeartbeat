function dxdt = onecell_de(t, x, f)
    global rest dt
    if x > 1
        dxdt = -x/dt;
    else
        dxdt = rest - x;
    end
end