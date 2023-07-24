function [t,y] = onecell_euler(de, ts, dt, y0, f)
    t = ts(1):dt:ts(2);
    y = zeros(size(t,2), size(y0,2));
    y(1,:) = y0;
    for i = 2:size(t,2)
        y(i,:) = (y(i-1,:) + de(t(i), y(i-1,:), f)*dt);
        if y(i,:) >= 2*pi
            y(i,:) = y(i,:) - 2*pi;
        end
    end
end