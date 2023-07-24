close all;    
global rest dt sigma

figure;
hold on;
y0 = 0;
rest = 1.01;
dt = 0.01;
[t,y] = onecell_euler(@onecell_de, [0, 10], dt, y0, 0);
plot(t,y);
plot(t, ones(length(t), 1));
plot(t, ones(length(t), 1)*rest);
legend("x", "x^*", "resting");

%% Plot bifurcation of frequency with threshold
t_end = 20;
freq = [];
rvals = 0.7:0.001:1.5;
dt = 0.01;
for rval = rvals
   rest = rval;
   [t,y] = onecell_euler(@onecell_de, [0, t_end], dt, y0, 0);
   [pks, locs] = findpeaks(y);
   n_peaks = length(pks);
%    if n_peaks >0
%        figure;
%        hold on;
%        plot(t,y);
%        scatter(t(locs), y(locs));
%    end
   if n_peaks > 0
       f = 1/mean(diff(t(locs)));
   else
       f = 0;
   end
   freq = [freq; f];
end

%%
figure;
plot(rvals, freq);
xlabel("Resting Potential");
ylabel("Frequency");

%% add white noise
close all;
t_end = 1000;
rest= 0.1;
dt = 0.02;
y0 = 0;
sigma = 0.1;
[t,y] = onecell_euler(@onecell_de_white, [0, t_end], dt, y0, 0);
[~, locs] = findpeaks(y, 'MinPeakHeight', 0.99, 'MinPeakProminence', 0.45);

figure;
hold on;
plot(t, y);
plot(t(locs), y(locs), "rx");

%% add white noise
close all;
t_end = 100000;
rest=0.99;
sigmas = logspace(-2, -0, 20);
dt = 0.02;
y0 = 0;
n_peaks_found = [];
isi_mu = [];
isi_sigma = [];
for s = sigmas
    sigma = s;
    [t,y] = onecell_euler(@onecell_de_white, [0, t_end], dt, y0, 0);
    [~, locs] = findpeaks(-y, 'MinPeakProminence', 0.5);
    n_peaks = length(locs);
    n_peaks_found = [n_peaks_found; n_peaks];
    isi = diff(t(locs));
    isi_mu = [isi_mu; mean(isi)];
    isi_sigma = [isi_sigma; std(isi)];
end

%%
figure;
plot(sigmas, isi_sigma./isi_mu);
xlabel("\sigma");
ylabel("ISI \sigma/\mu");

%%
figure;
plot(sigmas, isi_mu);
xlabel("\sigma");
ylabel("ISI \mu");

%%
figure;
plot(isi_mu, isi_sigma./isi_mu);
xlabel("ISI \mu");
ylabel("ISI \sigma/\mu");

%% Move threshold
close all;
t_end = 5000;
sigma = 0.1;
rvals = linspace(-0.1, 0.1, 20);
dt = 0.02;
n_peaks_found = [];
isi_mu = [];
isi_sigma = [];
y0 = 0;
for rval = rvals
    rest = rval;
    [t,y] = onecell_euler(@onecell_de_white, [0, t_end], dt, y0, 0);
    [~, locs] = findpeaks(y, 'MinPeakHeight', 0.99, 'MinPeakProminence', 0.45);
    n_peaks = length(locs);
    n_peaks_found = [n_peaks_found; n_peaks];
    isi = diff(locs);
    isi_mu = [isi_mu; mean(isi)];
    isi_sigma = [isi_sigma; std(isi)];
end

%%
figure;
plot(rvals, isi_sigma./isi_mu);
xlabel("Threshold");
ylabel("ISI \sigma/\mu");

%%
figure;
plot(rvals, 1/isi_mu);
xlabel("Threshold");
ylabel("ISI \mu");

%%
figure;
plot(isi_mu, isi_sigma./isi_mu);
xlabel("ISI \mu");
ylabel("ISI \sigma/\mu");