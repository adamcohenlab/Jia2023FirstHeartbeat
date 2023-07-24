%%
syms u v
eps_0 = 0.002;
k=8;
a = 0.15;
mu_1 = 1;
mu_2 = 1;

x11 = -k*u*(u-a)*(u-1) - v;
eps = eps_0 + mu_1*v/(u+mu_2);
x21 = -eps*k*(u-a-1);
x22 = -eps;

%%
eq1 = x11*u ==0;
sln1 = solve(eq1, v);

%%
eq2 = x21*u + x22*v == 0;
sln2 = solve(eq2, v);

%%
sln1_f = matlabFunction(sln1);
sln2_f1 = matlabFunction(sln2(1));
sln2_f2 = matlabFunction(sln2(2));

x = -0.1:0.01:1.1;
figure;
hold on;
plot(x, sln1_f(x));
plot(x, sln2_f1(x));
plot(x, sln2_f2(x));