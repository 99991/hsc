function alphamatting

addpath('mex_funs')

load data/alphamatting.mat

if 1
    hsc_fun = hsc_setup(A, A, rows, cols);

    x = pcg(A, b, 1e-6, 1000, hsc_fun, []);
else
    x = pcg(A, b, 1e-6, 1000);
end

sum(abs(x - x_true))
sum(abs(A * x - b))

x(x < 0) = 0;
x(x > 1) = 1;

imshow(reshape(x, [rows cols]));

end
