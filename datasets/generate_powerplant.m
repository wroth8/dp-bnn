% Parses the Combined Cycle Power Plant data, normalizes it and stores it
% in matlab format
% Download the dataset from
% - https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip
% Unzip and open Folds5x2_pp.xlsx and store Sheet1 as
% Folds5x2_pp_sheet1.csv (the first line of the csv-file should contain the
% header). Then run the following script.
%
% @author Wolfgang Roth

close all;
clear;

[x, t] = generate_powerplant_importfile('Folds5x2_pp_sheet1.csv');
x = bsxfun(@times, bsxfun(@minus, x, mean(x)), 1 ./ std(x));
save('powerplant.mat', 'x', 't');
