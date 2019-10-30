% Parses the Boston housing data, normalizes it and stores it in Matlab format.
% Download the dataset from
% - https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data
% and run the following script.
%
% @author Wolfgang Roth

close all;
clear;

fid = fopen('housing.data');
tline = fgets(fid);
data = [];
while ischar(tline) && length(tline) > 5
  pline = sscanf(tline, '%f %f %f %f %f %f %f %f %f %f %f %f %f %f');
  pline = pline';
  data = [data; pline];
  tline = fgets(fid);
end
fclose(fid);

clear fid tline pline tmp;
x = data(:, 1:(end-1));
t = data(:, end);
x = bsxfun(@times, bsxfun(@minus, x, mean(x)), 1 ./ std(x));

save('housing.mat', 'x', 't');