% Parses the abalone data, normalizes it and stores it in Matlab format.
% Download the dataset from
% - https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data
% and run the following script.
%
% @author Wolfgang Roth

close all;
clear;

fid = fopen('abalone.data');
tline = fgets(fid);
data = [];
while ischar(tline) && length(tline) > 5
  if tline(1) == 'M'
    sex = -1;
  elseif tline(1) == 'F'
    sex = 1;
  elseif tline(1) == 'I'
    sex = 0;
  else
    fprintf('"%s"\n', tline);
    error('Could not parse sex: %c', pline(1));
  end
  tline = tline(3:end);
  pline = sscanf(tline, '%f,%f,%f,%f,%f,%f,%f,%f');
  pline = pline';
  data = [data; [sex, pline]];
  tline = fgets(fid);
end
fclose(fid);

clear fid tline pline tmp;
x = data(:, 1:(end-1));
t = data(:, end);
x = bsxfun(@times, bsxfun(@minus, x, mean(x)), 1 ./ std(x));

save('abalone.mat', 'x', 't');