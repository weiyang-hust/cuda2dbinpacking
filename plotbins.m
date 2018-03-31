pack_bins = load('packing_result.txt');

figure

for i = 1:size(pack_bins, 1)
    hold on
    line([pack_bins(i, 1), pack_bins(i, 1), pack_bins(i, 1) + pack_bins(i, 3), pack_bins(i, 1) + pack_bins(i, 3), pack_bins(i, 1)], ...
         [pack_bins(i, 2), pack_bins(i, 2) + pack_bins(i, 4), pack_bins(i, 2) + pack_bins(i, 4), pack_bins(i, 2), pack_bins(i, 2)], ...
         'color', rand(3,1));
end