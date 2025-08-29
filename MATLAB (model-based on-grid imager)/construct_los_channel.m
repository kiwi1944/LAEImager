function h = construct_los_channel(lambda, rx_pos, tx_pos)

% lambda: wave length
% rx_pos: rx antenaa position, dims: 3 x antenna_num
% tx_pos: tx antenaa position, dims: 3 x antenna_num
% h：los channel, a matrix with the size of dim1 x dim2

% channel dims
dim1 = size(tx_pos, 2);
dim2 = size(rx_pos, 2);

% fast generation
dis = zeros(dim1, dim2);
if dim1 >= dim2
    for i = 1 : dim2
        tmp = tx_pos - rx_pos(:, i);
        dis(:, i) = sqrt(sum(abs(conj(tmp) .* tmp)))';
    end
else
    for i = 1 : dim1
        tmp = rx_pos - tx_pos(:, i);
        dis(i, :) = sqrt(sum(abs(conj(tmp) .* tmp)))';
    end
end

h = (1 / sqrt(4 * pi)) ./ dis .* exp( - 1j * 2 * pi * dis / lambda);

end

