function A = get_sensing_matrix_all(bs1_pos3, bs2_pos3, bs3_pos3, bs4_pos3, voxel_pos, lambda0, carriers, sensing_matrx_scale)

tx_pos3 = bs1_pos3;
rx_pos3 = [bs1_pos3, bs2_pos3, bs3_pos3, bs4_pos3];
A1 = get_sensing_matrix(lambda0, voxel_pos, tx_pos3, rx_pos3, carriers);

tx_pos3 = bs2_pos3;
rx_pos3 = [bs2_pos3, bs3_pos3, bs4_pos3];
A2 = get_sensing_matrix(lambda0, voxel_pos, tx_pos3, rx_pos3, carriers);

tx_pos3 = bs3_pos3;
rx_pos3 = [bs3_pos3, bs4_pos3];
A3 = get_sensing_matrix(lambda0, voxel_pos, tx_pos3, rx_pos3, carriers);

tx_pos3 = bs4_pos3;
rx_pos3 = bs4_pos3;
A4 = get_sensing_matrix(lambda0, voxel_pos, tx_pos3, rx_pos3, carriers);

A = [A1; A2; A3; A4] * sensing_matrx_scale;

end


function A = get_sensing_matrix(lambda0, voxel_pos, tx_pos3, rx_pos3, carriers)

c = 3e8;
H_tx_roi = construct_los_channel(lambda0, voxel_pos, tx_pos3)';
H_roi_rx = construct_los_channel(lambda0, rx_pos3, voxel_pos)';
A = zeros(size(H_roi_rx, 1) * size(H_tx_roi, 2) * length(carriers), size(voxel_pos, 2));

AA = zeros(size(H_roi_rx, 1) * size(H_tx_roi, 2), size(voxel_pos, 2));
for carrier_idx = 1 : length(carriers)
    frequency = carriers(carrier_idx);
    lambda = c / frequency;

    H_tx_roi = construct_los_channel(lambda, voxel_pos, tx_pos3)';
    H_roi_rx = construct_los_channel(lambda, rx_pos3, voxel_pos)';
    for i = 1 : size(H_roi_rx, 1)
        AA((i - 1) * size(H_tx_roi, 2) + 1 : i * size(H_tx_roi, 2), :) = H_tx_roi.' * diag(H_roi_rx(i, :));
    end

    A((carrier_idx - 1) * size(H_roi_rx, 1) * size(H_tx_roi, 2) + 1 : carrier_idx * size(H_roi_rx, 1) * size(H_tx_roi, 2), :) = AA;
end

end
