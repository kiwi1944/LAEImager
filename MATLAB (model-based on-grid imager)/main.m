
% author: Huang Yixuan, huangyx@seu.edu.cn, 2024/11/15

clear
clc


f0 = 2.6e9;
c = 3e8;
lambda0 = c / f0;
sensing_matrx_scale = 1e6; % avoid small values
carriers = f0;

trans_power = 50; % dBm
trans_scale = sqrt(10 ^ ((trans_power - 30) / 10));
antenna_gain = 2 * 2; % related to TX/RX antenna radiation pattern
channel_gain = sqrt(antenna_gain) * lambda0 / sqrt(4 * pi);
UAV_RCS = 0.01; % m^2


%% BS position, unit: m
% 4 BSs with UPAs
bs_height = 20;
aaa = 70; % half of the side length
bs_antenna_num = [1, 5, 5]; % UPA, antenna num for three dimensions, x, y, and z axis

bs1 = [ - aaa, aaa, bs_height];
ang = 1 * pi / 4;
bs1_pos3 = get_antenna_pos(bs1, lambda0, bs_antenna_num, ang);
bs2 = [aaa, aaa, bs_height];
ang = 3 * pi / 4;
bs2_pos3 = get_antenna_pos(bs2, lambda0, bs_antenna_num, ang);
bs3 = [aaa, - aaa, bs_height];
ang = 1 * pi / 4;
bs3_pos3 = get_antenna_pos(bs3, lambda0, bs_antenna_num, ang);
bs4 = [ - aaa, - aaa, bs_height];
ang = 3 * pi / 4;
bs4_pos3 = get_antenna_pos(bs4, lambda0, bs_antenna_num, ang);


%% get 2D voxel_pos
scale = 3; % scale the roi grid
x_mean0 = sqrt(UAV_RCS);
x_var0 = 0.001;
uav_height = 40; % UAV height, unit: m
uav_num = 6;

roi_center = [0, 0, uav_height]';
voxel_size = round([120 / scale, 120 / scale, 1]');
voxel_d = [1, 1, 1]' * scale;
voxel_pos0 = zeros(3, voxel_size(1) * voxel_size(2) * voxel_size(3));
for i = 1 : voxel_size(1)
    for j = 1 : voxel_size(2)
        for k = 1 : voxel_size(3)
            voxel_pos0(:, (i - 1) * voxel_size(2) * voxel_size(3) + ...
                (j - 1) * voxel_size(3) + k) = [i - 1; j - 1; k - 1];
        end
    end
end
voxel_pos = voxel_pos0 .* voxel_d + roi_center - (voxel_size - [1, 1, 1]') .* voxel_d / 2;



%% get sensing matrix
A = get_sensing_matrix_all(bs1_pos3, bs2_pos3, bs3_pos3, bs4_pos3, voxel_pos, lambda0, carriers, sensing_matrx_scale);
A = A * channel_gain * trans_scale;


%% get ROI
voxel_pos_new = zeros(3, uav_num);
voxel_min = min(voxel_pos, [], 2);
x = zeros(voxel_size(1), voxel_size(2));
selected_pos = [];
selected_num = 0;
pos_threshold = 3; % avoid too small distance
while selected_num < uav_num
    tmp = round([rand * voxel_size(1), rand * voxel_size(2)]);
    if tmp(1) < 1 || tmp(1) > voxel_size(1) || tmp(2) < 1 || tmp(2) > voxel_size(2)
        continue
    end
    flag = 1;
    for i = 1 : size(selected_pos, 1)
        dis = norm(selected_pos(i, :) - tmp);
        if dis < pos_threshold
            flag = 0;
            break
        end
    end
    if flag
        selected_pos = [selected_pos; tmp];
        selected_num = selected_num + 1;
        x(tmp(1), tmp(2)) = 1;
        voxel_pos_new(:, selected_num) = [voxel_min(1) + voxel_d(1) * (tmp(1) - 1), voxel_min(2) + voxel_d(2) * (tmp(2) - 1), uav_height];
    end
end
x = x.';

x_scale = x(:);
support = find(x_scale);
x_scale(support) = normrnd(x_mean0, x_var0, length(support), 1);
x_scale = reshape(x_scale, voxel_size(1), voxel_size(2));

figure(1)
set(gcf, 'Color', [1 1 1]);
h = heatmap(x_scale);
colorbar('off')
h.XDisplayLabels(1 : 1 : end) = {''};
h.YDisplayLabels(1 : 1 : end) = {''};


%% get noise free measurements
y0 = A * x_scale(:);


%% get noisy measurements
noise_power = -110; % dBm, noise power at a single antenna
noise_scale = sqrt(10 ^ ((noise_power - 30) / 10) * length(y0));

noise = randn(length(y0), 1) + 1j * randn(length(y0), 1);
noise = noise / norm(noise) * noise_scale * sensing_matrx_scale;
y = y0 + noise;


%% solve
sparsity = length(support);
[sp, sparseSupport] = CS_SP(sparsity * 1, A, y, 1e-10);
sp = abs(sp);
sp(sp < max(sp) / 5) = 0;
sp_mtx = reshape(sp, voxel_size(1), voxel_size(2));
figure(2)
set(gcf, 'Color', [1 1 1]);
h = heatmap(sp_mtx);
colorbar('off')
h.XDisplayLabels(1 : 1 : end) = {''};
h.YDisplayLabels(1 : 1 : end) = {''};


[ssim(sp_mtx, x_scale), mse(sp - x_scale(:))]












