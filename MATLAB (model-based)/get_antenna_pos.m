function antenna_pos = get_antenna_pos(center_pos, lambda0, antenna_num, ang)

% bs1: center pos of the antenna array
% lambda0: center carrier frequency
% bs_antenna_num: along x y z axis
% ang: antenna array rotation angle
% bs1_antenna_pos: dimension: 3 times x, where x is the totol antenna num

antenna_spacing = lambda0 / 2;

bs1x = (0 : antenna_spacing : antenna_spacing * (antenna_num(1) - 1)) - antenna_spacing * (antenna_num(1) - 1) / 2;
bs1y = (0 : antenna_spacing : antenna_spacing * (antenna_num(2) - 1)) - antenna_spacing * (antenna_num(2) - 1) / 2;
bs1z = (0 : antenna_spacing : antenna_spacing * (antenna_num(3) - 1)) - antenna_spacing * (antenna_num(3) - 1) / 2;
[bs1x, bs1y, bs1z] = meshgrid(bs1x, bs1y, bs1z);
bs1x = bs1x(:);
bs1y = bs1y(:);
bs1z = bs1z(:);

bs1x = + bs1x * cos(ang) + bs1y * sin(ang);
bs1y = - bs1x * sin(ang) + bs1y * cos(ang);

antenna_pos = [bs1x' + center_pos(1); bs1y' + center_pos(2); bs1z' + center_pos(3)];

end