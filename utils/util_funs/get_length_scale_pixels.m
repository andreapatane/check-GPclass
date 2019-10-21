function pix_2_mod = get_length_scale_pixels(theta_vec,feat_size)
[~,idxs] = sort(theta_vec,'descend');
pix_2_mod = idxs(1:feat_size);

end