python depth_normal_ali_stoch_moving.py --width 128 --height 128 --splats_img_size 128 \
--pixel_samples=1 --lr 2e-4   --disc_type cnn --cam_dist 1.2 --fovy 60 --batchSize 6  \
--gz_gi_loss 0.2 --est_normals --zloss 0.05  --unit_normalloss 0.0 \
--normal_consistency_loss_weight 10.0 --spatial_var_loss_weight 0.0 --grad_img_depth_loss 0.0 \
--spatial_loss_weight 0.0 \
--root_dir ./IQobjs/  \
--supervised_root_dir=./IQobjs/ \
--name iqtest_spherebg_rotate_360_alice --only_foreground
