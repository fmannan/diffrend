#python two_gans.py --width 128 --height 128 --splats_img_size 128 --lr 2e-4 --name two_gans_bowlhalfbox --disc_type cnn --cam_dist 1.6 --batchSize 6
#python gan.py --width 128 --height 128 --splats_img_size 128 --lr 2e-4 --name two_gans_bowlhalfbox --disc_type cnn --cam_dist 1.6 --batchSize 2 --root_dir=./halfbox_mvfg_input2

python gan.py --width 128 --height 128 --splats_img_size 128 --pixel_samples=1 --lr 2e-4 --name sphere --disc_type cnn --cam_dist 1.0 --batchSize 6   --root_dir ./halfbox_mvfg_input2 --gz_gi_loss 0.2 --est_normals --zloss 0.05  --unit_normalloss 0.0 --normal_consistency_loss_weight 10.0 --spatial_var_loss_weight 0.0 --grad_img_depth_loss 0.0 --spatial_loss_weight 0.0
