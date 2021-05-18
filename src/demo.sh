# export CUDA_VISIBLE_DEVICES=3
# EDSR baseline model (x2) + JPEG augmentation
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset --ext sep # --ext sep_reset
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset --data_train DIV2K+DIV2K-Q75 --data_test DIV2K+DIV2K-Q75
#python main.py --model EDSR --scale 2 --patch_size 96 --test_only --pre_train download --data_test DIV2K --data_range 801-810
#python statistics.py --model EDSR --scale 2 --patch_size 96 --test_only --gpu_id 3 > ../model_param/edsr_baseline_x2.log

# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --patch_size 144 --save edsr_baseline_x3 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save edsr_baseline_x4 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR in the paper (x2)
#python main.py --model EDSR --scale 2 --save edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_GPUs 3 --reset --epochs 100 --data_range 1-800/801-802
#python main.py --model EDSR --scale 2 --patch_size 96 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --gpu_id 3 --test_only \
#--pre_train ../pretrained/edsr_x2_ref.pt --data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-850 --self_ensemble
#python statistics.py --model EDSR --scale 2 --patch_size 96 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --test_only --gpu_id 3 > ../model_param/edsr_x2.log
#Set5+Set14+B100+Urban100

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save edsr_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]

# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble

#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-810 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --self_ensemble

# Test your own images
#python main.py --data_test Demo --scale 4 --pre_train download --test_only --save_results

# Advanced - Test with JPEG images
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train download --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save edsr_gan --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train download

# RDN BI model (x2)
#python main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 200 --batch_size 16 --patch_size 64 --reset --gpu_id 3 --data_range 1-800/801-810
# python statistics.py --model RDN --scale 2 --batch_size 1 --patch_size 64 --test_only --gpu_id 3 > ../model_param/rdn_D16C8G64_x2.log
#python main.py --model RDN --scale 2 --patch_size 64 --gpu_id 3 --test_only --pre_train ../pretrained/RDN_D16C8G64_BIx2.pt --data_test DIV2K --data_range 801-900
#python main.py --model RDN --scale 2 --patch_size 64 --gpu_id 3 --test_only --pre_train ../pretrained/RDN_D16C8G64_BIx2.pt --data_test DIV2K --data_range 801-900 --self_ensemble
# RDN BI model (x3)
#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 96 --reset
#python main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --patch_size 96 --reset --gpu_id 3 --data_range 1-800/801-810
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 128 --reset
#python main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --patch_size 128 --reset --gpu_id 1 --data_range 1-800/801-810

# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --patch_size 96 --gpu_id 3 --epochs 200
#python main.py --template RCAN --save RCAN_BIX2_C16G7R7_reduct4 --scale 2 --reset --patch_size 96 --epochs 300 --reduction 4
#python main.py --template RCAN --save RCAN_BIX2_C16G7R7_CA_x4 --scale 4 --reset --patch_size 192 --epochs 300 --reduction 4
#python statistics.py --template RCAN --scale 2 --patch_size 96 --test_only --gpu_id 1

python main.py --template RCAN --save RCAN_BIX2_C16G7R7_ESA3 --scale 2 --reset --patch_size 96 --epochs 300 # Std+ESA
python main.py --template RCAN --save RCAN_BIX2_C16G7R7_ESA4 --scale 2 --reset --patch_size 96 --epochs 300 # ESAplus
python main.py --template RCAN --save RCAN_BIX2_C16G7R7_ESA_CEA --scale 2 --reset --patch_size 96 --epochs 300
#python main.py --template RCAN --scale 2 --patch_size 96 --test_only --reduction 4 \
#--pre_train ../pretrained/RCAN_BIX2_G10R20_CA.pt --data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-900
#python main.py --template RCAN --scale 2 --patch_size 96 --test_only --reduction 4 \
#--pre_train ../pretrained/RCAN_BIX2_C16G7R7_reduct4.pt --data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-900

CUDA_VISIBLE_DEVICES=3 python main.py --template RCAN --scale 2 --patch_size 96 --test_only \
--data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --pre_train ../pretrained/RCAN_BIX2_C16G7R7_CBAM_Std.pt

# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --template RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --template RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --template RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt

# SplitSR
#python main.py --template SplitSR --save SplitSR_x2_C16G7R7_ESA --scale 2 --reset --patch_size 96 --epochs 300
#python main.py --template SplitSR --save SplitSR_x2_C16G7R7P48 --scale 2 --reset --patch_size 96 --gpu_id 2 --epochs 400 --n_feats 16
#python statistics.py --template SplitSR --scale 2 --patch_size 96 --test_only # --n_feats 16
python main.py --template SplitSR --scale 2 --patch_size 96 --test_only \
--pre_train ../pretrained/SplitSR_x2_C16G7R7_ESA.pt --data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-900

# MySR
#python main.py --template MySR --save MySR_x2_C48G6R4H3_ESAplus_HFF --scale 2 --reset --patch_size 96 --epochs 300
#python main.py --template MySR --save MySR_x4_C48G6R4H4_ESAplus --scale 4 --epochs 300
#python statistics.py --template MySR --scale 2 --patch_size 96 --test_only
python main.py --template MySR --scale 2 --patch_size 96 --test_only \
--pre_train ../pretrained/MySR_x2_C48G6R4H3_ESAplus_HFF.pt --data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-810

python main.py --template MySR --scale 4 --test_only --pre_train ../pretrained/MySR_x4_C48G6R4H4_ESAplus.pt \
--data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --patch_size 96

# MobileSR
#python statistics.py --template MobileSR --scale 2 --patch_size 96 --test_only --gpu_id 0 --n_feats 16 --att_type CA # > ../model_param/InvertedBlock_x2_C16G7R7_CA.log
#python main.py --template MobileSR --save InvertedBlock_x2_C16G7R7_SE --scale 2 --reset --patch_size 96 --gpu_id 1 --epochs 300 --n_feats 16 --att_type SE
#python main.py --template MobileSR --save InvertedBlock_x2_C16G7R7_CA --scale 2 --reset --patch_size 96 --gpu_id 2 --epochs 300 --n_feats 16 --att_type CA

#python main.py --template MobileSR --scale 2 --patch_size 96 --test_only --att_type SE \
#--pre_train ../pretrained/MobileV3_x2_C16R7G7_SE_beta.pt --data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-900
#python statistics.py --template RCAN --scale 2 --patch_size 96 --test_only --gpu_id 1
#python statistics.py --template RCAN --scale 2 --patch_size 96 --test_only --reduction 4 --gpu_id 2

# ShuffleNet
#python main.py --template ShuffleNet --save ShuffleNet_x2_G10R20_SE --scale 2 --reset --patch_size 96 --epochs 300
python main.py --template ShuffleNet --scale 2 --patch_size 96 --test_only \
--pre_train ../pretrained/ShuffleNet_x2_G10R20_SE.pt --data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-900
#python statistics.py --template ShuffleNet --scale 2 --patch_size 96 --test_only

# LatticeNet
#python statistics.py --template LatticeNet --scale 2 --patch_size 96 --test_only
python main.py --template LatticeNet --save LatticeNet_x2_C64LB4_600 --scale 2 --epochs 300 --patch_size 96 --pre_train ../pretrained/LatticeNet_x2_C64LB4.pt
python main.py --template LatticeNet --scale 2 --test_only --pre_train ../pretrained/LatticeNet_x2_C64LB4_600.pt \
--data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-900 --patch_size 96

# IMDN
python main.py --model IMDN --save IMDN_x2_C64_600 --scale 2 --reset --patch_size 96 --epochs 300 --pre_train ../pretrained/IMDN_x2_C64.pt
python main.py --model IMDN --save IMDN_x4_C64 --scale 4 --reset --epochs 300
python main.py --model IMDN --test_only --pre_train ../pretrained/IMDN_x2_C64_600.pt \
--data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-900 --scale 4 --patch_size 96

# RFDN
#python main.py --model RFDN --save RFDN_paper_x4_C48 --scale 4 --reset --epochs 300 --patch_size 96
python main.py --model RFDN --save RFDN_paper_x2_C48_600 --scale 2 --epochs 300 --patch_size 96 --pre_train ../pretrained/RFDN_paper_x2_C48.pt
#python statistics.py --model RFDN --scale 2 --patch_size 96 --test_only
python main.py --model RFDN --scale 4 --patch_size 96 --test_only \
--pre_train ../pretrained/RFDN_paper_x4_C48.pt --data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-900

# MAFFSRN
python main.py --model MAFFSRN --save MAFFSRN_x2_C32FFG4_600 --scale 2 --reset --patch_size 96 --epochs 300 --pre_train ../pretrained/MAFFSRN_x2_C32FFG4.pt
python main.py --model MAFFSRN --save MAFFSRN_x4_C32FFG4 --scale 4 --reset --epochs 300
#python statistics.py --model MAFFSRN --scale 2 --patch_size 96 --test_only
python main.py --model MAFFSRN --scale 2 --test_only --pre_train ../pretrained/MAFFSRN_x2_C32FFG4_600.pt \
--data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-900 --patch_size 96

# CARN
#python statistics.py --model CARN --scale 2 --patch_size 96 --test_only
#python main.py --model CARN --save CARN_x2_C64_600 --scale 2 --patch_size 96 --epochs 300 --pre_train ../pretrained/CARN_x2_C64.pt
python main.py --model CARN_M --save CARN_M_x4_C64 --scale 4 --reset --epochs 300
python main.py --model CARN --save CARN_x4_C64 --scale 4 --reset --epochs 300
#python main.py --model CARN_M --scale 2 --patch_size 96 --test_only \
#--pre_train ../pretrained/CARN_M_x2_C64.pt --data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-900
python main.py --model CARN_M --scale 2 --test_only --pre_train ../pretrained/CARN_x2_C64_600.pt \
--data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-900  --patch_size 96

python knowledge_distill.py --template MySR_old --save MySR_x2_C64G6R12_teach_at2_V7 --scale 2 --teach_pretrain ../pretrained/TeachSR_x2_C64G6R12.pt \
--reset --patch_size 96 --epochs 300 --w_at 1e+3 --pre_train ../pretrained/MySR_x2_C48G6R4H4_ESAplus.pt

python knowledge_distill.py --template MySR_old --save MySR_x2_C48G12R8_teach_at2_V2 --scale 2 --teach_pretrain ../pretrained/TeachSR_x2_C48G12R8.pt \
--patch_size 96 --epochs 300 --w_at 1e+3 --pre_train ../pretrained/MySR_x2_C48G6R4H4_ESAplus.pt

python statistics.py --template TeachSR --scale 2 --patch_size 96 --test_only
python main.py --template TeachSR --save TeachSR_x2_C64G6R12 --scale 2 --reset --patch_size 96 --epochs 300
#python main.py --template TeachSR --save TeachSR_x2_C64G6R12_600 --scale 2 --patch_size 96 --epochs 300 --pre_train ../pretrained/TeachSR_x2_C64G6R12.pt
python main.py --template TeachSR --scale 2 --patch_size 96 --test_only \
--pre_train ../pretrained/TeachSR_x2_C48G12R8.pt --data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-900

python main.py --model SRCNN --save SRCNN_x2 --scale 2 --reset --patch_size 96 --epochs 300
python statistics.py --model SRCNN --scale 2 --patch_size 96 --test_only
python main.py --model SRCNN --scale 2 --patch_size 96 --test_only \
--pre_train ../pretrained/SRCNN_x2.pt --data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-900

python main.py --model Bicubic --scale 2 --patch_size 96 --test_only --data_test DIV2K+Set5+Set14+B100+Urban100 --data_range 801-900

python main.py --template MySR_old --data_test Demo --scale 2 --pre_train ../pretrained/MySR_x2_C48G6R4H4_ESAplus.pt --test_only --save_results --test_pair

python main.py --template MySR_old --data_test Demo --scale 2 --test_only --save_results --test_pair \
--pre_train ../pretrained/MySR_x2_C64G6R12_teach_at2_V2.pt

python main.py --model RFDN --data_test Demo --scale 4 --test_only --save_results --test_pair \
--pre_train ../pretrained/RFDN_paper_x4_C48.pt


