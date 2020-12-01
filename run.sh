python train_net.py \
  --config-file yamls/coco/centernet_res18_coco_0.5.yaml  \
  --num-gpus 4 \
  --dist-url "tcp://127.0.0.1:50001" \

#python train_net.py \
#  --config-file yamls/coco/centernet_res50_coco_0.5.yaml  \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50001" \
#
#python train_net.py \
#  --config-file yamls/coco/centernet_res18_KD.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50001" \


#python train_net.py \
#  --config-file yamls/person_face/face_regnetY_4000GF.yaml  \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50001" \

#python train_net.py \
#  --config-file yamls/person_face/person_det_regnetY_4000GF.yaml  \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50001" \

#python train_net.py \
#  --config-file yamls/person_face/person_face_regnetx_400MF_multi_KD.yaml  \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50001" \
#  --eval-only MODEL.WEIGHTS exp_results/person_face/person_face_regnetX_multiKD_sgd_resize/model_final.pth

#python train_net.py \
#  --config-file yamls/person_face/face_regnetY_4000GF.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50002" \
#  #--eval-only MODEL.WEIGHTS exp_results/person_face/person_face_regnetX_multiKD_sgd/model_final.pth

#python train_net.py \
#  --config-file yamls/coco/centernet_res18_KD.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50002" \

#python train_net.py \
#  --config-file yamls/person_det/person_det_res50.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50001" \
#
#python train_net.py \
#  --config-file yamls/person_det/person_det_res18_KD.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50001" \

#python train_net.py \
#  --config-file yamls/coco_det/centernet_r_50_C4_0.5x_coco_car.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50001" \

#python train_net.py \
#  --config-file yamls/coco_det/centernet_r_50_C4_0.5x_coco_person.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50001" \

#python train_net.py \
#  --config-file yamls/coco_det/res18_multi_KD.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50001" \
# --eval-only MODEL.WEIGHTS exp_results/coco_det/coco_car_crowd_multiKD_coco80pretrain/model_final.pth
#
#python train_net.py \
#  --config-file yamls/coco_det/centernet_r_50_C4_0.5x_coco_person.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50001" \

#python train_net.py \
#  --config-file yamls/coco_det/res18_multi_KD.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50001" \

#python train_net.py \
#  --config-file yamls/centernet_r_18_C4_0.5x_coco.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50000" \
#
#python train_net.py \
#  --config-file yamls/centernet_res18_KD.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50000" \

#python train_net.py \
#  --config-file yamls/crowd_human/crowd_human_res18.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50001" \
#  --eval-only MODEL.WEIGHTS exp_results/crowd_human_exp_R18_sgd_BN/model_final.pth
#

#python train_net.py \
#  --config-file yamls/crowd_human/crowd_human_res18_rangerlars.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50000" \

#python train_net.py \
#  --config-file yamls/crowd_human/crowd_human_res18_swa.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50000" \

#python train_net.py \
#  --config-file yamls/crowd_human/crowd_human_res18_imgaug.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50000" \

#python train_net.py \
#  --config-file yamls/crowd_human/crowd_human_res18_SyncBN.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50000" \
#
#python train_net.py \
#  --config-file yamls/crowd_human/crowd_human_res50.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50000" \
#
#python train_net.py \
#  --config-file yamls/crowd_human/crowd_human_res18_KD.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50000" \

#python train_net.py \
#  --config-file yamls/crowd_human/res18_multi_KD.yaml \
#  --num-gpus 4 \
#  --dist-url "tcp://127.0.0.1:50000"