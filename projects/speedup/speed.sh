#python projects/speedup/centerX2onnx.py \
#--config-file "yamls/person_face/person_face_regnetx_400MF_multi_KD.yaml" \
#--model-path "models/person_face_regnetX400MF_sgd.pth" \
#--name "person_face_regnetX400MF_sgd" \
#--output "models" \
#--input_w 640 \
#--input_h 384

python projects/speedup/centerX2caffe.py \
--config-file "yamls/person_face/person_face_regnetx_400MF_multi_KD.yaml" \
--model-path "exp_results/person_face/person_face_regnetX_multiKD_sgd_resize/model_final.pth" \
--name "person_face_regnetX400MF_sgd_resize" \
--output "models" \
--input_w 640 \
--input_h 384

#python projects/speedup/centerX2pt.py \
#--config-file "yamls/person_det/person_det_res18_KD.yaml" \
#--model-path "exp_results/person_det/R18_adam_nodeform_KD/model_final.pth" \
#--name "person_det_res18_nodeform_KD" \
#--output "models" \
#--input_w 640 \
#--input_h 384