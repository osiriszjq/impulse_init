save_path='1031/'
for dataset in 'svhn'; do
for data_aug in '2'; do
for lr in '1e-4'; do
for init in 'impulse16_64_5_0.1_100' ; do  # 'mimetic512_64' 'impulse16_64_5_0.1_100' 'random512_64'
# for i in 1 2 3 4 5; do
for alpha in '0.0' '0.1' '0.2' '0.3' '0.4' '0.5'; do
python mytrain_vit_svhn.py --data_aug ${data_aug} --dataset ${dataset} --save_path ${save_path} --lr ${lr} --spatial_pe --spatial_x --init ${init} --use_value --trainable --alpha ${alpha}
done
# done
done

done
done
done