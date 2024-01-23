data_path='./data'
for dataset in 'cifar10' 'cifar100' 'svhn' 'tiny_imagenet'; do
for lr in '1e-4'; do
for init in 'impulse16_64_5_0.1_100' 'mimetic512_64' 'random512_64'; do  #  'impulse16_64_5_0.1_100' 'mimetic512_64' 'random512_64'
# for i in 1 2 3 4 5; do
for alpha in '0.0' '0.1' '0.2' '0.3' '0.4' '0.5'; do
python mytrain_vit.py --dataset ${dataset} --data_path ${save_path} --lr ${lr} --spatial_pe --spatial_x --init ${init} --use_value --trainable --alpha ${alpha}  --data_aug
python mytrain_vit.py --dataset ${dataset} --data_path ${save_path} --lr ${lr} --spatial_pe --spatial_x --init ${init} --use_value --trainable --alpha ${alpha}
done
# done

done
done
done