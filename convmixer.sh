data_path='./data'
for lr in '1e-3'; do
for dataset in 'cifar10' 'cifar100'; do
for heads in '512'; do
for init in 'random' 'softmax' 'box1' 'box25'; do

# for i in 1 2 3 4 5; do
python mytrain_convmixer.py --dataset ${dataset}  --data_path ${save_path} --init ${init} --heads ${heads} --lr ${lr} 
python mytrain_convmixer.py --dataset ${dataset}  --data_path ${save_path} --init ${init} --heads ${heads} --lr ${lr} --fix_spatial
# done

done
done
done
done