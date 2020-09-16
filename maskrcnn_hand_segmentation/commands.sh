
<< 'MULTILINE-COMMENT'
#python3 train_lr_classes.py --lr 0.001 --ep 3 
#python3 plot_losses.py --index 3
#python3 inference_lr_classes.py --lr 0.001 --ep 10 --visthr 0.9
#convert -delay 5 './test_results_maskrcnn_lr/test_%d.jpg'[1-640] test_kitchen_lr_ep10_.9.gif
python3 inference_lr_classes.py --lr 0.001 --ep 10 --visthr 0.8
convert -delay 5 './test_results_maskrcnn_lr/test_%d.jpg'[1-640] test_kitchen_lr_ep10_.8.gif

MULTILINE-COMMENT

python3 train.py --lr 0.001 --ep 6 --nmsthr 0.7
python3 plot_losses.py --index 6
python3 inference.py --lr 0.001 --ep 6 --nmsthr 0.7 --visthr 0.8
convert -delay 5 './test_results_maskrcnn/test_%d.jpg'[1-640] test_kitchen_ep6_nms.7_vis.8.gif

