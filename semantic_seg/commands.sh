'''
python train_reduced.py --config config1.yaml
echo "Training first model done"
mkdir loss_plots/1
mv loss_plots/*.jpg loss_plots/1
python test.py --model_path ./segnet_pascal_best_model.pkl --dataset pascal --img_path ~/data/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000121.jpg --out_path loss_plots/1/out_2007_000121.png
python test.py --model_path ./segnet_pascal_best_model.pkl --dataset pascal --img_path ~/data/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg --out_path loss_plots/1/out_2007_000032.png
echo "first model done"

python train_reduced.py --config config2.yaml
echo "Training second model done"
mkdir loss_plots/2
mv loss_plots/*.jpg loss_plots/2
python test.py --model_path ./segnet_pascal_best_model.pkl --dataset pascal --img_path ~/data/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000121.jpg --out_path loss_plots/2/out_2007_000121.png
python test.py --model_path ./segnet_pascal_best_model.pkl --dataset pascal --img_path ~/data/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg --out_path loss_plots/2//out_2007_000032.png
echo "second model done"

python train_reduced.py --config config3.yaml
echo "Training third model done"
mkdir loss_plots/3
mv loss_plots/*.jpg loss_plots/3
python test.py --model_path ./segnet_pascal_best_model.pkl --dataset pascal --img_path ~/data/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000121.jpg --out_path loss_plots/3/out_2007_000121.png
python test.py --model_path ./segnet_pascal_best_model.pkl --dataset pascal --img_path ~/data/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg --out_path loss_plots/3//out_2007_000032.png
echo "third model done"
'''
python train_reduced.py --config config4.yaml
echo "Training fourth model done"
mkdir loss_plots/4
mv loss_plots/*.jpg loss_plots/4
python test.py --model_path ./segnet_pascal_best_model.pkl --dataset pascal --img_path ~/data/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000121.jpg --out_path loss_plots/4/out_2007_000121.png
python test.py --model_path ./segnet_pascal_best_model.pkl --dataset pascal --img_path ~/data/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg --out_path loss_plots/4/out_2007_000032.png
echo "fourth model done"

python train_reduced.py --config config5.yaml
echo "Training fifth model done"
mkdir loss_plots/5
mv loss_plots/*.jpg loss_plots/5
python test.py --model_path ./segnet_pascal_best_model.pkl --dataset pascal --img_path ~/data/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000121.jpg --out_path loss_plots/5/out_2007_000121.png
python test.py --model_path ./segnet_pascal_best_model.pkl --dataset pascal --img_path ~/data/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg --out_path loss_plots/5/out_2007_000032.png
echo "fifth model done"

python train_reduced.py --config config6.yaml
echo "Training sixth model done"
mkdir loss_plots/6
mv loss_plots/*.jpg loss_plots/6
python test.py --model_path ./segnet_pascal_best_model.pkl --dataset pascal --img_path ~/data/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000121.jpg --out_path loss_plots/6/out_2007_000121.png
python test.py --model_path ./segnet_pascal_best_model.pkl --dataset pascal --img_path ~/data/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg --out_path loss_plots/6/out_2007_000032.png
echo "sixth model done"

python train_reduced.py --config config7.yaml
echo "Training seventh model done"
mkdir loss_plots/7
mv loss_plots/*.jpg loss_plots/7
python test.py --model_path ./segnet_pascal_best_model.pkl --dataset pascal --img_path ~/data/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000121.jpg --out_path loss_plots/7/out_2007_000121.png
python test.py --model_path ./segnet_pascal_best_model.pkl --dataset pascal --img_path ~/data/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg --out_path loss_plots/7/out_2007_000032.png
echo "seventh model done"



