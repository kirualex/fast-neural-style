# pytorch-fns



### Train

```
python neural_style/neural_style.py train \
--dataset ~/Documents/data \
--style-image ~/Documents/data/images/styles/filter_mosaic.jpg \
--vgg-model-dir ~/Documents/data/models \
--save-model-dir ~/Documents/data/models \
--checkpoint-name ~/Documents/data/models/abhiskk_checkpoint \
--checkpoint-interval 200 \
--content-weight 1.0 \
--style-weight 3.0 \
--batch-size 2 \
--epochs 2 \
--seed 77 \
--log-interval 5 \
--cuda 1 \
--fine-tune
```	

### Stylize test
```
python neural_style/neural_style.py eval \
--model ~/Documents/data/models/abhiskk_checkpoint.pth \
--content-image ~/Documents/data/images/test.jpg \
--output-image ~/Documents/data/images/stylized-test.jpg \
--cuda 0
```

### Export to CoreML

```
python neural_style/neural_style.py export \
--input-model ~/Documents/data/models/abhiskk_checkpoint.pth \
--output-model ~/Documents/data/models/abhiskk_model.mlmodel
```

```
/usr/local/lib/python2.7/dist-packages/torch/onnx
```


