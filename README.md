# pytorch-fns

### Environment

```
// If env already created
source activate pytorch 

// If not, let's create it
conda create pytorch python=3.6
conda install pytorch=0.3.1
conda install torchvision=0.2.1
```

### Train

```
python neural_style/neural_style.py train \
--dataset ~/Documents/data \
--style-image ~/Documents/data/images/styles/filter_grosset.jpg \
--image-size 256 \
--style-size 256 \
--batch-size 2 \
--epochs 2 \
--seed 27 \
--style-weight 1e10 \
--content-weight 2e5 \
--checkpoint-model-dir ~/Documents/data/models/pytorch-checkpoints \
--checkpoint-interval 1000 \
--save-model-dir ~/Documents/data/models \
--log-interval 10 \
--cuda 1 
```

### Stylize test
```
python neural_style/neural_style.py eval \
--model ~/Documents/data/models/pytorch-checkpoints/checkpoint.pth \
--content-image ~/Documents/data/images/test.jpg \
--output-image ~/Documents/data/images/stylized-test.jpg \
--cuda 0
```

### Export to CoreML

```
python ./neural_style/neural_style.py eval  \
--content-image ~/Documents/data/images/test.jpg \
--output-image ~/Documents/data/images/stylized-test.jpg \
--model ~/Documents/data/models/pytorch-checkpoints/checkpoint.pth \
--cuda 0 \
--export_onnx ~/Documents/data/models/pytorch_model.onnx \
&& \
python ./onnx_to_coreml.py \
~/Documents/data/models/pytorch_model.onnx  \
~/Documents/data/models/model.mlmodel
```


