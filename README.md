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
rm -f ~/Documents/data/models/pytorch-checkpoints/*.pth \
&& \
rm -f ~/Documents/data/images/pytorch/*.jpg \
&& \
python neural_style/neural_style.py train \
--dataset ~/Documents/data/training \
--style-image ~/Documents/data/images/styles/anfield.jpg \
--style-size 620 \
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

### Export to CoreML

```
python ./neural_style/neural_style.py eval  \
--content-image ~/Documents/data/images/test.jpg \
--output-image ~/Documents/data/images/stylized-test.jpg \
--model ~/Documents/data/models/pytorch-checkpoints/checkpoint_43000.pth \
--cuda 0 \
--export_onnx ~/Documents/data/models/pytorch_model.onnx \
&& \
python ./onnx_to_coreml.py \
~/Documents/data/models/pytorch_model.onnx  \
~/Documents/data/models/anfield.mlmodel
```

### Stylize test
```
python neural_style/neural_style.py eval \
--model ~/Documents/data/models/pytorch-checkpoints/checkpoint_100.pth \
--content-image ~/Documents/data/images/test.jpg \
--output-image ~/Documents/data/images/stylized-test.jpg \
--cuda 0
```

