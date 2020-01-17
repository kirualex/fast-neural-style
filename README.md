# pytorch-fns

### Environment

```bash
// If env already created
source activate pytorch 

// If not, let's create it
conda create pytorch python=3.6
conda install pytorch=0.4.2
pip install -r requirements.txt
```

### Train

```bash
rm -f ~/Documents/data/models/pytorch-checkpoints/*.pth \
&& \
rm -f ~/Documents/data/images/pytorch/*.jpg \
&& \
python neural_style/neural_style.py train \
--dataset ~/Documents/data/training \
--style-image ~/Documents/data/images/styles/beer.jpg \
--style-size 720 \
--batch-size 2 \
--epochs 1 \
--seed 27 \
--style-weight 1e10 \
--content-weight 1.e5 \
--checkpoint-model-dir ~/Documents/data/models/pytorch-checkpoints \
--checkpoint-interval 1000 \
--save-model-dir ~/Documents/data/models \
--log-interval 10 \
--cuda 1 
```

### Export to CoreML

```bash
python ./converter.py  \
--pth_model ~/Documents/data/models/pytorch-checkpoints/checkpoint_18500.pth \
--output ~/Documents/data/models/mlmodels/test.mlmodel
```

### Stylize test
```bash
python neural_style/neural_style.py eval \
--model ~/Documents/data/models/pytorch-checkpoints/checkpoint_100.pth \
--content-image ~/Documents/data/images/test.jpg \
--output-image ~/Documents/data/images/stylized-test.jpg \
--cuda 0
```

```
python ./src/main.py test \
--content-image ~/Documents/data/images/test.jpg \
--output-image ~/Documents/data/images/stylized-test.jpg \
--model ~/Documents/data/models/meta.pth \
--cuda 1
```

