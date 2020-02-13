import os
import sys
import onnx_coreml
import onnx
import coremltools
import torch
import argparse

from pathlib import Path
from onnx import onnx_pb, utils
from onnx_coreml import convert
from neural_style.transformer_net import TransformerNet

parser = argparse.ArgumentParser()
parser.add_argument('--pth_model', help='Path for your .pth model', type=str, default=None)
parser.add_argument('--onnx_model', help='Path for your .onnx model', type=str, default=None)
parser.add_argument('--alpha', help='Alpha for trained model', type=float, default=1.0)
parser.add_argument('--output', help='Path for your output model', type=str, default="model.mlmodel")
args = parser.parse_args()

print("----")
os.system("pip freeze | grep onnx")
os.system("pip freeze | grep coremltools")
print("----")

model_in = args.pth_model
onnx_in = args.onnx_model
model_out = args.output

# ---- Pytorch -> ONNX

if model_in :

    filename = Path(model_in).stem
    onnx_filename = 'model.onnx'

    print(f'> Converting {filename} to ONNX')
    
    # Define input / output names
    input_names = ["inputImage"]

    model = TransformerNet(args.alpha)
    dummy_input = torch.rand(1, 3, 720, 720)
    output_names = ["outputImage"]

    model.load_state_dict(torch.load(model_in))

    # Convert the PyTorch model to ONNX
    torch.onnx.export(model,
                      dummy_input,
                      onnx_filename,
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names)

    print("> Conversion to ONNX done")
    onnx_model = onnx.load(onnx_filename)
elif onnx_in :
    onnx_model = onnx.load(onnx_in)


# ---- ONNX -> CoreML
if onnx_model :
    print(f'> Converting to CoreML')
    polished_model = onnx.utils.polish_model(onnx_model)

    # Load the ONNX model as a CoreML model
    coreml_model = convert(polished_model,
        image_input_names=['inputImage'],
        image_output_names=['outputImage'])

    print("> Conversion to CoreML done")

    coreml_model.author = "Monoqle"
    coreml_model.license = "All rights reserved"
    coreml_model.save(model_out)
    print(coreml_model.visualize_spec)

    os.system("rm model.onnx")
    print("----")
    print("Finished")
    print("----")
else :
    print("Nothing to process")

