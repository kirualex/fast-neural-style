import os
import sys
import onnx_coreml
import onnx
import coremltools
from onnx import onnx_pb, utils
from onnx_coreml import convert

print("----")
os.system("pip freeze | grep onnx")
os.system("pip freeze | grep coremltools")
print("----")

model_in = sys.argv[1]
model_out = sys.argv[2]

onnx_model = onnx.load(model_in)
polished_model = onnx.utils.polish_model(onnx_model)

coreml_model = convert(polished_model, image_input_names=['0'], image_output_names=['156'])
coreml_model.author = "Monoqle"
coreml_model.license = "All rights reserved"

spec = coreml_model.get_spec()
coremltools.models.utils.rename_feature(spec, '0', 'inputImage')
coremltools.models.utils.rename_feature(spec, '156', 'outputImage')
spec.neuralNetwork.preprocessing[0].featureName = 'inputImage'
coremltools.models.utils.save_spec(spec, model_out)
coremltools.models.utils.save_spec(spec, model_out)
