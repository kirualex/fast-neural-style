import sys
import onnx_coreml
import onnx
import coremltools
from onnx import onnx_pb
from onnx_coreml import convert

print("----")
print("onnx: ", onnx.__version__)
print("----")

model_in = sys.argv[1]
model_out = sys.argv[2]

model_file = open(model_in, 'rb')
model_proto = onnx_pb.ModelProto()
model_proto.ParseFromString(model_file.read())

coreml_model = convert(model_proto, image_input_names=['0'], image_output_names=['156'])
coreml_model.author = "Monoqle"
coreml_model.license = "All rights reserved"

spec = coreml_model.get_spec()
coremltools.models.utils.rename_feature(spec, '0', 'inputImage')
coremltools.models.utils.rename_feature(spec, '156', 'outputImage')
spec.neuralNetwork.preprocessing[0].featureName = 'inputImage'
coremltools.models.utils.save_spec(spec, model_out)
coremltools.models.utils.save_spec(spec, model_out)
