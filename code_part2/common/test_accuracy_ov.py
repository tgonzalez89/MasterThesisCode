import cv2
import os
import sys
import numpy as np
from prep_data import prep_data
from openvino.inference_engine import IENetwork, IECore

model_xml = sys.argv[1]
print(f"\nTESTING OPENVINO ACCURACY ON MODEL {model_xml}\n")

if len(sys.argv) > 3:
    data = prep_data(size=int(sys.argv[3]), data_type='test')
else:
    data = prep_data(data_type='test')

test_data = data[0]
test_labels = data[1]

print("Testing accuracy...")

model_bin = os.path.splitext(model_xml)[0] + ".bin"

ie = IECore()
net = IENetwork(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
exec_net = ie.load_network(network=net, device_name=sys.argv[2])

n, c, h, w = net.inputs[input_blob].shape
ov_test_data = list()
for i in range(len(test_data)):
    ov_img = cv2.resize(test_data[i], (w, h))
    ov_img = ov_img.reshape((w, h, c))
    ov_img = ov_img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    ov_test_data.append(ov_img)

good = 0
for i in range(len(test_data)):
    res = exec_net.infer(inputs={input_blob: ov_test_data[i]})
    a = np.argmax(test_labels[i])
    p = np.argmax(res[out_blob])
    if a == p:
        good += 1
print(f"Accuracy:  {good / len(test_data)}")
