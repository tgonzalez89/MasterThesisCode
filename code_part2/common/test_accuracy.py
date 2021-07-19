import sys
import numpy as np
from prep_data import prep_data
from tensorflow.keras.models import load_model

model_path = sys.argv[1]

print(f"\nTESTING TENSORFLOW ACCURACY ON MODEL {model_path}\n")

if len(sys.argv) > 2:
    data = prep_data(size=int(sys.argv[2]), data_type='test')
else:
    data = prep_data(data_type='test')

test_data = data[0]
test_labels = data[1]

print("Testing accuracy...")

model = load_model(model_path)

actual = test_labels
predicted = model.predict(test_data)
good = 0
for i in range(len(predicted)):
    a = np.argmax(actual[i])
    p = np.argmax(predicted[i])
    if a == p:
        good += 1
print(f"Accuracy:  {good / len(predicted)}")
