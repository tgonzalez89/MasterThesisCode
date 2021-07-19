import sys
import time
from prep_data import prep_data
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model

model_path = sys.argv[1]
print(f"\nBENCHMARKING TENSORFLOW MODEL {model_path}\n")

if len(sys.argv) > 3:
    data = prep_data(size=int(sys.argv[3]), data_type='test')
else:
    data = prep_data(data_type='test')

test_data = data[0]
test_labels = data[1]

print("Benchmarking...")

bench_time = int(sys.argv[2])

model = load_model(model_path)

start_t = time.time()
end_t = start_t
it_count = 0
while (end_t - start_t) < bench_time:
    predicted = model.predict(test_data)
    end_t = time.time()
    it_count += 1
print(f"Count:      {len(test_data) * it_count} iterations")
print(f"Duration:   {(end_t - start_t) * 1000} ms")
print(f"Throughput: {len(test_data) * it_count / (end_t - start_t)} FPS")
