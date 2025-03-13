import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Step 1: Simulate medical image data (e.g., pixel values)
X = np.random.rand(100, 256)  # 100 images with 256 features each
y = np.random.randint(0, 2, 100)  # Binary labels: 0 (healthy), 1 (tumor)

# Step 2: Train a simple machine learning model
model = RandomForestClassifier()
model.fit(X, y)

# Step 3: Predict on new data (a new simulated medical image)
new_image = np.random.rand(1, 256)
prediction = model.predict(new_image)

# Step 4: Output the result
print("Tumor detected!" if prediction[0] == 1 else "No tumor detected.")
