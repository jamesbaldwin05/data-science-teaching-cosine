# 01 Deep Learning Introduction

**Concept**  
Deep learning uses neural networks with many layers. Python libraries like PyTorch and TensorFlow make building and training networks easier. For this module, torch is optional.

### Example
```python
# Lightweight MLP using PyTorch (requires torch installed)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np

    X = torch.tensor(np.random.randn(100, 4), dtype=torch.float32)
    y = torch.tensor(np.random.randint(0, 3, 100), dtype=torch.long)

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(4, 16), nn.ReLU(),
                nn.Linear(16, 3)
            )
        def forward(self, x):
            return self.layers(x)

    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
    print("Final loss:", loss.item())
except ImportError:
    print("torch not installed. Run: pip install torch")
```

### Exercise
"""
Modify the MLP above to add another hidden layer (e.g., 16 → 8 → 3 units).
"""
```python
# Your code here (see above example for structure)
```

### Quiz
**Question:** Which library is commonly used for deep learning in Python?
- A) scikit-learn
- B) tensorflow
- C) pandas
- D) seaborn
**Answer:** B