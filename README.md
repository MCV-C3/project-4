# Team 4

## Members
- Álvaro Díaz Laureano
- Benet Ramió i Comas
- Mohammed Oussama Ammouri
- Marina Rosell Murillo

## Week 4

### Adding New Models

This project supports **automatic model loading** based on the configuration file.  
To add a new model without modifying the training pipeline code, follow these steps:

#### 1. Implement the Model Class

Create your model in the `models/` directory as a standard PyTorch module.

Example file:

```python
# models/my_new_net.py

import torch.nn as nn

class MyNewNet(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        ...
```

#### Requirements:

- The model must accept `num_classes` as a constructor argument.

- All other configurable arguments should be passed via keyword arguments.

- The class name must match the name you plan to use in the config file.

#### 2. Expose the Model in `models/__init__.py`

To make the model discoverable by the training script, import it in:

```python
# models/__init__.py

from .base_net import BasicCNN
from .squeeze_net import SqueezeNet
from .shufflenet_mini import ShuffleNetMini

# Add your new model here
from .my_new_net import MyNewNet
```

This is the only Python-side change required.
No modifications to the main training script are needed.

#### 3. Configure the Model in YAML

Models are instantiated dynamically using the `model.name` field and the contents of the `model.params` section.

Add your model to a configuration file like this:

```yaml
model:
  name: MyNewNet
  params:
    hidden_dim: 128
    dropout: 0.3
```

At runtime:

- The class `MyCoolNet` will be automatically loaded from `models/`

- All fields inside `params:` will be passed directly to the model constructor

- The framework automatically injects `num_classes` based on the dataset


---

**Summary:**

To add a new model:

- Create the model class in the `models/` directory

- Import it in `models/__init__.py`

- Define its parameters in the `model.params` section of a config file

🎉 That’s it! No changes to the training pipeline are required.

### Customizing Sweep Experiment Names

When running hyperparameter sweeps, standard run names (e.g., `jumping-lion-2`) can be hard to identify.
To enforce a specific name for your output folders:

1. Add `experiment_name` to your sweep parameters in the YAML config.
2. The `main.py` script automatically detects this key in `model.params`, sets the WandB run name, and creates the output directory accordingly.

**Example Sweep Configuration:**
```yaml
parameters:
  model.params:
    values:
      - {dropout: 0.1, experiment_name: "MyModel_LowDropout"}
      - {dropout: 0.5, experiment_name: "MyModel_HighDropout"}
```

**Result:**
- Output folders will be created at: `results/sweeps/<SWEEP_ID>/MyModel_LowDropout`
- WandB run name will be set to: `MyModel_LowDropout`
