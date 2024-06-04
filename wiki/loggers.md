
# Loggers

Two loggers are provided, to aid in the logging of provenance information in machine learning experiments.

# Prov4MLLogger

Allows simple logging of provenance information in machine learning experiments.

```python
from prov4ml import Prov4MLLogger

logger = Prov4MLLogger()
# add logger to lightning trainer, 
# then call self.log() to log metrics
```

Currently only arbitrary metrics can be logged using this logger. All carbon and system metrics can be logged using the default method, 
explained in the respective [sections](./system.md).

# IntertwinAI Logger

Allows logging of provenance information in machine learning experiments.

```python
from prov4ml import ProvMLItwinAILogger

logger = ProvMLItwinAILogger()
logger.create_logger_context()
# add logger to the intertwinai interface, 
# then call logger.log() to log metrics

# ...

# once done, destroy the logger context to also save 
logger.destroy_logger_context()
```

| Parameter | Type     | Description                |
| :--------: | :-------: | :-------------------------: |
| `item` | `Any` | **Required**. Item to be logged |
| `identifier` | `str` | **Required**. Identifier of the item |
| `kind` | `prov4ml.LoggingItemKind` | **Required**. Kind of the item |
| `context` | `prov4ml.Context` | **Required**. Context of the item |
| `step` | `int` | **Optional**. Step of the item |

The following methods are available for logging all kinds of metrics:

```python
logger.log(item=loss.item(), identifier="MSE", kind=prov4ml.LoggingItemKind.METRIC, context=prov4ml.Context.VALIDATION, step=current_epoch)   
```

```python
logger.log(item=self, identifier="model_version_0", kind=prov4ml.LoggingItemKind.MODEL_VERSION, context=prov4ml.Context.TRAINING, step=current_epoch)
```

```python
logger.log(item=None, identifier=None, kind=prov4ml.LoggingItemKind.SYSTEM_METRIC, context=prov4ml.Context.TRAINING, step=current_epoch)
```

```python
logger.log(item=None, identifier=None, kind=prov4ml.LoggingItemKind.CARBON_METRIC, context=prov4ml.Context.TRAINING, step=current_epoch)
```

```python
logger.log(item=None, identifier="execution_time_label", kind=prov4ml.LoggingItemKind.EXECUTION_TIME, context=prov4ml.Context.TRAINING, step=current_epoch)
```

