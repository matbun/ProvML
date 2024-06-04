
# Artifacts logging

The library allows users to log artifacts in the form of files. 
The artifacts can be used to memorize the state of a model at a given point in time, 
or to store the results of a model evaluation. 

# Logging artifacts

The function `prov4ml.log_artifact()` logs an artifact with the specified `artifact_path` in the given context. 
The step parameter is optional and can be used to specify the step at which the artifact was generated. 

```python
prov4ml.log_artifact(artifact_path, context, step, timestamp)
```

| Parameter | Type     | Description                |
| :--------: | :-------: | :-------------------------: |
| `artifact_path` | `str` | **Required**. Path to the artifact |
| `context` | `prov4ml.Context` | **Required**. Context of the artifact |
| `step` | `int` | **Optional**. Step of the artifact |
| `timestamp` | `int` | **Optional**. Timestamp of the artifact |


# Logging the final model

The function `prov4ml.log_model()` logs the final model with the specified `model_name`.
This function is often used for logging the final model after training, or an arbitrary model during the training process.

```python
prov4ml.log_model(model, model_name, log_model_info, log_as_artifact)
``` 

| Parameter | Type     | Description                |
| :--------: | :-------: | :-------------------------: |
| `model` | `torch.nn.Module` | **Required**. Model to be saved |
| `model_name` | `str` | **Required**. Name of the model |
| `log_model_info` | `bool` | **Optional**. Whether to log the model information |
| `log_as_artifact` | `bool` | **Optional**. Whether to log the model as an artifact |


# Logging model versions

The function `prov4ml.save_model_version()` saves a model version with the specified `model_name` and `context`.
The step and timestamp parameters are optional and can be used to specify the step and timestamp at which the model was generated.
This function is often used for saving intermediate models during the training process, as if it were a checkpoint.
    
```python
prov4ml.save_model_version(model, model_name, context, step, timestamp)
```

| Parameter | Type     | Description                |
| :--------: | :-------: | :-------------------------: |
| `model` | `torch.nn.Module` | **Required**. Model to be saved |
| `model_name` | `str` | **Required**. Name of the model |
| `context` | `prov4ml.Context` | **Required**. Context of the model |
| `step` | `int` | **Optional**. Step of the model |
| `timestamp` | `int` | **Optional**. Timestamp of the model |



