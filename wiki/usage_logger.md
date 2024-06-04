

# Example of usage with IntertwinAI Logger

This section provides an example of how to use Prov4ML with PyTorch and the IntertwinAI Logger.

The following code snippet shows how to log metrics, system metrics, carbon metrics, and model versions. 

#### Example

```python
logger = ProvMLItwinAILogger()
logger.create_logger_context()

for epoch in tqdm(range(EPOCHS)):
    for i, (x, y) in enumerate(train_loader):
        optim.zero_grad()
        y_hat = mnist_model(x)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optim.step()
        logger.log(item=loss.item(), identifier="MSE_train", kind=prov4ml.LoggingItemKind.METRIC, context=prov4ml.Context.TRAINING, step=epoch)
    
    # log system and carbon metrics (once per epoch), as well as the execution time
    logger.log(item=None, identifier="carbon", kind=prov4ml.LoggingItemKind.CARBON_METRIC, context=prov4ml.Context.TRAINING, step=epoch)
    logger.log(item=None, identifier="system", kind=prov4ml.LoggingItemKind.SYSTEM_METRIC, context=prov4ml.Context.TRAINING, step=epoch)
    # save incremental model versions
    logger.log(item=mnist_model, identifier=f"mnist_model_version_{epoch}", kind=prov4ml.LoggingItemKind.MODEL_VERSION, context=prov4ml.Context.TRAINING, step=epoch)

logger.destroy_logger_context()
```

