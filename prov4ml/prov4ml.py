import os
import mlflow
from mlflow import ActiveRun
import prov.model as prov
import prov.dot as dot

from typing import Optional, Dict, Any
from contextlib import contextmanager

from .utils import energy_utils
from .utils import flops_utils
from .utils.funcs import prov4ml_experiment_matches
from .logging import log_execution_start_time, log_execution_end_time
from .provenance.provenance_graph import first_level_prov
from .constants import MLFLOW_SUBDIR, ARTIFACTS_SUBDIR, PROV4ML_DATA

@contextmanager
def start_run_ctx(
    prov_user_namespace: str,
    experiment_name: Optional[str] = None,
    provenance_save_dir: Optional[str] = None,
    mlflow_save_dir: Optional[str] = None,
    nested: bool = False,
    tags: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    log_system_metrics: Optional[bool] = None, 
    create_graph: Optional[bool] = False, 
    create_svg: Optional[bool] = False
    ) -> ActiveRun: # type: ignore
    """
    Starts an MLflow run and generates provenance information.

    Args:
        prov_user_namespace (str): The user namespace for provenance tracking.
        experiment_name (Optional[str], optional): The name of the experiment to associate the run with. Defaults to None.
        provenance_save_dir (Optional[str], optional): The directory to save provenance data. Defaults to None.
        mlflow_save_dir (Optional[str], optional): The directory to save MLflow artifacts. Defaults to None.
        nested (bool, optional): If True, starts a nested run. Defaults to False.
        tags (Optional[Dict[str, Any]], optional): Dictionary of tags to associate with the run. Defaults to None.
        description (Optional[str], optional): Description of the run. Defaults to None.
        log_system_metrics (Optional[bool], optional): If True, logs system metrics during the run. Defaults to None.

    Returns:
        ActiveRun: The active run object.

    """
    global USER_NAMESPACE, PROV_SAVE_PATH

    USER_NAMESPACE = prov_user_namespace
    PROV_SAVE_PATH = provenance_save_dir

    if create_svg and not create_graph:
        raise ValueError("Cannot create SVG without creating the graph.")

    global_rank = os.getenv("SLURM_PROCID", None)
    PROV4ML_DATA.EXPERIMENT_NAME = experiment_name + f"_GR{global_rank}" if global_rank else experiment_name

    # look at PROV dir how many experiments are there with the same name
    if not os.path.exists(PROV_SAVE_PATH):
        os.makedirs(PROV_SAVE_PATH, exist_ok=True)
    prev_exps = os.listdir(PROV_SAVE_PATH) if PROV_SAVE_PATH else []
    run_id = len([exp for exp in prev_exps if prov4ml_experiment_matches(experiment_name, exp)]) 

    PROV4ML_DATA.EXPERIMENT_DIR = os.path.join(PROV_SAVE_PATH, experiment_name + f"_{run_id}")

    if mlflow_save_dir:
        mlflow.set_tracking_uri(os.path.join(mlflow_save_dir, MLFLOW_SUBDIR))

    exp = mlflow.get_experiment_by_name(name=experiment_name)
    if not exp:
        if mlflow_save_dir: 
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=os.path.join(mlflow_save_dir, ARTIFACTS_SUBDIR)
            )
        else: 
            experiment_id = mlflow.create_experiment(name=experiment_name)
    else:
        experiment_id = exp.experiment_id

    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog(silent=True)

    current_run = mlflow.start_run(
        experiment_id=experiment_id,
        nested=nested,
        tags=tags,
        description=description,
        log_system_metrics=log_system_metrics, 
    )

    energy_utils._carbon_init()
    flops_utils._init_flops_counters()

    log_execution_start_time()

    yield current_run #return the mlflow context manager, same one as mlflow.start_run()

    log_execution_end_time()

    run_id=mlflow.active_run().info.run_id
    
    mlflow.end_run() #end the run, as per mlflow documentation

    client = mlflow.MlflowClient()
    active_run=client.get_run(run_id)

    doc = prov.ProvDocument()

    #set namespaces
    doc.set_default_namespace(USER_NAMESPACE)
    doc.add_namespace('prov','http://www.w3.org/ns/prov#')
    doc.add_namespace('xsd','http://www.w3.org/2000/10/XMLSchema#')
    doc.add_namespace('mlflow', 'mlflow') #TODO: find namespaces of mlflow and prov-ml ontologies
    doc.add_namespace('prov-ml', 'prov-ml')

    doc = first_level_prov(active_run,doc)
    # doc = second_level_prov(active_run,doc)    

    #datasets are associated with two sets of tags: input tags, of the DatasetInput object, and the tags of the dataset itself
    # for input_tag in dataset_input.tags:
    #     attributes[f'mlflow:{input_tag.key.strip("mlflow.")}']=str(input_tag.value)
    # for key,value in ds_tags['tags'].items():
    #     attributes[f'mlflow:{str(key).strip("mlflow.")}']=str(value)

    graph_filename = f'provgraph_{PROV4ML_DATA.EXPERIMENT_NAME}.json'
    dot_filename = f'provgraph_{PROV4ML_DATA.EXPERIMENT_NAME}.dot'

    if not os.path.exists(PROV4ML_DATA.EXPERIMENT_DIR):
        os.makedirs(PROV4ML_DATA.EXPERIMENT_DIR, exist_ok=True)

    path_graph = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, graph_filename)

    # if PROV_SAVE_PATH and not os.path.exists(PROV_SAVE_PATH):
    #     os.makedirs(PROV_SAVE_PATH)

    with open(path_graph,'w') as prov_graph:
        doc.serialize(prov_graph)

    if create_graph:
        path_dot = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, dot_filename)
        with open(path_dot, 'w') as prov_dot:
            prov_dot.write(dot.prov_to_dot(doc).to_string())

def start_run(
    prov_user_namespace: str,
    experiment_name: Optional[str] = None,
    provenance_save_dir: Optional[str] = None,
    mlflow_save_dir: Optional[str] = None,
    nested: bool = False,
    tags: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    log_system_metrics: Optional[bool] = None
    ) -> None:
    """Starts an MLflow run with the specified configurations and options.
    
    Args:
        prov_user_namespace (str): The user namespace for provenance tracking.
        experiment_name (Optional[str], optional): The name of the experiment to associate the run with. Defaults to None.
        provenance_save_dir (Optional[str], optional): The directory to save provenance data. Defaults to None.
        mlflow_save_dir (Optional[str], optional): The directory to save MLflow artifacts. Defaults to None.
        nested (bool, optional): If True, starts a nested run. Defaults to False.
        tags (Optional[Dict[str, Any]], optional): Dictionary of tags to associate with the run. Defaults to None.
        description (Optional[str], optional): Description of the run. Defaults to None.
        log_system_metrics (Optional[bool], optional): If True, logs system metrics during the run. Defaults to None.
    """
    global USER_NAMESPACE, PROV_SAVE_PATH

    USER_NAMESPACE = prov_user_namespace
    PROV_SAVE_PATH = provenance_save_dir

    global_rank = os.getenv("SLURM_PROCID", None)
    PROV4ML_DATA.EXPERIMENT_NAME = experiment_name + f"_GR{global_rank}" if global_rank else experiment_name

    # look at PROV dir how many experiments are there with the same name
    if not os.path.exists(PROV_SAVE_PATH):
        os.makedirs(PROV_SAVE_PATH, exist_ok=True)
    prev_exps = os.listdir(PROV_SAVE_PATH) if PROV_SAVE_PATH else []
    run_id = len([exp for exp in prev_exps if prov4ml_experiment_matches(experiment_name, exp)]) 

    PROV4ML_DATA.EXPERIMENT_DIR = os.path.join(PROV_SAVE_PATH, experiment_name + f"_{run_id}")

    if mlflow_save_dir:
        experiment_num = 0
        if os.path.exists(mlflow_save_dir):
            exps = os.listdir(mlflow_save_dir)
        else:
            exps = []
        for exp in exps:
            if experiment_name in exp:
                experiment_num += 1
        mlflow.set_tracking_uri(os.path.join(mlflow_save_dir, MLFLOW_SUBDIR + f"_{experiment_num}"))

    exp = mlflow.get_experiment_by_name(name=experiment_name)
    if not exp:
        if mlflow_save_dir: 
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=os.path.join(mlflow_save_dir, ARTIFACTS_SUBDIR)
            )
        else: 
            experiment_id = mlflow.create_experiment(name=experiment_name)
    else:
        experiment_id = exp.experiment_id

    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog(silent=True)

    mlflow.start_run(
        experiment_id=experiment_id,
        nested=nested,
        tags=tags,
        description=description,
        log_system_metrics=log_system_metrics, 
    )

    energy_utils._carbon_init()
    flops_utils._init_flops_counters()

    log_execution_start_time()

def end_run(
        create_graph: Optional[bool] = False, 
        create_svg: Optional[bool] = False
        ): 
    """Ends the active MLflow run, generates provenance graph, and saves it."""

    if create_svg and not create_graph:
        raise ValueError("Cannot create SVG without creating the graph.")
    
    log_execution_end_time()

    run_id=mlflow.active_run().info.run_id
    
    mlflow.end_run() #end the run, as per mlflow documentation

    client = mlflow.MlflowClient()
    active_run=client.get_run(run_id)

    doc = prov.ProvDocument()

    #set namespaces
    doc.set_default_namespace(USER_NAMESPACE)
    doc.add_namespace('prov','http://www.w3.org/ns/prov#')
    doc.add_namespace('xsd','http://www.w3.org/2000/10/XMLSchema#')
    doc.add_namespace('mlflow', 'mlflow') #TODO: find namespaces of mlflow and prov-ml ontologies
    doc.add_namespace('prov-ml', 'prov-ml')
    
    doc = first_level_prov(active_run,doc)
    # doc = second_level_prov(active_run,doc)    

    #datasets are associated with two sets of tags: input tags, of the DatasetInput object, and the tags of the dataset itself
    # for input_tag in dataset_input.tags:
    #     attributes[f'mlflow:{input_tag.key.strip("mlflow.")}']=str(input_tag.value)
    # for key,value in ds_tags['tags'].items():
    #     attributes[f'mlflow:{str(key).strip("mlflow.")}']=str(value)

    graph_filename = f'provgraph_{PROV4ML_DATA.EXPERIMENT_NAME}.json'
    dot_filename = f'provgraph_{PROV4ML_DATA.EXPERIMENT_NAME}.dot'
    
    if not os.path.exists(PROV4ML_DATA.EXPERIMENT_DIR):
        os.makedirs(PROV4ML_DATA.EXPERIMENT_DIR, exist_ok=True)
    
    path_graph = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, graph_filename)

    # if PROV_SAVE_PATH and not os.path.exists(PROV_SAVE_PATH):
    #     os.makedirs(PROV_SAVE_PATH, exist_ok=True)

    with open(path_graph,'w') as prov_graph:
        doc.serialize(prov_graph)

    if create_graph:
        path_dot = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, dot_filename)
        with open(path_dot, 'w') as prov_dot:
            prov_dot.write(dot.prov_to_dot(doc).to_string())

    if create_svg:
        os.system(f"dot -Tsvg -O {path_dot}")
