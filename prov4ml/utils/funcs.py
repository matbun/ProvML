
def prov4ml_experiment_matches(experiment_name, exp_folder):
    """Check if the experiment name matches the experiment name in the provenance data."""
    exp_folder = "_".join(exp_folder.split("_")[:-1])
    return experiment_name == exp_folder