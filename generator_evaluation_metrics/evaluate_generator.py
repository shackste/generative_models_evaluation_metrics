""" This file contains the main code to be executed for evaluation of a galaxy image generator.py
    The code is build using pytorch and assuming pytorch models.
"""
from types import GeneratorType
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from results import Results
from cluster_measures.evaluate_cluster_distribution import evaluate_distribution
from distribution_measures.wasserstein import wasserstein

InceptionV3 = 1  ## !!!! used for testing. remove this, once the encoder is loaded from other file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

compute_metric = { ## these have to be loaded from separate files
#    "IS" : inception_score,
    "cluster" : partial(evaluate_distribution, combined=True),
    "cluster separate" : partial(evaluate_distribution, combined=False),
    "wasserstein" : partial(wasserstein, blur=0.005, scaling=0.95, splits=4)
}

@torch.no_grad()
def evaluate_generator(generator, *,
                       generator_name: str = None,
                       data_loader_target: DataLoader = None,
                       data_loader_reference: DataLoader = None,
                       seed: int = None,
                       metrics: list = ["IS", "FID", "KID", "PPL", "chamfer", "wasserstein", "cluster", "AggregatedLabelControlAccuracy"],
                       encoders: dict = {"InceptionV3": InceptionV3},
                       N_cluster: int = 10, # number of clusters used in cluster measures
                       ) -> Results:
    """ Compute evaluation metrics for the galaxy generator model as well as reference values.
        For each metric, the reference is obtained by comparing two separate subsets from the original data, named target and test set.
        The evaluation metric is obtained by generating images using labels in the test set and comparing the results to the target set.
        All metrics use a representation in reduced dimensions.

        Parameters
        ----------
        generator : pytorch Module
            pre-loaded generator instance in eval mode, ready for evaluation
        generator_name : string
            identifier of the generator. used for clarity when comparing several generators
        data_loader_target : DataLoader object
            iterable that returns a batch of (samples, label) from the target set
        data_loader_reference : DataLoader object
            iterable that returns a batch of (samples, label) from the set of reference.
            These labels are used to generate samples
        seed : float, default=None
            seed used for random numbers in numpy, torch and torch.cuda to obtain identical results in separate calls
        metrics : list of strings
            list of metric names to be computed
        encoders : dict (name, reduction model creator)
            contains the models used for dimension reduction.
            These should be provided as a creator function, which returns the pre-trained model ready for dimension reduction.
    """

    ## force same results in every call
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    ## use generic name, if none was given
    if not generator_name: generator_name = "generator"

    ## set number of clusters used for cluster metrics
    for cluster_key in ["cluster", "cluster separate"]:  ### this is an ugly workaround. Can you think of something better?
        if cluster_key in metrics: compute_metric[cluster_key] = partial(compute_metric[cluster_key], N_cluster=N_cluster)

    results = Results()
    for enc_name, Enc in encoders.items():
        ## reduce dimensions
        enc = Enc().to(device).eval()
        encoded_generated = get_features(image_generator(generator, data_loader_reference), enc)
        encoded_test = get_features(data_loader_reference, enc)
        encoded_target = get_features(data_loader_target, enc)
        del enc
        ## compute evaluation metrics
        for metric_name in metrics:
            value_reference = compute_metric[metric_name](encoded_target, encoded_test)
            value_generator = compute_metric[metric_name](encoded_target, encoded_generated)
            results.append(value_reference, model="reference", encoder=enc_name, metric=metric_name)
            results.append(value_generator, model=generator_name, encoder=enc_name, metric=metric_name)
    return results




def get_features(dataloader,  # iterable generator that provides a tuple (images, dummy).
                 encoder # encoder used for dimensions reduction.
                ) -> torch.Tensor:
    """ obtain features using encoder on all samples in the dataloader """
    features = torch.cat(
        [encoder(samples.to(device)) for samples, _ in dataloader],
        dim=0
    )
    return features


def image_generator(model, # pytorch generator model that takes latent vector and label vector as input. has to contain dim_z = latent dimension
                    dataloader: DataLoader, # dataloader to provide labels for the predicted distribution. provides tuples of (dummy, labels)
                    ) -> GeneratorType:
    for _, labels in tqdm(dataloader, desc=f"generate images {type(model).__name__}"):
        latent = torch.randn(dataloader.batch_size, model.dim_z, device=device)
        labels = labels.to(device)
        images = model(latent, labels)
        yield images, labels
