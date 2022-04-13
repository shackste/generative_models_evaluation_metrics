# Evaluation Metrics for generative models.

This repo contains code to evaluate the quality of generative models.

These include Cluster-based metrics that use k-meas in reduced feature space to assess the distribution of data.
Furthermore, it entails several metrics commonly used troughout the machine learning community: Incepton Score (IS), Frechet Inception Distance (FID), Kernel Inception Distance (KID), Perceptual Path Lenght (PPL), Chamfer distance, Geometric Distance, Wasserstein distance.
Moreover, Aggregated Attribute Control Accuracy is included for labels.
In addition, for the case of generating images of galaxies, morphological statistics obtained using the statmorph code () are used to assess the physical soundness of the generated galaxy images.

This code has been developed for the publication by Hackstein, Kinakh, Bailer and Melchior 2022 (in preparation).
When using this code in a publication, please refer to that article.

## Usage