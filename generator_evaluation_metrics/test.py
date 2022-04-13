import unittest

import torch

from evaluate_generator import evaluate_generator

N_samples = 16
N_batch = N_samples // 4

object_shape = (10,)
encoded_shape = (3,)
label_shape = (1,)


class GeneratorDummy(torch.nn.Module):
	dim_z = encoded_shape[0]
	def forward(self, latent, label):
		return torch.randn(latent.shape[0], *object_shape)


class EncoderDummy(torch.nn.Module):
	def forward(self, sample):
		return torch.randn(sample.shape[0], *encoded_shape)


def creator_encoder_dummy():
	return EncoderDummy()


class DataSetDummy(torch.utils.data.Dataset):
	def __init__(self, N_samples):
		self.samples = [(torch.randn(*object_shape), torch.randn(*label_shape))
						 for _ in range(N_samples)]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		return self.samples[idx]

dataset_dummy = DataSetDummy(N_samples)
dataloader_dummy = torch.utils.data.DataLoader(dataset_dummy, batch_size=N_batch, shuffle=True)


class EvaluationTest(unittest.TestCase):

	def test_evaluate_generator(self):
		generator = GeneratorDummy()  ## should use pre-trained parameters. Has to entail .dim_z = expected latent dimension
		generator.eval()
		dataloader_target = dataloader_dummy
		dataloader_reference = dataloader_dummy
		encoders = {"encoder_dummy": creator_encoder_dummy}
		metrics = ["cluster", "wasserstein"]
		seed = 1



		results = evaluate_generator(
					   generator,
                       generator_name = "generator_dummy",
                       data_loader_target = dataloader_target,
                       data_loader_reference = dataloader_reference,
                       seed = seed,
                       metrics = metrics,
                       encoders = encoders,
                       )

		print(results)

if __name__ == '__main__':
	unittest.main()