from __future__ import absolute_import
import torch
import argparse
import numpy as np
import os
from models.resnext import resnet101
import torch.nn as nn
import pathlib
from dataset import VideoDataset
from dataset import FrameDataset
import torchvision
import transforms_vid
from multiprocessing import cpu_count
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool3d
from scipy import linalg


def get_activations(files, data_type, model, batch_size, size, length, dims, device):
	"""Calculates the activations of the pool_3 layer for all images.
	Params:
	-- files       : List of image files paths
	-- model       : Instance of inception model
	-- batch_size  : Batch size of images for the model to process at once.
					 Make sure that the number of samples is a multiple of
					 the batch size, otherwise some samples are ignored. This
					 behavior is retained to match the original FID score
					 implementation.
	-- dims        : Dimensionality of features returned by Inception
	-- device      : Device to run calculations
	Returns:
	-- A numpy array of dimension (num images, dims) that contains the
	   activations of the given tensor when feeding inception with the
	   query tensor.
	"""
	model.eval()

	if batch_size > len(files):
		print(('Warning: batch size is bigger than the data size. Setting batch size to data size'))
		batch_size = len(files)

	transform = torchvision.transforms.Compose([
		transforms_vid.ClipResize((size, size)),
		transforms_vid.ClipToTensor(),
		transforms_vid.ClipNormalize(mean=[0.4345, 0.4051, 0.3775], std=[0.2768, 0.2713, 0.2737])]
	)

	if data_type == 'video':
		ds = VideoDataset(files, length, transform)
	elif data_type == 'frame':
		ds = FrameDataset(files, length, transform)
	else:
		raise NotImplementedError
	dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, drop_last=False, num_workers=cpu_count())

	pred_arr = torch.zeros(len(files), dims).to(device)

	start_idx = 0

	for batch in tqdm(dl):

		batch = batch.to(device)

		with torch.no_grad():
			pred = model(batch)

		if pred.size(2) != 1 or pred.size(3) != 1 or pred.size(4) != 1:
			pred = adaptive_avg_pool3d(pred, output_size=(1, 1, 1))

		pred = pred.squeeze(4).squeeze(3).squeeze(2)
		pred_arr[start_idx:start_idx + pred.shape[0]] = pred
		start_idx = start_idx + pred.shape[0]

	pred_arr = pred_arr.cpu().numpy()

	return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
	"""Numpy implementation of the Frechet Distance.
	The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
	and X_2 ~ N(mu_2, C_2) is
			d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
	Stable version by Dougal J. Sutherland.
	Params:
	-- mu1   : Numpy array containing the activations of a layer of the
			   inception net (like returned by the function 'get_predictions')
			   for generated samples.
	-- mu2   : The sample mean over activations, precalculated on an
			   representative data set.
	-- sigma1: The covariance matrix over activations for generated samples.
	-- sigma2: The covariance matrix over activations, precalculated on an
			   representative data set.
	Returns:
	--   : The Frechet Distance.
	"""

	mu1 = np.atleast_1d(mu1)
	mu2 = np.atleast_1d(mu2)

	sigma1 = np.atleast_2d(sigma1)
	sigma2 = np.atleast_2d(sigma2)

	assert mu1.shape == mu2.shape, \
		'Training and test mean vectors have different lengths'
	assert sigma1.shape == sigma2.shape, \
		'Training and test covariances have different dimensions'

	diff = mu1 - mu2

	# Product might be almost singular
	covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

	if not np.isfinite(covmean).all():
		msg = ('fid calculation produces singular product; adding %s to diagonal of cov estimates') % eps
		print(msg)
		offset = np.eye(sigma1.shape[0]) * eps
		covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

	# Numerical error might give slight imaginary component
	if np.iscomplexobj(covmean):
		if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
			m = np.max(np.abs(covmean.imag))
			raise ValueError('Imaginary component {}'.format(m))
		covmean = covmean.real

	tr_covmean = np.trace(covmean)

	return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(files, data_type, model, batch_size, size, length, dims, device):
	"""Calculation of the statistics used by the FID.
	Params:
	-- files       : List of image files paths
	-- model       : Instance of inception model
	-- batch_size  : The images numpy array is split into batches with
					 batch size batch_size. A reasonable batch size
					 depends on the hardware.
	-- dims        : Dimensionality of features returned by Inception
	-- device      : Device to run calculations
	Returns:
	-- mu    : The mean over samples of the activations of the pool_3 layer of
			   the inception model.
	-- sigma : The covariance matrix of the activations of the pool_3 layer of
			   the inception model.
	"""
	act = get_activations(files, data_type, model, batch_size, size, length, dims, device)
	mu = np.mean(act, axis=0)
	sigma = np.cov(act, rowvar=False)

	return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, size, length, dims, device):

	if path.endswith('.npz'):
		f = np.load(path)
		m, s = f['mu'][:], f['sigma'][:]
		f.close()
	else:
		path = pathlib.Path(path)
		files = list(path.glob('*.mp4'))
		data_type = 'video'
		m, s = calculate_activation_statistics(files, data_type, model, batch_size, size, length, dims, device)

	return m, s


def calculate_fid_given_paths(paths, batch_size, size, length, dims, device):

	"""
	calculates the fid of two paths
	"""

	for p in paths:
		if not os.path.exists(p):
			raise RuntimeError('Invalid path: %s' % p)

	model = nn.DataParallel(resnet101(sample_duration=16).cuda())
	model.load_state_dict(torch.load('resnext-101-kinetics.pth')['state_dict'])

	m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, size, length, dims, device)
	m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, size, length, dims, device)

	fid_value = calculate_frechet_distance(m1, s1, m2, s2)

	return fid_value


def main(args):

	device = 'cuda'
	fid_value = calculate_fid_given_paths(args.path, args.batch_size, args.size, args.length, args.dims, device)

	print('FID: ', fid_value)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size to use')
	parser.add_argument('--dims', type=int, default=2048)
	parser.add_argument('--length', type=int, default=16, help=('number of frames in videos'))
	parser.add_argument('--size', type=int, default=112, help=('spatial size of videos'))
	parser.add_argument('path', type=str, nargs=2, help=('Paths to the generated images or to .npz statistic files'))

	args = parser.parse_args()

	main(args)
