import numpy as np
import torch
import argparse
import fid
from models.resnext import resnet101
import torch.nn as nn
import glob

def main(args):

	device = 'cuda'

	print('Loading ResNext101 model...')
	model = nn.DataParallel(resnet101(sample_duration=16).cuda())
	model.load_state_dict(torch.load('resnext-101-kinetics.pth')['state_dict'])

	print('Loading video paths...')

	if args.dataset == 'uva':
		files = glob.glob(args.data_path + '*.mp4')
		data_type = 'video'
	else:
		raise NotImplementedError
	mu, sigma = fid.calculate_activation_statistics(files, data_type, model, args.batch_size, args.size, args.length, args.dims, device)
	np.savez_compressed('./stats/'+args.dataset+'.npz', mu=mu, sigma=sigma)

	print('finished')


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size to use')
	parser.add_argument('--length', type=int, default=16, help=('number of frames in videos'))
	parser.add_argument('--size', type=int, default=112, help=('spatial size of videos'))
	parser.add_argument('--dims', type=int, default=2048)
	parser.add_argument('--dataset', type=str, default='uva')
	parser.add_argument('--data_path', type=str)	

	args = parser.parse_args()

	main(args)
