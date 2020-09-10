from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model.networks import Generator
import cfg
import skvideo.io
import numpy as np
import os


def save_videos(path, vids, n):

	for i in range(n):
		v = vids[i].permute(0,2,3,1).cpu().numpy()
		v *= 255
		v = v.astype(np.uint8)
		skvideo.io.vwrite(os.path.join(path, "%d.mp4"%(i)), v, outputdict={"-vcodec":"libx264"})

	return


def main():

	args = cfg.parse_args()

	# write into tensorboard
	log_path = os.path.join(args.demo_path, args.demo_name + '/log')
	vid_path = os.path.join(args.demo_path, args.demo_name + '/vids')

	if not os.path.exists(log_path) and not os.path.exists(vid_path):
		os.makedirs(log_path)
		os.makedirs(vid_path)
	writer = SummaryWriter(log_path)

	device = torch.device("cuda:0")

	G = Generator().to(device)
	G = nn.DataParallel(G)
	G.load_state_dict(torch.load(args.model_path))

	with torch.no_grad():
		G.eval()

		za = torch.randn(args.n, args.d_za, 1, 1, 1).to(device)
		zm = torch.randn(args.n, args.d_zm, 1, 1, 1).to(device)

		vid_fake = G(za, zm)

		vid_fake = vid_fake.transpose(2,1) # bs x 16 x 3 x 64 x 64
		vid_fake = ((vid_fake - vid_fake.min()) / (vid_fake.max() - vid_fake.min())).data

		writer.add_video(tag='generated_videos', global_step=1, vid_tensor=vid_fake)
		writer.flush()

		# save into videos
		print('==> saving videos...')
		save_videos(vid_path, vid_fake, args.n)

	return


if __name__ == '__main__':

	main()