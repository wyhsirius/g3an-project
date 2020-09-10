import torch
import torch.nn as nn
from .ops import G3, FSA
import torch.nn.utils.spectral_norm as spectralnorm
from torch.nn import init

# def weights_init_normal(m):
# 	classname = m.__class__.__name__
# 	if classname.find("Conv") != -1:
# 		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
# 	elif classname.find("BatchNorm") != -1:
# 		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
# 		torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
	def __init__(self, c_a=128, c_m=10, ch=64, mode='1p2d', use_attention=True):
		super(Generator, self).__init__()

		self.use_attention = use_attention

		self.block1 = G3(c_a, c_a+c_m,  c_m, ch*8, mode, 4, 4, 1, 1, 0, 0) # 4 x 4 x 4
		self.block2 = G3(ch*8, ch*8*2, ch*8, ch*8, mode, 4, 4, 2, 2, 1, 1) # 8 x 8 x 8
		self.block3 = G3(ch*8, ch*8*2, ch*8, ch*4, mode, 4, 4, 2, 2, 1, 1) # 16 x 16 x 16
		self.block4 = G3(ch*4, ch*4*2, ch*4, ch*2, mode, 4, 1, 2, 1, 1, 0) # 16 x 32 x 32
		self.block5 = G3(ch*2, ch*2*2, ch*2,   ch, mode, 4, 1, 2, 1, 1, 0) # 16 x 64 x 64

		if self.use_attention:
			self.fsa = FSA(ch*4)

		self.to_rgb = nn.Sequential(
			nn.Conv3d(ch*2, 3, (1,3,3), 1, (0,1,1)),
			nn.Tanh()
		)

		self.init_weights()

	def init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Conv3d):
				init.normal_(module.weight, 0, 0.02)
			elif isinstance(module, nn.BatchNorm3d):
				init.normal_(module.weight.data, 1.0, 0.02)
				init.constant_(module.bias.data, 0.0)

	def forward(self, za, zm):

		#za = za.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		#zm = zm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
		zv = torch.cat([za.repeat(1,1,zm.size(2),1,1), zm], 1)

		hs, hv, ht = self.block1(za, zv, zm)
		hs, hv, ht = self.block2(hs, hv, ht)
		hs, hv, ht = self.block3(hs, hv, ht)
		hs, hv, ht = self.block4(hs, hv, ht)
		if self.use_attention:
			hv = self.fsa(hv)
		hs, hv, ht = self.block5(hs, hv, ht)
		out = self.to_rgb(hv)

		return out


class VideoDiscriminator(nn.Module):
	def __init__(self, ch=64):
		super(VideoDiscriminator, self).__init__()

		self.net = nn.Sequential(
			spectralnorm(nn.Conv3d(3,   ch, (1,4,4), (1,2,2), (0,1,1))),
			nn.LeakyReLU(0.2, inplace=True),
			spectralnorm(nn.Conv3d(ch, ch, (4,1,1), 1, 0)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv3d(ch, ch*2, (1,4,4), (1,2,2), (0,1,1))),
			nn.LeakyReLU(0.2, inplace=True),
			spectralnorm(nn.Conv3d(ch*2, ch*2, (4,1,1), 1, 0)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv3d(ch*2, ch*4, (1,4,4), (1,2,2), (0,1,1))),
			nn.LeakyReLU(0.2, inplace=True),
			spectralnorm(nn.Conv3d(ch*4, ch*4, (4,1,1), 1, 0)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv3d(ch*4, ch*8, (1,4,4), (1,2,2), (0,1,1))),
			nn.LeakyReLU(0.2, inplace=True),
			spectralnorm(nn.Conv3d(ch*8, ch*8, (4,1,1), 1, 0)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv3d(ch*8, ch*8, (1,4,4), 1, 0)),
			nn.LeakyReLU(0.2, inplace=True),
			spectralnorm(nn.Conv3d(ch*8,    1, (4,1,1), 1, 0))
		)

		self.init_weights()

	def init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Conv3d):
				init.normal_(module.weight, 0, 0.02)

	def forward(self, x):

		out = self.net(x)

		return out.squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1)


class ImageDiscriminator(nn.Module):
	def __init__(self, ch=64):
		super(ImageDiscriminator, self).__init__()

		self.net = nn.Sequential(
			spectralnorm(nn.Conv2d(3, ch, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv2d(ch, ch*2, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv2d(ch*2, ch*4, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv2d(ch*4, ch*8, 4, 2, 1)),
			nn.LeakyReLU(0.2, inplace=True),

			spectralnorm(nn.Conv2d(ch*8, 1, 4, 1, 0)),
		)

		self.init_weights()

	def init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				init.normal_(module.weight, 0, 0.02)

	def forward(self, x):

		out = self.net(x)

		return out.squeeze(-1).squeeze(-1).squeeze(-1)


if __name__ == '__main__':

	za = torch.randn(1, 128, 1, 1, 1)
	zm = torch.randn(1, 10, 4, 1, 1)
	#za = za.repeat(1, 1, 3, 1, 1)
	net = Generator()
	out = net(za, zm)
	print(out.size())

	# d = VideoDiscriminator()
	# out = d(out)
	# print(out.size())
	#
	# x = torch.rand(4, 3, 64, 64)
	# dd = ImageDiscriminator()
	# out = dd(x)
	# print(out.size())
