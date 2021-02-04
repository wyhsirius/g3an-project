import skvideo.io
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import json
import av
import glob
import torchvision
import transforms_vid

class VideoDataset(Dataset):
	def __init__(self, files, length, transform=None):

		self.files = files

		self.length = length
		self.transform = transform

	def __len__(self):

		return len(self.files)

	def __getitem__(self, i):

		path = self.files[i]
		videogen = av.open(str(path))
		video = [frame.to_image() for frame in videogen.decode(video=0)]
		nframes = len(video)

		#start = nframes // 2 - self.length
		#video = video[start : start + self.length*2 : 2]
		
		start = nframes // 2 - self.length // 2
		#print(start, start + self.length)
		video = video[start : start + self.length]

		# transform
		if self.transform is not None:
			video = self.transform(video)

		return video


class FrameDataset(Dataset):
	def __init__(self, files, length, transform=None):

		self.files = files

		self.length = length
		self.transform = transform

	def __len__(self):

		return len(self.files)

	def __getitem__(self, i):

		video_path = self.files[i]
		frames_path = sorted(glob.glob(video_path + '/*.png'))

		video = [Image.open(path).convert('RGB') for path in frames_path]
		nframes = len(video)

		# temporal center cropping
		start = nframes // 2 - self.length // 2
		video = video[start : start + self.length]

		# transform
		if self.transform is not None:
			video = self.transform(video)

		return video


