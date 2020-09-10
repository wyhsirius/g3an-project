import torch
import random

def train(args, epoch, G, VD, ID, optimizer_G, optimizer_VD, optimizer_ID, criterion, dataloader, writer, device):

	# train mode
	G.train()
	VD.train()
	ID.train()

	for i, x in enumerate(dataloader):

		global_steps = epoch * len(dataloader) + i
		bs = x.size(0)
		real_vid = x.to(device)
		real_img = real_vid[:,:,random.randint(0, x.size(2)-1), :, :]

		#################### train D ##################
		optimizer_VD.zero_grad()
		optimizer_ID.zero_grad()

		VD_real = VD(real_vid)
		ID_real = ID(real_img)

		za = torch.randn(bs, args.d_za, 1, 1, 1).to(device)
		zm = torch.randn(bs, args.d_zm, 1, 1, 1).to(device)

		fake_vid = G(za, zm)
		fake_img = fake_vid[:,:, random.randint(0, x.size(2)-1),:,:]
		VD_fake = VD(fake_vid.detach())
		ID_fake = ID(fake_img.detach())

		y_real = torch.ones(VD_real.size()).to(device)
		y_fake = torch.zeros(VD_fake.size()).to(device)

		errVD = criterion(VD_real, y_real) + criterion(VD_fake, y_fake)
		errID = criterion(ID_real, y_real) + criterion(ID_fake, y_fake)
		
		errVD.backward()
		optimizer_VD.step()

		errID.backward()
		optimizer_ID.step()

		################## train G ###################
		optimizer_G.zero_grad()

		VG_fake = VD(fake_vid)
		IG_fake = ID(fake_img)

		errVG = criterion(VG_fake, y_real)
		errIG = criterion(IG_fake, y_real)
		errG = errVG + errIG
		
		errG.backward()
		optimizer_G.step()

		writer.add_scalar('G_vid_loss', errVG.item(), global_steps)
		writer.add_scalar('G_img_loss', errIG.item(), global_steps)
		writer.add_scalar('D_vid_loss', errVD.item(), global_steps)
		writer.add_scalar('D_img_loss', errID.item(), global_steps)
		writer.flush()

		if global_steps % args.print_freq == 0:
			print("[Epoch %d/%d] [Iter %d/%d] [VD loss: %f] [VG loss: %f] [ID loss: %f] [IG loss: %f]"
			      %(epoch, args.max_epoch, i, global_steps, errVD.item(), errVG.item(), errID.item(), errIG.item()))


def vis(epoch, G, za, zm, device, writer):

	with torch.no_grad():

		G.eval()
			
		za = za.to(device)
		zm = zm.to(device)
		n_za = za.size(0)
		n_zm = zm.size(0)

		za = za.unsqueeze(1).repeat(1,n_zm,1,1,1,1).contiguous().view(n_za*n_zm,-1,1,1,1)
		zm = zm.repeat(n_za,1,1,1,1)

		vid_fake = G(za, zm)
		
		vid_fake = vid_fake.transpose(2,1)
		vid_fake = ((vid_fake - vid_fake.min()) / (vid_fake.max() - vid_fake.min())).data
		
		writer.add_video(tag='generated_videos', global_step=epoch, vid_tensor=vid_fake)
		writer.flush()

