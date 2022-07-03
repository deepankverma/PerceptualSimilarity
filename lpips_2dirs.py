import argparse
import os
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

print(opt)

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir0)
files1 = os.listdir(opt.dir1)

# print(files)
# print(files1)

for file in files:
	# print(file)
	# print((os.path.join(opt.dir0,file)))
	# if(os.path.exists(os.path.join(opt.dir1,file))):

		# print(file)
		# Load images
	for file1 in files1:
		# print(file1)
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file1)))
		# print((os.path.join(opt.dir0,file)),(os.path.join(opt.dir1,file1)))

		if(opt.use_gpu):
		  img0 = img0.cuda()
		  img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		# print('%s: %.3f'%(file,dist01))
		f.writelines('%s:%s: %.6f\n'%(file,file1,dist01))

f.close()
