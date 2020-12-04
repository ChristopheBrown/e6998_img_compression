import os
import shutil

for root, dirs, files in os.walk(os.getcwd()):

	for all_roots in root:
		for the_dir in dirs:
			for file in files:
				if file endswith(".jpg"):
					print(all_roots + the_dir + file)