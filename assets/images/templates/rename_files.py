import os
import shutil
from pathlib import Path

def copy_rename(fsub, ftype, old_file_name, new_file_name):
	Path(fsub).mkdir(parents=True, exist_ok=True)

	src_dir = os.getcwd()
	dst_dir= os.path.join(os.getcwd() , fsub)
	src_file = os.path.join(src_dir, old_file_name + "." + ftype)
	shutil.copy(src_file,dst_dir)

	dst_file = os.path.join(dst_dir, old_file_name + "." + ftype)
	new_dst_file_name = os.path.join(dst_dir, new_file_name + "." + ftype)
	os.rename(dst_file, new_dst_file_name)

def copy_rename_series(fsub, start, ftype, series):
	src_dir = os.getcwd()

	files = [x for i,x in enumerate(os.listdir(src_dir)) if ftype in str(x)]
	files = [x.strip("." + ftype) for x in files]

	for count, filename in enumerate(files):
		new_file_name = series + "_" + str(count+int(start))
		if filename != ".DS_store":
			copy_rename(fsub, ftype, filename, new_file_name)

print()
multiple = input("Do you want to rename one file? [y/n]: ")
if multiple == "y":
	print()
	ftype = input("What is the file type?: ")
	print()
	fsub = input("What is the new folder name?: ")
	print()
	old_file_name = input("what is the old filename?: ")
	print()
	new_file_name = input("what is the new filename?: ")
	print()
	copy_rename(fsub, ftype, old_file_name, new_file_name)
if multiple == "n":
	print()
	ftype = input("What is the file type?: ")
	print()
	start = input("What is the starting range?: ")
	print()
	fsub = input("What is the new folder name?: ")
	print()
	series = input("What is the series name?: ")
	print()
	copy_rename_series(fsub, start, ftype, series)

print("[Done]") 

