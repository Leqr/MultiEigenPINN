from ImportFile import *

path_to_solved = os.getcwd()+"/Solved"

plt.figure()

for subdir, dirs, files in os.walk(path_to_solved):
    for file in files:
        path_to_file = dir + "/" + file
        if extension == ".pkl":
