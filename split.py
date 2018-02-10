#!/usr/bin/env python

import sys, subprocess, uuid, os, math, shutil

if len(sys.argv) != 3 and len(sys.argv) != 4:
    print('usage: {0} machinefile svm_file [split_svm_file]'.format(sys.argv[0]))
    sys.exit(1)
machinefile_path, src_path = sys.argv[1:3]

machines = []
for line in open(machinefile_path):
    machine = line.strip()
    machines.append(machine)
nr_machines = len(machines)

src_basename = os.path.basename(src_path)
if len(sys.argv) == 4:
    dst_path = sys.argv[3]
else:
    dst_path = '{0}'.format(src_basename)
if (nr_machines == 1):
    temp_path = src_path
    temp_dst_path = dst_path
    if machine == '127.0.0.1' or machine == 'localhost':
        if (len(sys.argv) != 4):
            cmd = ''
        else:
            cmd = 'cp {0} {1}'.format(temp_path, temp_dst_path)
    else:
        cmd = 'scp {0} {1}:{2}'.format(temp_path, machine,
                                       os.path.join(os.getcwd(), temp_dst_path))
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    p.communicate()
    print('The data has been copied to {0}'.format(machine))
    sys.exit(1)

cmd = 'wc -l {0}'.format(src_path)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
nr_instances = int(p.stdout.read().strip().split()[0])
p.communicate()

while True:
    temp_dir = 'tmp_{0}'.format(uuid.uuid4())
    if not os.path.exists(temp_dir): break
os.mkdir(temp_dir)

print('Spliting data...')
nr_digits = int(math.log10(nr_machines))+1
cmd = 'split -l {0} --numeric-suffixes -a {1} {2} {3}.'.format(
          int(math.ceil(float(nr_instances)/nr_machines)), nr_digits, src_path,
          os.path.join(temp_dir, src_basename))
p = subprocess.Popen(cmd, shell=True)
p.communicate()

for i, machine in enumerate(machines):
    temp_path = os.path.join(temp_dir, src_basename + '.' + 
                             str(i).zfill(nr_digits))
    temp_dst_path = dst_path + '.' + str(i).zfill(nr_digits)
    if machine == '127.0.0.1' or machine == 'localhost':
        cmd = 'mv {0} {1}'.format(temp_path, temp_dst_path)
    else:
        cmd = 'scp {0} {1}:{2}'.format(temp_path, machine,
                                       os.path.join(os.getcwd(), temp_dst_path))
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    p.communicate()
    print('The subset of data has been copied to {0}'.format(machine))

shutil.rmtree(temp_dir)
