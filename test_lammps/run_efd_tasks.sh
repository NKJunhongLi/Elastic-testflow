cd work/CONTCAR.lmp/

cd EFD_task/
for i in ./*/; do
    cd $i
	/home/lijh/anaconda3/envs/deepmd/bin/lmp -i in.lammps
    cd ../
done
cd ../

for i in task.*/; do
    cd $i
    cd EFD
	for j in ./*/; do
        cd $j
        /home/lijh/anaconda3/envs/deepmd/bin/lmp -i in.lammps
        cd ../
    done
    cd ../../
done
