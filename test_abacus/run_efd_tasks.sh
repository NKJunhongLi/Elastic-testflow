cd work/t_P42nmc.STRU/

cd EFD_task/
for i in ./*/; do
    cd $in
	OMP_NUM_THREADS=1 mpirun -n 4 abacus
    cd ../
done
cd ../

for i in task.*/; do
    cd $i
    cd EFD
	for j in ./*/; do
        cd $j
        OMP_NUM_THREADS=1 mpirun -n 4 abacus
        cd ../
    done
    cd ../../
done
