cd work/t_P42nmc.STRU/

for i in ./task.*/; do
    cd $i
	OMP_NUM_THREADS=1 mpirun -n 4 abacus
	cd ../
done
