for i in ./task.*/; do
    cd $i
	lmp -i in.lammps
	cd ../
done
