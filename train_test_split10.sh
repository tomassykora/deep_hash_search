mkdir data_train10
mkdir data_val10
mkdir data_test10
x=0
for class in fixed_data/*
	do
	let "x++"
	 if ( [ $x -gt 10 ] )
	then 
	exit
	fi 
	classname="$(basename $class)"
	mkdir data_train10/$classname
	mkdir data_test10/$classname
        mkdir data_val10/$classname

	i=0;
	for file in $class/*
		do
		let "i++";
		if ( [ $i -lt 51 ] )
		then
			cp $file data_test10/$classname/
		elif ( [ $i -lt 101 ] && [ $i -gt 50 ] )
               	then
		        cp $file data_val10/$classname/
		else 
			cp $file data_train10/$classname/
		fi
		done
	done
