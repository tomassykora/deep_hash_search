mkdir data_train5
mkdir data_val5
mkdir data_test5
x=0
for class in fixed_data/*
	do
	let "x++"
	 if ( [ $x -gt 5 ] )
	then 
	exit
	fi 
	classname="$(basename $class)"
	mkdir data_train5/$classname
	mkdir data_test5/$classname
        mkdir data_val5/$classname

	i=0;
	for file in $class/*
		do
		let "i++";
		if ( [ $i -lt 51 ] )
		then
			cp $file data_test5/$classname/
		elif ( [ $i -lt 101 ] && [ $i -gt 50 ] )
               	then
		        cp $file data_val5/$classname/
		else 
			cp $file data_train5/$classname/
		fi
		done
	done
