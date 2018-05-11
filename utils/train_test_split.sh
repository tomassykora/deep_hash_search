mkdir data_train
mkdir data_val
mkdir data_test
for class in fixed_data/*
	do 
	classname="$(basename $class)"
	mkdir data_train/$classname
	mkdir data_test/$classname
        mkdir data_val/$classname

	i=0;
	for file in $class/*
		do
		let "i++";
		if ( [ $i -lt 51 ] )
		then
			cp $file data_test/$classname/
		elif ( [ $i -lt 101 ] && [ $i -gt 50 ] )
               	then
		        cp $file data_val/$classname/
		else 
			cp $file data_train/$classname/
		fi
		done
	done
