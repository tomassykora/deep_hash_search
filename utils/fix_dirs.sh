mkdir fixed_data
for letter in data/*
	do
	echo "$letter"
	for class in $letter/*
		do
		echo $class
		for file in $class/*
			do
			echo "$file"
			if [[ -d $file ]]; then
			    echo "$file is a directory"
		            classname="$(basename $class)_$(basename $file)"
			    if ! [[ -d "fixed_data/$classname" ]]; then
				    mkdir "fixed_data/$classname"
			    fi
			    for f in $file/*
				do 
				cp $f fixed_data/$classname/
				done
			elif [[ -f $file ]]; then
	                    classname=$(basename $class)
                            if ! [[ -d "fixed_data/$classname" ]]; then
                                    mkdir "fixed_data/$classname"
                            fi
                            cp $file fixed_data/$classname/
	
			fi
			done
		done
	done
