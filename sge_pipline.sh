#!/bin/bash

#get the groups
#get the number of highpass files
sortfile=hmmsort
sortwindow=100000
sortp=1e-10
hfiles=`ls *_highpass.[0-9]* | grep -v mat`
baseh=`echo ${hfiles} | awk -F "." '{print $1 }' | uniq`
base=`echo ${baseh} | awk -F "_" '{ for(i=1;i<NF;i++) {print $i}}' | paste -s -d "_" -`
#base=`echo ${baseh} | awk -F "_" '{print $1}'`
groups=`awk '/Active/ {print $3}' ${base}_descriptor.txt | sort | uniq`
nfiles=`echo $hfiles | awk '{print NF}'`
echo $nfiles
#loop through each group
for g in $groups
do
	if [ $g -gt 0 ]
	then
		fname=`printf "%sg%.4d" $base $g`
		echo $fname
		i=1
		if [ ! -e ${sortfile}g${g}.mat ]
		then
			jobid=`echo "cd $PWD;$HOME/Documents/matlab/hmmsort_example_ForRoger/run_hmm_learn_tetrode.sh /Applications/MATLAB_R2010a.app/ $baseh.0001 ${sortfile}g$g Group $g" | qsub -j y -V -N hmmLearng${g} -o $HOME/tmp/ -l mem=20G | awk '{print $3}'`
			while [ $i -le $nfiles ]; do f=$fname.`printf "%.4d" $i`.mat; test -e $f|| echo "cd $PWD;touch $f; hostname; $HOME/Documents/matlab/hmmsort_example_ForRoger/run_hmm_decode.sh /Applications/MATLAB_R2010a.app/ ${sortfile}g$g $sortwindow $sortp SourceFile $baseh.$( printf "%.4d" $i ) Group $g save;test -s $f || rm $f"| qsub -j y -V -N hmmDecode$g$i -o $HOME/tmp/ -l mem=2G -hold_jid $jobid; let i=$i+1;done
			#echo "cd $PWD;$HOME/Documents/matlab/hmmsort_example_ForRoger/run_hmm_learn_tetrode.sh /Applications/MATLAB_R2010a.app/ $baseh.0001 ${sortfile}g$g Group $g" > $HOME/tmp/sortoutput.$g
			
		else	
		#for i in {1..${nfiles}}; $f=$fname.$( printf "%.4d" $i ).mat test -e $f|| echo "cd $PWD;touch $f; hostname; $HOME/Documents/matlab/hmmsort_example_ForRoger/run_hmm_decode.sh /Applications/MATLAB_R2010a.app/ ${sortfile}g$g $sortwindow $sortp SourceFile $baseh.$( printf "%.4d" $i ) Group $g save;test `ls -l $f | awk '{print $5}'` = 0 && rm $f" | qsub -j y -V -N hmmDecode -o $HOME/tmp/ -l mem=2G -hold_jid $jobid; done
			while [ $i -le $nfiles ]; do f=$fname.`printf "%.4d" $i`.mat; test -e $f|| echo "cd $PWD;touch $f; hostname; $HOME/Documents/matlab/hmmsort_example_ForRoger/run_hmm_decode.sh /Applications/MATLAB_R2010a.app/ ${sortfile}g$g $sortwindow $sortp SourceFile $baseh.$( printf "%.4d" $i ) Group $g save;test -s $f || rm $f"| qsub -j y -V -N hmmDecode$g$i -o $HOME/tmp/ -l mem=2G ; let i=$i+1;done
		fi
		#while [ $i -le $nfiles ]; do f=$fname.`printf "%.4d" $i`.mat; test -e $f|| echo "cd $PWD;touch $f; hostname; $HOME/Documents/matlab/hmmsort_example_ForRoger/run_hmm_decode.sh /Applications/MATLAB_R2010a.app/ ${sortfile}g$g $sortwindow $sortp SourceFile $baseh.$( printf "%.4d" $i ) Group $g save;test -s $f || rm $f" > $HOME/tmp/output.$i; let i=$i+1;done
	fi
done
