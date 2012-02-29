#!/bin/bash
#BINDIR=/opt/cluster/usr/bin/roger
BINDIR=/opt/cluster/tmp
#get the groups
#get the number of highpass files
sortfile=hmmsort
sortwindow=100000
sortp=1e-10
chunksize=50000
hfiles=`ls *_highpass.[0-9]* | grep -v mat`
baseh=`echo ${hfiles} | awk -F "." '{print $1 }' | uniq`
base=`echo ${baseh} | awk -F "_" '{ for(i=1;i<NF;i++) {print $i}}' | paste -s -d "_" -`
#base=`echo ${baseh} | awk -F "_" '{print $1}'`
groups=`awk '/Active/ {print $3}' ${base}_descriptor.txt | sort | uniq`
nfiles=`echo $hfiles | awk '{print NF}'`
outfiles=''
echo $nfiles
#loop through each group
for g in $groups
do
	if [ $g -gt 0 ]
	then
		fname=`printf "%sg%.4d" $base $g`
		echo $fname
		i=1
		while [ $i -le $nfiles ]
		do
			nr=$( printf %.4d $i )
			outfile=${baseh}g$( printf %.4d $g ).$( printf %.4d $i ).hdf5
			outfiles=${outfiles}${outfile},
			#break the file into task chunks of 3 million data points each
			nchunks=`ls -l ${baseh}.${nr} | awk '{print int($5/3000000/36/2+0.5)}'`
			if [ $nchunks -gt 0 ]
			then
				c=1
				while [ $c -le $nchunks ]
				do
					outfile=${baseh}g$( printf %.4d $g ).$( printf %.4d $i ).$c.hdf5
					if [ ! -e $PWD/hmmsort/$outfile ]
					then
						jobid[$i]=`echo "touch $PWD/hmmsort/${outfile};cp $PWD/${baseh}.${nr} /tmp/; cp $PWD/*descriptor.txt /tmp/; cd /tmp/;SGE_TASK_ID=$c SGE_TASK_FIRST=1 SGE_TASK_LAST=$nchunks $BINDIR/hmm_learn_tetrode.py --sourceFile $baseh.${nr} --group $g --chunkSize $chunksize ;cp /tmp/${outfile} ${PWD}/hmmsort/; rm /tmp/${outfile};rm /tmp/${baseh}.${nr}" | qsub -j y -V -N hmmLearn${base}${g}_${i}_$c -o $HOME/tmp/ -e $HOME/tmp/ -l mem=5G -l s_rt=7000 -soft -l paths=*$PWD*| awk '{print $3}'| awk -F . '{print $1}'`
					fi
					let c=$c+1
				done
			else
				outfile=${baseh}g$( printf %.4d $g ).$( printf %.4d $i ).hdf5
				if [ ! -e $PWD/hmmsort/$outfile ]
				then
					jobid[$i]=`echo "touch $PWD/hmmsort/${outfile};cp $PWD/${baseh}.${nr} /tmp/; cp $PWD/*descriptor.txt /tmp/; cd /tmp/;$BINDIR/hmm_learn_tetrode.py --sourceFile $baseh.${nr} --group $g --outFile ${outfile} --chunkSize $chunksize ;cp /tmp/${outfile} ${PWD}/hmmsort/; rm /tmp/${outfile}; rm /tmp/${baseh}.${nr}" | qsub -j y -V -N hmmLearn${base}${g}_$i -o $HOME/tmp/ -e $HOME/tmp/ -l mem=5G -l s_rt=7000 -soft -l paths=*$PWD* | awk '{print $3}' | awk -F . '{print $1}'`
				fi
			fi
			let i=$i+1
		done
		jobidstr=`echo ${jobid[*]} | sed -e 's/ /,/g'`
		#one job to gather all the results
		if [ ! -e hmmsort/${base}g$( printf %.4d $g ).hdf5 ]
		then
			if [  ${#jobid[*]} -gt 0 ]
			then
				newjobid=`echo "cd $PWD; $BINDIR/hmm_learn_tetrode.py --sourceFile ${outfiles} --combine --group $g"| qsub -j y -V -N hmmGather${base}$g -o $HOME/tmp/ -e $HOME/tmp/ -l mem=20G -hold_jid $jobidstr -m e -M roger.herikstad@gmail.com`
			else
				newjobid=`echo "cd $PWD; $BINDIR/hmm_learn_tetrode.py --sourceFile ${outfiles} --combine --group $g"| qsub -j y -V -N hmmGather${base}$g -o $HOME/tmp/ -e $HOME/tmp/ -l mem=20G -l -m e -M roger.herikstad@gmail.com`
			fi
		fi
		#reset i
		i=0
		while [ $i -le $nfiles ]; do f=$fname.`printf "%.4d" $i`.mat; test -e hmmsort/$f|| echo "cd $PWD;touch $f; hostname; $BINDIR/run_hmm_decode.sh /Applications/MATLAB_R2010a.app/ hmmsort/${sortfile}g$g $sortwindow $sortp SourceFile $baseh.$( printf "%.4d" $i ) Group $g save hdf5;test -s $f || rm $f"| qsub -j y -V -N decode${base}$g_$i -o $HOME/tmp/ -e $HOME/tmp/ -l mem=2G -l s_rt=7000  -hold_jid $newjobid; let i=$i+1;done

	fi
done
#echo "cd $PWD; /opt/cluster/tmp/hmm_learn_tetrode.py --sourceFile tiger_p6_misc_highpass.0008 --group 4 --chunkSize=50000" | qsub -j y -V -N hmmLearng4 -l mem=5G -l s_rt=7000 -t 1-20 -o $HOME/tmp/

