#ifndef _VITERBI_HPP_INCLUDED
#define _VITERBI_HPP_INCLUDED
#include "chainviterbi.cpp"


double viterbi_nd(vec<double> &observed_data,
		vec<double> &p,
		double *cinv,
		vec<unsigned int> &numstates,
		vec<double> *mu_cell,
		vec<double> &initial_guess,
		vec<state_type> &mlseq,
		const int overlaps=2,
		const int dim=1);

double viterbi_nd_multiring(vec<double> &observed_data,
		double *cinv,
		int nummatrices,
		double *trmatrices,
		int *trmatlenths,
		int *activestates,
		vec<double> *mu_cell,
		vec<double> &initial_guess,
		vec<vec<int> > &mlseq,
		vec<hmm_tr_mat_entry> *custom_transitions=NULL,
		int overlaps = 2,
		const int dim = 1);

#ifndef WIN32

double viterbi_nd_record(vec<double> &observed_data,
		vec<double> &p,
		double *cinv,
		vec<unsigned int> &numstates,
		vec<double> *mu_cell,
		vec<double> &initial_guess,
		vec<state_type> &mlseq,
		const int overlaps=2,
		const int dim=1,
		const char * filename="record");

void decode_from_file(vec<state_type> &mlseq,
		vec<double> &descisions,
		const char * filename);

void decode_from_file_with_error(vec<state_type> &mlseq,
		vec<bool> &errors,
		const char * filename);
#endif

void get_shift_p_idx(vec<unsigned int> &numstates,const int overlaps,vec<state_type> &states){
	statespace_enum st(numstates,overlaps);

	vec<state_type> nontrivialstates;
	vec<state_type> trivialstates;
	st.getStates(trivialstates,nontrivialstates);

	unsigned int states_count=0;
	for (unsigned int state_idx=0;state_idx<nontrivialstates.len;state_idx++){
		state_type state=nontrivialstates[state_idx];
		vec<state_type> *tmptable = generatelookuptable(state,numstates,overlaps);
		states_count += tmptable->len;
		delete tmptable;
	}
	states.setLength(states_count);
	unsigned int p_idx=0;
	for (unsigned int state_idx=0;state_idx<nontrivialstates.len;state_idx++){
		state_type state=nontrivialstates[state_idx];
		vec<state_type> *tmptable = generatelookuptable(state,numstates,overlaps);
		for (int j=0;j<tmptable->len;j++){
			states[p_idx]=tmptable->data[j];
			p_idx++;
		}
		delete tmptable;
	}
}
////////////////////////////end.cheap stuff

double viterbi_nd_multiring(vec<double> &observed_data,
		double *cinv,
		int nummatrices,
		double *trmatrices,
		int *trmatlenths,
		int *activestates,
		vec<double> *mu_cell,
		vec<double> &initial_guess,
		vec<vec<int> > &mlseq,
		vec<hmm_tr_mat_entry> *custom_transitions,
		int overlaps,
		const int dim){

	bool custom=(custom_transitions != NULL);

	overlaps++;
	double **trmats = new double*[nummatrices];
	int offset=0;
	for (int i=0;i<nummatrices;i++) {
		trmats[i]=&trmatrices[offset];
		offset += trmatlenths[i]*trmatlenths[i];
	}

	vec<hmmstate> a;
	a.setLength(trmatlenths[0]);
	for (int i=0;i<a.len;i++) {
		a[i].active=activestates[i];
		a[i].identifyers.setLength(1);
		a[i].identifyers[0]=i;
	}
	vec<hmmstate> result(a);
	offset=a.len;
	for (int i=1;i<nummatrices;i++) {
		vec<hmmstate> b(trmatlenths[i]);
		for (int j=0;j<b.len;j++) {
			b[j].active=activestates[j+offset];
			b[j].identifyers.setLength(1);
			b[j].identifyers[0]=j;
		}
		offset+=b.len;
		tensor_addition(a, b, result, overlaps);
		a=result;
	}
/*
	for (int i=0;i<a.len;i++){
		for (int j=0;j<a[i].identifyers.len;j++){
			cout << a[i].identifyers[j] << ((j!=(a[i].identifyers.len-1))?",":"");
		}
		cout << "\t" << a[i].active;
		cout << "\n";
	}
*/
///////// End of determination of important states
	vec<hmmtrstate> **colons = new vec<hmmtrstate>*[nummatrices];
	offset=0;
	for (int i=0; i<nummatrices; i++) {
		colons[i]=new vec<hmmtrstate>[trmatlenths[i]];
		for (int j=0;j<trmatlenths[i];j++){
			int count=0;
			for (int k=0;k<trmatlenths[i];k++){
				if (trmats[i][k+j*trmatlenths[i]] > 0.0) 
					count++;
			}
			colons[i][j].setLength(count);
			count=0;
			for (int k=0;k<trmatlenths[i];k++){
				if (trmats[i][k+j*trmatlenths[i]] > 0.0) {
					colons[i][j][count].p=trmats[i][k+j*trmatlenths[i]];
					colons[i][j][count].active=activestates[offset+k];
					colons[i][j][count].identifyers.setLength(1);
					colons[i][j][count].identifyers[0]=k;
					count++;
				}
			}
		}
		offset += trmatlenths[i]; 
	}

/////////
	vec<hmmtrstate> *allstates = new vec<hmmtrstate>[result.len];
	int trivialcount=0;
	int nontrivialcount=0;
	for (int i=0; i<result.len; i++) {
		vec<hmmtrstate> tmp=colons[0][result[i].identifyers[0]];
		vec<hmmtrstate> finalstates(tmp);
		for (int j=1;j<nummatrices;j++){
			tensor_merge(tmp,colons[j][result[i].identifyers[j]],finalstates,overlaps);
			tmp=finalstates;
		}
		allstates[i]=finalstates;
		if (finalstates.len > 1) {
			nontrivialcount++;
		} else {
			trivialcount++;
		}
	}
/*	
	cout << "trivialcount: " << trivialcount << "\n";
	cout << "nontrivialcount: " << nontrivialcount << "\n";

	for (int i=0; i<result.len; i++) {
		cout << "(";
		for (int j=0; j<result[i].identifyers.len; j++) {
			cout << result[i].identifyers[j];
			(j==(result[i].identifyers.len-1))?cout << ")":cout << ",";
		}
		cout << " my come from ";
		for (int r=0; r<allstates[i].len; r++) {
			cout << "(";
			for (int j=0; j<allstates[i][r].identifyers.len; j++) {
				cout << allstates[i][r].identifyers[j];
				(j==(allstates[i][r].identifyers.len-1))?cout << ")":cout << ",";
			}
		}
		cout << "\n";
	}
*/
	vec<trivialstate> trivials(trivialcount);
	vec<nontrivialstate> nontrivials(nontrivialcount);
	int trivialc=0;
	int nontrivialc=0;
	for (int i=0;i<result.len;i++) {
		if (allstates[i].len > 1) {
			nontrivials[nontrivialc].stateident = result[i].identifyers;
			nontrivials[nontrivialc].mu.setLength(dim);
			for (int k=0;k<dim;k++) {
				nontrivials[nontrivialc].mu[k]=0;
				for (int j=0;j<nontrivials[nontrivialc].stateident.len;j++) {
					nontrivials[nontrivialc].mu[k] += mu_cell[j][nontrivials[nontrivialc].stateident[j]*dim+k];
				}
			}
			nontrivialc++;
		} else {
			trivials[trivialc].stateident = result[i].identifyers;
			trivials[trivialc].mu.setLength(dim);
			for (int k=0;k<dim;k++) {
				trivials[trivialc].mu[k]=0;
				for (int j=0;j<trivials[trivialc].stateident.len;j++) {
					trivials[trivialc].mu[k] += mu_cell[j][trivials[trivialc].stateident[j]*dim+k];
				}
			}
			trivialc++;
		}
	}

	hashtable<trivialstate> htt(trivials,true);
	hashtable<nontrivialstate> htnt(nontrivials,true);

	trivialc=0;
	nontrivialc=0;
	if (custom) {
		hashtable<hmm_tr_mat_entry> ctt(*custom_transitions,false);
		for (int i=0;i<result.len;i++) {
			if (allstates[i].len > 1) {
/*				cout << i << " " << nontrivialc << " : ";
				for (int j=0;j<nontrivials[nontrivialc].stateident.len;j++) {
					cout << nontrivials[nontrivialc].stateident[j] << " ";
				}
				cout << endl;*/
				nontrivials[nontrivialc].priors.setLength(allstates[i].len);
				for (int j=0;j<allstates[i].len;j++) {
					nontrivials[nontrivialc].priors[j] = lookupstatenode(allstates[i][j].identifyers,htt,htnt);
					hmm_tr_mat_entry ct;
					ct.instate=allstates[i][j].identifyers;
					ct.outstate=result[i].identifyers;
					int idx=ctt.lookup(ct);
					if (idx >= 0) {
						nontrivials[nontrivialc].priors[j].logp = custom_transitions->operator[](idx).p;
					} else {
						nontrivials[nontrivialc].priors[j].logp = logd(allstates[i][j].p);
					}
				}
				nontrivialc++;
			} else {
/*				cout << i << " " << trivialc << " ";
				for (int j=0;j<trivials[trivialc].stateident.len;j++) {
					cout << trivials[trivialc].stateident[j] << " ";
				}
				cout << endl;*/
				trivials[trivialc].prior = lookupstatenode(allstates[i][0].identifyers,htt,htnt);
				hmm_tr_mat_entry ct;
				ct.instate=allstates[i][0].identifyers;
				ct.outstate=result[i].identifyers;
				int idx=ctt.lookup(ct);
				if (idx >= 0) {
					trivials[trivialc].prior.logp = custom_transitions->operator[](idx).p;
				} else {
					trivials[trivialc].prior.logp = 0.0;
				}

				trivialc++;
			}
		}
	} else {
		for (int i=0;i<result.len;i++) {
			if (allstates[i].len > 1) {
/*				cout << i << " " << nontrivialc << " : ";
				for (int j=0;j<nontrivials[nontrivialc].stateident.len;j++) {
					cout << nontrivials[nontrivialc].stateident[j] << " ";
				}
				cout << endl;*/
				nontrivials[nontrivialc].priors.setLength(allstates[i].len);
				for (int j=0;j<allstates[i].len;j++) {
					nontrivials[nontrivialc].priors[j] = lookupstatenode(allstates[i][j].identifyers,htt,htnt);
					nontrivials[nontrivialc].priors[j].logp = logd(allstates[i][j].p);
				}
				nontrivialc++;
			} else {
/*				cout << i << " " << trivialc << " ";
				for (int j=0;j<trivials[trivialc].stateident.len;j++) {
					cout << trivials[trivialc].stateident[j] << " ";
				}
				cout << endl;*/
				trivials[trivialc].prior = lookupstatenode(allstates[i][0].identifyers,htt,htnt);
				trivials[trivialc].prior.logp = 0.0;
				trivialc++;
			}
		}
	}
/*	for (int i=0;i<trivialcount;i++){
		cout << i << " " << i+nontrivialcount << " : ";
		for (int j=0;j<trivials[i].stateident.len;j++) {
			cout << trivials[i].stateident[j] << " ";
		}
		cout << endl;
	}
	for (int i=0;i<nontrivialcount;i++){
		cout << i << " : ";
		for (int j=0;j<nontrivials[i].stateident.len;j++) {
			cout << nontrivials[i].stateident[j] << " ";
		}
		cout << endl;
	}*/

	delete [] allstates;
	for (int i=0; i<nummatrices; i++) {
		delete [] colons[i];
	}
	delete []colons;



	int totalstatescount=nontrivialcount+trivialcount;
	vec<double> curprob(totalstatescount);
	vec<double> lastprob(totalstatescount);
	lastprob.data[0]=1.0;
	for (int i=1;i<totalstatescount;i++){
		lastprob.data[i]=0.0;
	}
	logVec(lastprob);

	vec<unsigned int> seq;
	spike_history_new history(nontrivials,trivials,observed_data.len/dim,overlaps,seq);

	vec<double> tmp(dim);
	double exponent=0.0;
	double *x;
	double current_probability=0.0;
	vec<unsigned int> tracker;
	unsigned int idx, state_idx;
	state_identifyer max_probability_index;
	double max_probability;

	for (int t=1;t<observed_data.len/dim;t++) {
		x=&observed_data[t*dim];
		tracker.len=nontrivialcount;
		tracker.data=history.getCurrentColumn();

		for (int state_idx=0;state_idx<nontrivialcount;state_idx++){
			for (int i=0; i < dim; i++) {
				tmp.data[i] = x[i]-nontrivials[state_idx].mu[i];
			}
			exponent=a_transpose_cinv_a(tmp,cinv);

			max_probability=-1e308*10;
			max_probability_index=nontrivials[state_idx].priors[0].stateidx;
			for (int pre=0;pre<nontrivials[state_idx].priors.len;pre++) {
				idx = nontrivials[state_idx].priors[pre].stateidx+(nontrivials[state_idx].priors[pre].trivial?nontrivialcount:0);
				current_probability = lastprob.data[idx]+nontrivials[state_idx].priors[pre].logp;
				if (current_probability>max_probability) {
					  max_probability_index = idx;
					  max_probability = current_probability;
				}
			}
			curprob.data[state_idx] = max_probability+exponent;
			tracker.data[state_idx] = max_probability_index;
		}
		for (int state_idx=0;state_idx<trivialcount;state_idx++){
			for (int i=0; i < dim; i++) {
				tmp.data[i] = x[i]-trivials[state_idx].mu[i];
			}
			exponent=a_transpose_cinv_a(tmp,cinv);

			idx = trivials[state_idx].prior.stateidx + (trivials[state_idx].prior.trivial?nontrivialcount:0);
			curprob.data[state_idx+nontrivialcount] = lastprob.data[idx]+exponent;
		}
		history.step_time();
#ifdef VERBOSE
		if (t%100==0) cout << 100.0*(double)t/((double)observed_data.len/dim) << "% done\n";
#endif

#ifndef WIN32
		std::swap(lastprob.data,curprob.data);
#else
		swap(lastprob.data,curprob.data);
#endif
	}
	tracker.data=NULL;
	tracker.len=0;
	max_probability=-1e308*10;
	current_probability=0.0;
	max_probability_index;
	for (unsigned int i=0;i<totalstatescount;i++) {
		current_probability = lastprob.data[i];
		if (current_probability>max_probability) {
			  max_probability_index = i;
			  max_probability = current_probability;
		}
		
	}
    max_probability_index = 0;
	history.finalize(max_probability_index);
	mlseq.setLength(seq.len);
	for (int i=0;i<seq.len;i++) {
		mlseq[i] = (seq[i]<nontrivials.len)?nontrivials[seq[i]].stateident:trivials[seq[i]-nontrivials.len].stateident;
	}
   	return lastprob.data[max_probability_index];
}

double viterbi_nd(vec<double> &observed_data,
		vec<double> &p,
		double *cinv,
		vec<unsigned int> &numstates,
		vec<double> *mu_cell,
		vec<double> &initial_guess,
		vec<state_type> &mlseq,
		const int overlaps,
		const int dim) {
	// calculate constants

	assert(observed_data.len % dim == 0);

	statespace_enum st(numstates,overlaps);

	vec<state_type> nontrivialstates;
	vec<state_type> trivialstates;
	st.getStates(trivialstates,nontrivialstates);

	hashtable<state_type> lookup_nontrivial(nontrivialstates);
	hashtable<state_type> lookup_trivial(trivialstates);

	int nontrivialcount=nontrivialstates.len;
	int trivialcount=trivialstates.len;
	int totalstatescount=nontrivialcount+trivialcount;

	#ifdef VERBOSE
	cout << "Using " << (nontrivialcount*sizeof(state_identifyer))/1024 << " kb/timestep for " << nontrivialcount << " States" << endl;
	#endif

	vec<state_identifyer> *lookuptable=new vec<state_identifyer>[nontrivialcount];
	vec<double> *lookuptable_trprob=new vec<double>[nontrivialcount];


	for (unsigned int state_idx=0;state_idx<nontrivialcount;state_idx++){
		state_type state=nontrivialstates[state_idx];
		vec<state_type> *tmptable = generatelookuptable(state,numstates,overlaps);
		vec<state_identifyer> *curlookuptable = new vec<state_identifyer>(tmptable->len);
		lookuptable_trprob[state_idx].setLength(curlookuptable->len);
		for (int j=0;j<tmptable->len;j++){
			curlookuptable->operator [ ](j) = calc_prob_pos(tmptable->data[j],lookup_nontrivial,lookup_trivial);
			double ltp=0.0;
			vec<unsigned int> ss1=vec<unsigned int>(numstates.len);
			vec<unsigned int> ss2=vec<unsigned int>(numstates.len);
			splitState(state,numstates,ss1);
			splitState(tmptable->data[j],numstates,ss2);
			ltp=logd(calctransmissionprob(ss2,ss1,p,numstates));
			lookuptable_trprob[state_idx][j] = ltp;
		}
		lookuptable[state_idx] = *curlookuptable;
		delete tmptable;
	}


	vec<state_identifyer> lookuptable_idx_trivial(trivialcount);
	for (unsigned int state_idx=0;state_idx<trivialcount;state_idx++){
		state_type state=trivialstates[state_idx];
		lookuptable_idx_trivial[state_idx] = calc_prob_pos(rewindtrivialstate(state,numstates),lookup_nontrivial,lookup_trivial);
	}

	vec<double> mu_nontrivial(nontrivialcount*dim);
	for (int i=0;i<nontrivialcount;i++) {
		vec<unsigned int> sstate;
		splitState(nontrivialstates[i],numstates,sstate);
		for (int curdim=0; curdim < dim; curdim++) {
			mu_nontrivial[i*dim+curdim]=0.0;
			for (int j=0;j<sstate.len;j++) {
//				std::cout << j << " " << curdim << " " << i << "\n";
				mu_nontrivial[i*dim+curdim] += mu_cell[j][sstate[j]*dim+curdim];
			}
		}
	}

	vec<double> mu_trivial(trivialcount*dim);
	for (int i=0;i<trivialcount;i++) {
		vec<unsigned int> sstate;
		splitState(trivialstates[i],numstates,sstate);
		for (int curdim=0; curdim < dim; curdim++) {
			mu_trivial[i*dim+curdim] = 0.0;
			for (int j=0;j<sstate.len;j++) {
				mu_trivial[i*dim+curdim] += mu_cell[j][sstate[j]*dim+curdim];
			}
		}
	}

	spike_history history(numstates,nontrivialstates,trivialstates,observed_data.len/dim,overlaps,lookuptable_idx_trivial,mlseq);

	vec<double> curprob(totalstatescount);
	vec<double> lastprob(totalstatescount);
	if (initial_guess.len > 0) {
		for (int i=0;i<totalstatescount;i++){
			lastprob.data[calc_prob_pos(i,lookup_nontrivial,lookup_trivial)]=initial_guess[i];
		}
	} else {
		lastprob.data[calc_prob_pos(0,lookup_nontrivial,lookup_trivial)]=1.0;
		for (int i=1;i<totalstatescount;i++){
			lastprob.data[i]=0.0;
		}
	}

	logVec(lastprob);

	vec<double> tmp(dim);
	double exponent=0.0;
	double *x;
	double current_probability=0.0;
	vec<state_identifyer> tracker;
	unsigned int idx, state_idx;
	state_identifyer max_probability_index;
	int i,pre;
	double max_probability;

	for (int t=1;t<observed_data.len/dim;t++) {
		x=&observed_data[t*dim];
		tracker.len=nontrivialcount;
		tracker.data=history.getCurrentColumn();

		for (state_idx=0;state_idx<nontrivialcount;state_idx++){
			for (i=0; i < dim; i++) {
				tmp.data[i] = x[i]-mu_nontrivial.data[state_idx*dim+i];
			}
			exponent=a_transpose_cinv_a(tmp,cinv);

			max_probability=-1e308*10;
			max_probability_index=lookuptable[state_idx][0];
			for (pre=0;pre<lookuptable[state_idx].len;pre++) {
				idx = lookuptable[state_idx][pre];
				current_probability = lastprob.data[idx]+lookuptable_trprob[state_idx][pre];
				if (current_probability>max_probability) {
					  max_probability_index = lookuptable[state_idx][pre];
					  max_probability = current_probability;
				}
			}
			curprob.data[state_idx] = max_probability+/*k1+*/exponent;
			tracker.data[state_idx] = max_probability_index;
		}
		for (state_idx=0;state_idx<trivialcount;state_idx++){
			for (i=0; i < dim; i++) {
				tmp.data[i] = x[i]-mu_trivial.data[state_idx*dim+i];
			}
			exponent=a_transpose_cinv_a(tmp,cinv);

			idx = lookuptable_idx_trivial[state_idx];
			curprob.data[state_idx+nontrivialcount] = lastprob.data[idx]+/*k1+*/exponent;
		}
		history.step_time();
#ifdef VERBOSE
		cout << 100.0*(double)t/((double)observed_data.len/4.0) << "% done\n";
#endif

#ifndef WIN32
		std::swap(lastprob.data,curprob.data);
#else
		swap(lastprob.data,curprob.data);
#endif
	}
	tracker.data=NULL;
	tracker.len=0;


    max_probability=-1e308*10;
	current_probability=0.0;
	max_probability_index;
	for (unsigned int i=0;i<totalstatescount;i++) {
		current_probability = lastprob.data[i];
		if (current_probability>max_probability) {
			  max_probability_index = i;
			  max_probability = current_probability;
		}
		
	}

	history.finalize(max_probability_index);

	delete[] lookuptable;
	delete[] lookuptable_trprob;
	return lastprob.data[max_probability_index];
/*	history.finalize(0);

	delete[] lookuptable;
	delete[] lookuptable_trprob;
	return lastprob.data[0];*/
}

#ifndef WIN32


double viterbi_nd_record(vec<double> &observed_data,
		vec<double> &p,
		double *cinv,
		vec<unsigned int> &numstates,
		vec<double> *mu_cell,
		vec<double> &initial_guess,
		vec<state_type> &mlseq,
		const int overlaps,
		const int dim,
		const char * filename) {
	// calculate constants
	FILE *f_backtracking_matrix,*f_sec_max_idx,*f_sec_max_val,*f_max_val,*f_config;
	string fname(filename);
	f_backtracking_matrix=fopen((fname+"_backtracking.data").c_str(),"w");
	if (f_backtracking_matrix == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return 0.0;
	}
	f_max_val=fopen((fname+"_max_val.data").c_str(),"w");
	if (f_max_val == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return 0.0;
	}
	f_sec_max_idx=fopen((fname+"_sec_max_idx.data").c_str(),"w");
	if (f_sec_max_idx == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return 0.0;
	}
	f_sec_max_val=fopen((fname+"_sec_max_val.data").c_str(),"w");
	if (f_sec_max_val == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return 0.0;
	}
	f_config=fopen((fname+"_config.data").c_str(),"w");
	if (f_config == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return 0.0;
	}

	assert(observed_data.len % dim == 0);

	statespace_enum st(numstates,overlaps);

	vec<state_type> nontrivialstates;
	vec<state_type> trivialstates;
	st.getStates(trivialstates,nontrivialstates);

	hashtable<state_type> lookup_nontrivial(nontrivialstates);
	hashtable<state_type> lookup_trivial(trivialstates);

	int nontrivialcount=nontrivialstates.len;
	int trivialcount=trivialstates.len;
	int totalstatescount=nontrivialcount+trivialcount;

	#ifdef VERBOSE
	cout << "Using " << (nontrivialcount*sizeof(state_identifyer))/1024 << " kb/timestep for " << nontrivialcount << " States" << endl;
	#endif

	vec<state_identifyer> *lookuptable=new vec<state_identifyer>[nontrivialcount];
	vec<double> *lookuptable_trprob=new vec<double>[nontrivialcount];


	for (unsigned int state_idx=0;state_idx<nontrivialcount;state_idx++){
		state_type state=nontrivialstates[state_idx];
		vec<state_type> *tmptable = generatelookuptable(state,numstates,overlaps);
		vec<state_identifyer> *curlookuptable = new vec<state_identifyer>(tmptable->len);
		lookuptable_trprob[state_idx].setLength(curlookuptable->len);
		for (int j=0;j<tmptable->len;j++){
			curlookuptable->operator [ ](j) = calc_prob_pos(tmptable->data[j],lookup_nontrivial,lookup_trivial);
			double ltp=0.0;
			vec<unsigned int> ss1=vec<unsigned int>(numstates.len);
			vec<unsigned int> ss2=vec<unsigned int>(numstates.len);
			splitState(state,numstates,ss1);
			splitState(tmptable->data[j],numstates,ss2);
			ltp=logd(calctransmissionprob(ss2,ss1,p,numstates));
			lookuptable_trprob[state_idx][j] = ltp;
		}
		lookuptable[state_idx] = *curlookuptable;
		delete tmptable;
	}


	vec<state_identifyer> lookuptable_idx_trivial(trivialcount);
	for (unsigned int state_idx=0;state_idx<trivialcount;state_idx++){
		state_type state=trivialstates[state_idx];
		lookuptable_idx_trivial[state_idx] = calc_prob_pos(rewindtrivialstate(state,numstates),lookup_nontrivial,lookup_trivial);
	}

	vec<double> mu_nontrivial(nontrivialcount*dim);
	for (int i=0;i<nontrivialcount;i++) {
		vec<unsigned int> sstate;
		splitState(nontrivialstates[i],numstates,sstate);
		for (int curdim=0; curdim < dim; curdim++) {
			mu_nontrivial[i*dim+curdim]=0.0;
			for (int j=0;j<sstate.len;j++) {
//				std::cout << j << " " << curdim << " " << i << "\n";
				mu_nontrivial[i*dim+curdim] += mu_cell[j][sstate[j]*dim+curdim];
			}
		}
	}

	vec<double> mu_trivial(trivialcount*dim);
	for (int i=0;i<trivialcount;i++) {
		vec<unsigned int> sstate;
		splitState(trivialstates[i],numstates,sstate);
		for (int curdim=0; curdim < dim; curdim++) {
			mu_trivial[i*dim+curdim] = 0.0;
			for (int j=0;j<sstate.len;j++) {
				mu_trivial[i*dim+curdim] += mu_cell[j][sstate[j]*dim+curdim];
			}
		}
	}

	spike_history history(numstates,nontrivialstates,trivialstates,observed_data.len/dim,overlaps,lookuptable_idx_trivial,mlseq);

	vec<double> curprob(totalstatescount);
	vec<double> lastprob(totalstatescount);
	if (initial_guess.len > 0) {
		for (int i=0;i<totalstatescount;i++){
			lastprob.data[calc_prob_pos(i,lookup_nontrivial,lookup_trivial)]=initial_guess[i];
		}
	} else {
		lastprob.data[calc_prob_pos(0,lookup_nontrivial,lookup_trivial)]=1.0;
		for (int i=1;i<totalstatescount;i++){
			lastprob.data[i]=0.0;
		}
	}

	logVec(lastprob);

	vec<double> tmp(dim);
	double exponent=0.0;
	double *x;
	double current_probability=0.0;
	vec<state_identifyer> tracker;
	unsigned int idx, state_idx;
	state_identifyer max_probability_index;
	int i,pre;
	double max_probability;
	vec<double> maxprob(nontrivialcount),secmaxprob(nontrivialcount);
	vec<state_identifyer> sec_max_idx(nontrivialcount);
	int sWritten;

	for (int t=1;t<observed_data.len/dim;t++) {
		x=&observed_data[t*dim];
		tracker.len=nontrivialcount;
		tracker.data=history.getCurrentColumn();

		for (state_idx=0;state_idx<nontrivialcount;state_idx++){
			for (i=0; i < dim; i++) {
				tmp.data[i] = x[i]-mu_nontrivial.data[state_idx*dim+i];
			}
			exponent=a_transpose_cinv_a(tmp,cinv);

			max_probability=-1e308*10;
			secmaxprob[state_idx]=-1e308*10;
			max_probability_index=lookuptable[state_idx][0];
			sec_max_idx[state_idx]=lookuptable[state_idx][0];
			for (pre=0;pre<lookuptable[state_idx].len;pre++) {
				idx = lookuptable[state_idx][pre];
				current_probability = lastprob.data[idx]+lookuptable_trprob[state_idx][pre];
				if (current_probability>max_probability) {
					  secmaxprob[state_idx] = max_probability;
					  sec_max_idx[state_idx] = max_probability_index;
					  max_probability_index = lookuptable[state_idx][pre];
					  max_probability = current_probability;
				} else if (current_probability>secmaxprob[state_idx]) {
					  secmaxprob[state_idx] = current_probability;
					  sec_max_idx[state_idx] = lookuptable[state_idx][pre];
				}
			}
			curprob.data[state_idx] = max_probability+/*k1+*/exponent;
			tracker.data[state_idx] = max_probability_index;

			maxprob.data[state_idx] = max_probability;
		}

		sWritten=fwrite(maxprob.data,sizeof(double),maxprob.len,f_max_val);
		assert(sWritten==maxprob.len);
		sWritten=fwrite(tracker.data,sizeof(state_identifyer),tracker.len,f_backtracking_matrix);
		assert(sWritten==tracker.len);
		sWritten=fwrite(secmaxprob.data,sizeof(double),secmaxprob.len,f_sec_max_val);
		assert(sWritten==secmaxprob.len);
		sWritten=fwrite(sec_max_idx.data,sizeof(state_identifyer),sec_max_idx.len,f_sec_max_idx);
		assert(sWritten==sec_max_idx.len);

		for (state_idx=0;state_idx<trivialcount;state_idx++){
			for (i=0; i < dim; i++) {
				tmp.data[i] = x[i]-mu_trivial.data[state_idx*dim+i];
			}
			exponent=a_transpose_cinv_a(tmp,cinv);

			idx = lookuptable_idx_trivial[state_idx];
			curprob.data[state_idx+nontrivialcount] = lastprob.data[idx]+/*k1+*/exponent;
		}

		history.step_time();
#ifdef VERBOSE
		cout << 100.0*(double)t/((double)observed_data.len/4.0) << "% done\n";
#endif

#ifndef WIN32
		std::swap(lastprob.data,curprob.data);
#else
		swap(lastprob.data,curprob.data);
#endif
	}
	tracker.data=NULL;
	tracker.len=0;

	max_probability=-1e308*10;
	current_probability=0.0;
	max_probability_index;
	for (unsigned int i=0;i<totalstatescount;i++) {
		current_probability = lastprob.data[i];
		if (current_probability>max_probability) {
			  max_probability_index = i;
			  max_probability = current_probability;
		}
		
	}

	history.finalize(max_probability_index);

	delete[] lookuptable;
	delete[] lookuptable_trprob;

	fwrite(&max_probability_index,sizeof(max_probability_index),1,f_config);
	fwrite(&overlaps,sizeof(overlaps),1,f_config);
	int datalen=observed_data.len/dim;
	fwrite(&datalen,sizeof(datalen),1,f_config);
	fwrite(&numstates.len,sizeof(numstates.len),1,f_config);
	fwrite(&numstates.data[0],sizeof(numstates.data[0]),numstates.len,f_config);

	fclose(f_config);
	fclose(f_sec_max_val);
	fclose(f_sec_max_idx);
	fclose(f_max_val);
	fclose(f_backtracking_matrix);
	return lastprob.data[max_probability_index];
}

void decode_from_file(vec<state_type> &mlseq,
		vec<double> &descisions,
		const char * filename) {
	FILE *f_backtracking_matrix,*f_sec_max_idx,*f_sec_max_val,*f_max_val,*f_config;
	string fname(filename);
	f_backtracking_matrix=fopen((fname+"_backtracking.data").c_str(),"r");
	if (f_backtracking_matrix == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return;
	}
	f_max_val=fopen((fname+"_max_val.data").c_str(),"r");
	if (f_max_val == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return;
	}
	f_sec_max_idx=fopen((fname+"_sec_max_idx.data").c_str(),"r");
	if (f_sec_max_idx == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return;
	}
	f_sec_max_val=fopen((fname+"_sec_max_val.data").c_str(),"r");
	if (f_sec_max_val == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return;
	}

	f_config=fopen((fname+"_config.data").c_str(),"r");
	if (f_config == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return;
	}

	unsigned int datalength;
	vec<unsigned int> numstates;
	int overlaps;
	state_identifyer maxstate;

	fread(&maxstate,sizeof(maxstate),1,f_config);
	fread(&overlaps,sizeof(overlaps),1,f_config);
	fread(&datalength,sizeof(datalength),1,f_config);
	int tmp;
	fread(&tmp,sizeof(tmp),1,f_config);
	numstates.setLength(tmp);
	fread(&numstates.data[0],sizeof(numstates.data[0]),numstates.len,f_config);

	statespace_enum st(numstates,overlaps);
	vec<state_type> nontrivial;
	vec<state_type> trivial;
	st.getStates(trivial,nontrivial);
	int nontrivialcount=nontrivial.len;
	int trivialcount=trivial.len;
	int totalstatescount=nontrivialcount+trivialcount;
	hashtable<state_type> lookup_nontrivial(nontrivial);
	hashtable<state_type> lookup_trivial(trivial);
	vec<state_identifyer> v(datalength);
	vec<state_identifyer> rewound_states(trivial.len);

	for (unsigned int state_idx=0;state_idx<trivial.len;state_idx++){
		state_type state=trivial[state_idx];
		rewound_states[state_idx] = calc_prob_pos(rewindtrivialstate(state,numstates),lookup_nontrivial,lookup_trivial);
	}
	
	mlseq.setLength(datalength);
	v.setLength(datalength);
	descisions.setLength(datalength);

	descisions[datalength-1]=-1e308*10;
	v[datalength-1] = calc_actual_state(maxstate,nontrivial,trivial);
	state_identifyer laststate=maxstate;
	double nom,den;
	int sRead,err;
	for (int t=datalength-2;t>=0;t--){
		if (laststate < nontrivial.len) {  //check if it is a trivial one
//			fpos_t pos=(fpos_t)(t*nontrivial.len+laststate)*(fpos_t)sizeof(state_identifyer);
			err=fseeko(f_backtracking_matrix,(t*nontrivial.len+laststate)*sizeof(state_identifyer),SEEK_SET);
			if (err != 0)
				fprintf(stderr,"%s\n",strerror(errno));

			sRead=fread(&v.data[t],sizeof(state_identifyer),1,f_backtracking_matrix);
			if (sRead != 1)
				if (ferror(f_backtracking_matrix))
					fprintf(stderr,"\n%s\n",strerror(errno));
			assert(sRead==1);
			err=fseeko(f_max_val,(t*nontrivialcount+laststate)*sizeof(double),SEEK_SET);
			if (err != 0)
				fprintf(stderr,"%s\n",strerror(errno));
			sRead=fread(&nom,sizeof(double),1,f_max_val);
			if (sRead != 1) {
				if (ferror(f_max_val))
					fprintf(stderr,"\n%s\n",strerror(errno));
				if (feof(f_max_val))
					fprintf(stderr,"end of file\n");
			}
			assert(sRead==1);
			err=fseeko(f_sec_max_val,(t*nontrivialcount+laststate)*sizeof(double),SEEK_SET);
			if (err != 0)
				fprintf(stderr,"%s\n",strerror(errno));
			sRead=fread(&den,sizeof(double),1,f_sec_max_val);
			if (sRead != 1) {
				if (ferror(f_sec_max_val))
					fprintf(stderr,"\n%s\n",strerror(errno));
				if (feof(f_sec_max_val))
					fprintf(stderr,"end of file\n");
			}
			assert(sRead==1);
			descisions[t] = (nom-den);

		} else {
			v[t]=rewound_states[laststate-nontrivial.len];
			descisions[t] = -1e308*10;
		}
		laststate=v[t];
	}

	for (int i=0; i < v.len; i++) {
		mlseq[i] = calc_actual_state(v[i],nontrivial,trivial);
	}
		
	fclose(f_config);
	fclose(f_sec_max_val);
	fclose(f_sec_max_idx);
	fclose(f_max_val);
	fclose(f_backtracking_matrix);
}

void decode_from_file_with_error(vec<state_type> &mlseq,
		vec<bool> &errors,
		const char * filename) {
	FILE *f_backtracking_matrix,*f_sec_max_idx,*f_sec_max_val,*f_max_val,*f_config;
	string fname(filename);
	f_backtracking_matrix=fopen((fname+"_backtracking.data").c_str(),"r");
	if (f_backtracking_matrix == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"open error");
		#endif
		return;
	}
	f_sec_max_idx=fopen((fname+"_sec_max_idx.data").c_str(),"r");
	if (f_sec_max_idx == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"open error");
		#endif
		return;
	}
	f_config=fopen((fname+"_config.data").c_str(),"r");
	if (f_config == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return;
	}

	unsigned int datalength;
	vec<unsigned int> numstates;
	int overlaps;
	state_identifyer maxstate;

	fread(&maxstate,sizeof(maxstate),1,f_config);
	fread(&overlaps,sizeof(overlaps),1,f_config);
	fread(&datalength,sizeof(datalength),1,f_config);
	int tmp;
	fread(&tmp,sizeof(tmp),1,f_config);
	numstates.setLength(tmp);
	fread(&numstates.data[0],sizeof(numstates.data[0]),numstates.len,f_config);

	statespace_enum st(numstates,overlaps);
	vec<state_type> nontrivial;
	vec<state_type> trivial;
	st.getStates(trivial,nontrivial);
	int nontrivialcount=nontrivial.len;
	int trivialcount=trivial.len;
	int totalstatescount=nontrivialcount+trivialcount;
	hashtable<state_type> lookup_nontrivial(nontrivial);
	hashtable<state_type> lookup_trivial(trivial);
	vec<state_identifyer> v(datalength);
	vec<state_identifyer> rewound_states(trivial.len);

	for (unsigned int state_idx=0;state_idx<trivial.len;state_idx++){
		state_type state=trivial[state_idx];
		rewound_states[state_idx] = calc_prob_pos(rewindtrivialstate(state,numstates),lookup_nontrivial,lookup_trivial);
	}
	
	mlseq.setLength(datalength);
	v.setLength(datalength);

	v[datalength-1] = calc_actual_state(maxstate,nontrivial,trivial);
	state_identifyer laststate=maxstate;
	double nom,den;
	int sRead,err;
	for (int t=datalength-2;t>=0;t--){
		if (laststate < nontrivial.len) {  //check if it is a trivial one
			if (!errors[t]) {
				err=fseeko(f_backtracking_matrix,(t*nontrivial.len+laststate)*sizeof(state_identifyer),SEEK_SET);
				if (err != 0)
					fprintf(stderr,"%s\n",strerror(errno));
				sRead=fread(&v.data[t],sizeof(state_identifyer),1,f_backtracking_matrix);
				if (sRead != 1)
					if (ferror(f_backtracking_matrix))
						fprintf(stderr,"\n%s\n",strerror(errno));
				assert(sRead==1);
			} else {
				err=fseeko(f_sec_max_idx,(t*nontrivial.len+laststate)*sizeof(state_identifyer),SEEK_SET);
				if (err != 0)
					fprintf(stderr,"%s\n",strerror(errno));
				sRead=fread(&v.data[t],sizeof(state_identifyer),1,f_sec_max_idx);
				if (sRead != 1)
					if (ferror(f_sec_max_idx))
						fprintf(stderr,"\n%s\n",strerror(errno));
				assert(sRead==1);
			}

		} else {
			v[t]=rewound_states[laststate-nontrivial.len];
		}
		laststate=v[t];
	}

	for (int i=0; i < v.len; i++) {
		mlseq[i] = calc_actual_state(v[i],nontrivial,trivial);
	}
	
	fclose(f_config);
	fclose(f_sec_max_idx);
	fclose(f_backtracking_matrix);
}
#endif
#endif
