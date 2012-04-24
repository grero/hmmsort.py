#ifndef _VITERBIHELPERS_HPP_INCLUDED
#define _VITERBIHELPERS_HPP_INCLUDED

using namespace std;


//typedef unsigned long long int state_type;
#ifndef WIN32
	typedef long long int __int64;
	typedef unsigned long long int __uint64;
#endif

	typedef __uint64 state_type;
	typedef unsigned int state_identifyer;



template <class T>
class simple_stack : private vec<T> {
	private:
		unsigned int cur_pos;
		unsigned int alloc_incr;
	public:
		simple_stack(unsigned int allocation_len=10);
		void setAllocLength(unsigned int allocation_len);
		void push(T element);
};

class spike_history {
	private:
		unsigned int current_time;
		unsigned int overlaps;
		vec<state_type> &mlseq;
		vec<state_identifyer> tmppath;
		vec<unsigned int> &numstates;
		vec<state_type> &nontrivial;
		vec<state_type> &trivial;
		vec<state_identifyer> &rewound_states;

		hashtable<state_type> lookup_nontrivial;

		vec<state_type> rewound_trivial;
#ifndef WIN32
		vec<unsigned short> rewind_trivial_amount;
#else
		vec<__int16> rewind_trivial_amount;
#endif
		vec<state_identifyer> history;

		unsigned int allocation_granularity;

		unsigned int windowsize;
		unsigned int mem_available;

		unsigned int offset;
		unsigned int matrix_start;

		void pack();
		void recalculate_size();
		void partial_decode(vec<state_identifyer> &v, state_identifyer max_state, unsigned int maxsteps=0, unsigned int offset=0);
	public:
		void finalize(state_identifyer max_state);
		void step_time();
		state_identifyer * getCurrentColumn();
		spike_history(vec<unsigned int> &in_numstates,
							 vec<state_type> &nontrivial,
							 vec<state_type> &trivial,
							 unsigned int time,
							 unsigned int overlaps,
							 vec<state_identifyer> &rewound_states,
							 vec<state_type> &mlseq);
};

struct hmm_tr_mat_entry{
	vec<int> instate;
	vec<int> outstate;
	double p;
};

bool operator > (const hmm_tr_mat_entry &lhs, const hmm_tr_mat_entry &rhs) {
	int i=0;
	for (int i=0;i<lhs.instate.len;i++)
		if (lhs.instate[i] != rhs.instate[i])
			return (lhs.instate[i] > rhs.instate[i]);
	for (int i=0;i<lhs.outstate.len;i++)
		if (lhs.outstate[i] != rhs.outstate[i])
			return (lhs.outstate[i] > rhs.outstate[i]);
	return false;
}

bool operator < (const hmm_tr_mat_entry &lhs, const hmm_tr_mat_entry &rhs) {
	int i=0;
	for (int i=0;i<lhs.instate.len;i++)
		if (lhs.instate[i] != rhs.instate[i])
			return (lhs.instate[i] < rhs.instate[i]);
	for (int i=0;i<lhs.outstate.len;i++)
		if (lhs.outstate[i] != rhs.outstate[i])
			return (lhs.outstate[i] < rhs.outstate[i]);
	return false;
}

bool operator == (const hmm_tr_mat_entry &lhs, const hmm_tr_mat_entry &rhs) {
	int i=0;
	for (int i=0;i<lhs.instate.len;i++)
		if (lhs.instate[i] != rhs.instate[i])
			return false;
	for (int i=0;i<lhs.outstate.len;i++)
		if (lhs.outstate[i] != rhs.outstate[i])
			return false;
	return true;
}


struct hmmtrstate{
	int active;
	double p;
	vec<int> identifyers;
};


struct hmmstate{
	int active;
	vec<int> identifyers;
};

struct stategraphnode {
	double logp;
	bool trivial;
	int stateidx;
};

struct trivialstate{
	vec<double> mu;
	vec<int> stateident;
	stategraphnode prior;
};

struct nontrivialstate{
	vec<double> mu;
	vec<int> stateident;
	vec<stategraphnode> priors;
};

class spike_history_new {
	private:
		unsigned int current_time;
		unsigned int overlaps;
		vec<unsigned int> &mlseq;
		vec<nontrivialstate> &nontrivials;
		vec<trivialstate> &trivials;
		vec<unsigned int> rewound_states;
		vec<unsigned int> tmppath;

		/*vec<state_type> rewound_trivial;
#ifndef WIN32
		vec<unsigned short> rewind_trivial_amount;
#else
		vec<__int16> rewind_trivial_amount;
#endif*/
		unsigned int **history;

		unsigned int allocation_granularity;
		unsigned int offset;         //

		unsigned int windowsize;     //allocation_granularity*segmentcount
		
		unsigned int segmentcount;   //number of segments each containing allocation_granularity cloumns
		unsigned int mem_available;  //number of free columns
		unsigned int matrix_start;   //logical position in matrix, that represents time [offset]
		unsigned int getelem(unsigned int m, unsigned int n);

		void pack();
		void recalculate_size();
		void partial_decode(vec<unsigned int> &v, unsigned int max_state, unsigned int maxsteps=0);
	public:
		void finalize(unsigned int max_state);
		void step_time();
		unsigned int * getCurrentColumn();
		~spike_history_new();
		spike_history_new(vec<nontrivialstate> &nontrivials,
				  vec<trivialstate> &trivials,
				  unsigned int time,
				  unsigned int overlaps,
				  vec<unsigned int> &mlseq);
};

class statespace_enum {
	private:
		unsigned int max_recursion;

		vec<unsigned int> cur_split_state;

		int trivialcount;
		int nontrivialcount;

		unsigned int curfillpos_trivial;
		unsigned int curfillpos_nontrivial;
		vec<state_type> *trivial_ref;
		vec<state_type> *nontrivial_ref;

		void addtrivial(state_type state);
		void addnontrivial(state_type state);

		void count_if_k_are_active(unsigned int k);
		void fill_if_k_are_active(unsigned int k);
		void recursive_fill (int start, int stop, unsigned int recursion,int nonones);
		void recursive_count(int start, int stop, unsigned int recursion,int nonones);

		void count();
	public:
		statespace_enum(vec<unsigned int> &in_numstates, unsigned int in_overlaps)
			: numstates(in_numstates), overlaps(in_overlaps)  {};
		vec<unsigned int> &numstates;
		unsigned int overlaps;

		void getStates(vec<state_type> &trivial,vec<state_type> &nontrivial);
};


/******************************************************************************/

state_type combineStates(vec<unsigned int> &sstate, const vec<unsigned int> &numstates) {
	state_type result=0;
	state_type basis=1;
	for (int i=0;i<sstate.len;i++){
		result+=basis*(state_type) sstate[i];
		basis*=(state_type) numstates[i];
	}
	return result;
}

void splitState(state_type state,const vec<unsigned int> &numstates, vec<unsigned int> &sstate){
	sstate.setLength(numstates.len);
	for (int i=0;i<numstates.len;i++) {
		sstate[i] = state % (state_type) numstates[i];
		state=(state-(state_type)sstate[i])/(state_type)numstates[i];
	}
}

inline bool istrivial(state_type state, unsigned int overlaps , const vec<unsigned int> &numstates) {
	vec<unsigned int> sstate;
	splitState(state,numstates,sstate);
	unsigned int largerthanone=0;
	unsigned int zeros=0;
	for (int i=0; i < numstates.len; i++) {
		if (sstate[i]==0)
			zeros++;
		if (sstate[i]>1)
			largerthanone++;
	}
	return ((zeros==0) || (overlaps-largerthanone==0));
}

inline state_type rewindtrivialstate(state_type state, const vec<unsigned int> &numstates){
	vec<unsigned int> sstate;
	splitState(state, numstates, sstate);
	for (int i=0;i<numstates.len;i++) {
		if (sstate[i] > 0)
			sstate[i]--;
	}
	return combineStates(sstate,numstates);
}

inline state_identifyer calc_prob_pos(state_type state, hashtable<state_type> &lookup_nontrivial, hashtable<state_type> &lookup_trivial ){
	int idx = lookup_nontrivial.lookup(state);
	if (idx >= 0)
		return idx;
	else
		return lookup_trivial.lookup(state)+lookup_nontrivial.len;
}

inline state_type calc_actual_state(state_identifyer state, const vec<state_type> &nontrivial, const vec<state_type> &trivial) {
	if (state < nontrivial.len)
		return nontrivial[state];
	else 
		return trivial[state-nontrivial.len];
	
}

/******************************************************************************/


template <class T>
simple_stack<T>::simple_stack(unsigned int allocation_len) : alloc_incr(allocation_len){
	vec<T>::setLength(allocation_len);
}

template <class T>
void simple_stack<T>::push(T element){
	if (cur_pos == vec<T>::len) {
		vec<T>::setLength(vec<T>::len+alloc_incr);
	}
	vec<T>::data[cur_pos] = element;
	cur_pos++;
}

template <class T>
void simple_stack<T>::setAllocLength(unsigned int allocation_len){
	unsigned int newlen= ((cur_pos / allocation_len)+1)*allocation_len;
	vec<T>::setLength(newlen);
	alloc_incr=allocation_len;
}

/******************************************************************************/

void tensor_merge(vec<hmmtrstate> &a, vec<hmmtrstate> &b, vec<hmmtrstate> &result, int cutoff) {
	result.setLength(a.len*b.len);
	int count=0;
	for (int i=0;i<a.len;i++){
		for (int j=0;j<b.len;j++) {
			int curval=a[i].active+b[j].active;
			double curp=a[i].p*b[j].p;
			if (curval<cutoff) {
				result[count].active=curval;
				result[count].p=curp;
				result[count].identifyers.setLength(a[i].identifyers.len+b[j].identifyers.len);
				for (int q=0;q<a[i].identifyers.len;q++)
					result[count].identifyers[q]=a[i].identifyers[q];
				for (int q=0;q<b[j].identifyers.len;q++)
					result[count].identifyers[q+a[i].identifyers.len]=b[j].identifyers[q];
				count++;
			}
		}
	}
	result.setLength(count);
}

void tensor_addition(vec<hmmstate> &a, vec<hmmstate> &b, vec<hmmstate> &result, int cutoff) {
	result.setLength(a.len*b.len);
	int count=0;
	for (int i=0;i<a.len;i++){
		for (int j=0;j<b.len;j++) {
			int curval=a[i].active+b[j].active;
			if (curval<cutoff) {
				result[count].active=curval;
				result[count].identifyers.setLength(a[i].identifyers.len+b[j].identifyers.len);
				for (int q=0;q<a[i].identifyers.len;q++)
					result[count].identifyers[q]=a[i].identifyers[q];
				for (int q=0;q<b[j].identifyers.len;q++)
					result[count].identifyers[q+a[i].identifyers.len]=b[j].identifyers[q];
				count++;
			}
		}
	}
	result.setLength(count);
	
}



bool operator > (const vec<int> &lhs, const vec<int> &rhs) {
	int i=0;
	for (int i=0;i<lhs.len;i++){
		if (lhs[i] != rhs[i]) {
			return (lhs[i] > rhs[i]);
		}
	}
	return false;
}

bool operator < (const vec<int> &lhs, const vec<int> &rhs) {
	int i=0;
	for (int i=0;i<lhs.len;i++){
		if (lhs[i] != rhs[i]) {
			return (lhs[i] < rhs[i]);
		}
	}
	return false;
}

bool operator == (const vec<int> &lhs, const vec<int> &rhs) {
	int i=0;
	for (int i=0;i<lhs.len;i++){
		if (lhs[i] != rhs[i]) {
			return false;
		}
	}
	return true;
}

bool operator > (const trivialstate &lhs, const trivialstate &rhs) {
	return (lhs.stateident > rhs.stateident);
}

bool operator < (const trivialstate &lhs, const trivialstate &rhs) {
	return (lhs.stateident < rhs.stateident);
}

bool operator == (const trivialstate &lhs, const trivialstate &rhs) {
	return (lhs.stateident == rhs.stateident);
}

bool operator > (const nontrivialstate &lhs, const nontrivialstate &rhs) {
	return (lhs.stateident > rhs.stateident);
}

bool operator < (const nontrivialstate &lhs, const nontrivialstate &rhs) {
	return (lhs.stateident < rhs.stateident);
}

bool operator == (const nontrivialstate &lhs, const nontrivialstate &rhs) {
	return (lhs.stateident == rhs.stateident);
}

stategraphnode lookupstatenode(const vec<int> &ident, hashtable<trivialstate> &htt,hashtable<nontrivialstate> &htnt) {
	stategraphnode result;
	trivialstate tmp1;
	tmp1.stateident = ident;
	int idx=htt.lookup(tmp1);
	if (idx >= 0) {
		result.trivial = true;
		result.stateidx = idx;
	} else {
		nontrivialstate tmp2;
		tmp2.stateident = ident;
		result.trivial = false;
		result.stateidx = htnt.lookup(tmp2);
	}
	return result;
}

spike_history_new::spike_history_new(vec<nontrivialstate> &nontrivials,
				  vec<trivialstate> &trivials,
				  unsigned int time,
				  unsigned int overlaps,
				  vec<unsigned int> &mlseq)
	: nontrivials(nontrivials),
	  trivials(trivials),
	  current_time(0),
	  mlseq(mlseq),
	  overlaps(overlaps)
{
	tmppath.setLength(time);

	offset=0;
	int k=2;
	allocation_granularity = 40;
	windowsize = allocation_granularity*k;
	mem_available=windowsize;
	matrix_start=0;
	segmentcount = k;
	history = new unsigned int*[k];
	for (int i=0;i<k;i++)
		history[i] = new unsigned int[allocation_granularity*nontrivials.len];
	

//	rewound_trivial.setLength(trivial.len);
//	rewind_trivial_amount.setLength(trivial.len);
	rewound_states.setLength(trivials.len);
	for (int i=0; i < trivials.len; i++) {
//		rewind_trivial_amount[i] = 0;
		rewound_states[i]=trivials[i].prior.stateidx+(trivials[i].prior.trivial?nontrivials.len:0);
/*		while (istrivial(rewound_trivial[i],overlaps,numstates)) {
			rewound_trivial[i]=rewindtrivialstate(rewound_trivial[i],numstates);
			rewind_trivial_amount[i]++;
		}*/
	}
}

spike_history_new::~spike_history_new(){
	for (int i=0;i<segmentcount;i++)
		delete[] history[i];
	delete[] history;
}

unsigned int spike_history_new::getelem(unsigned int m, unsigned int n) {
	unsigned int actualcolumn=(m-offset+matrix_start) % (segmentcount*allocation_granularity);
	unsigned int segid=(actualcolumn / allocation_granularity);
	unsigned int segoffset=(actualcolumn % allocation_granularity);
	return history[segid][segoffset*nontrivials.len+n];
}


void spike_history_new::recalculate_size(){
	if (mem_available < allocation_granularity) {
		unsigned int newsize = windowsize+allocation_granularity;
		segmentcount++;
		unsigned int **newhist=new unsigned int*[segmentcount];

		unsigned int *newseg=new unsigned int[nontrivials.len*allocation_granularity];
		int segid=matrix_start/allocation_granularity;
		for (int column=0;column<matrix_start%allocation_granularity;column++) {
			for (int row=0;row<nontrivials.len;row++) {
				int idx=nontrivials.len*column+row;
				newseg[idx]=history[segid][idx];
			}
		}
		for (int i=0;i<segid;i++)
			newhist[i]=history[i];
		newhist[segid]=newseg;
		for (int i=segid+1;i<segmentcount;i++)
			newhist[i]=history[i-1];
		delete[] history;
		history=newhist;
		mem_available+=allocation_granularity;
		matrix_start+=allocation_granularity;
		windowsize=newsize;
		cout<<"windowssize= " << windowsize<< endl;
#ifdef VERBOSE
#endif
	}
}

//define dorecord
void spike_history_new::pack(){
	vec<unsigned int> reference(windowsize-mem_available);
	vec<unsigned int> current(windowsize-mem_available);

	partial_decode(reference,0,0);

	int numok=windowsize-mem_available;

	for (state_identifyer i=1; i < nontrivials.len; i++) {
		partial_decode(current,i,numok);
		while (numok>0) {
			if (current[numok-1] == reference[numok-1]) {
				break;
			}
			numok--;
		}
		if (numok==0)
        	break;
	}
	if (numok > 0) {
		for (state_identifyer i=0; i < trivials.len; i++) {
			partial_decode(current,i+nontrivials.len);

			while (numok>0) {
				if (current[numok-1] == reference[numok-1]) {
					break;
				}
				numok--;
			}
		}
	}


	for (int i=0; i < numok; i++) {
		tmppath[i+offset] = reference[i];
	}

	offset+=numok;
	matrix_start = (matrix_start+numok) % windowsize;
	mem_available+=numok;
	recalculate_size();
}


void spike_history_new::finalize(unsigned int max_state){
	vec<unsigned int> reference(windowsize-mem_available);
//	unsigned int maxstateidx=max_state.stateidx+(max_state.trivial?nontrivials.len:0);

	partial_decode(reference,max_state,0);
	for (int i=0; i < reference.len; i++) {
		tmppath[i+offset] = reference[i];
	}
	offset+=reference.len;
	tmppath[offset]=max_state;
	
	mlseq.setLength(tmppath.len);

	for (int i=0; i < tmppath.len; i++) {
		mlseq[i] = tmppath[i];
	}
}



void spike_history_new::partial_decode(vec<unsigned int> &v, unsigned int max_state, unsigned int maxsteps){
	int path_t=windowsize-mem_available-1;
	unsigned int laststate=max_state;
	unsigned int curt=current_time-1;
	for (int t=windowsize-mem_available-1; t >= 0; t--) {
		if (laststate < nontrivials.len) {  //check if it is a nontrivial one
			v[path_t]=getelem(curt,laststate);
		} else { // trivial case
			v[path_t]=rewound_states[laststate-nontrivials.len];
		}
		curt--;
		laststate=v[path_t];
		path_t--;
		if (maxsteps > 0) {
			maxsteps--;
			if (maxsteps==0)
			break;
		}
	}
}


void spike_history_new::step_time(){
	current_time++;
	mem_available--;
	if (mem_available==0) {
		pack();
	}
}

unsigned int *spike_history_new::getCurrentColumn(){
	unsigned int actualcolumn=(current_time-offset+matrix_start) % (segmentcount*allocation_granularity);
	unsigned int segid=(actualcolumn / allocation_granularity);
	unsigned int segoffset=(actualcolumn % allocation_granularity);
	return &history[segid][segoffset*nontrivials.len];
}

/******************************************************************************/

spike_history::spike_history(vec<unsigned int> &in_numstates,
							 vec<state_type> &nontrivial,
							 vec<state_type> &trivial,
							 unsigned int time,
							 unsigned int overlaps,
							 vec<state_identifyer> &rewound_states,
							 vec<state_type> &mlseq)
	: nontrivial(nontrivial),
	  trivial(trivial),
	  numstates(in_numstates),
	  current_time(0),
	  lookup_nontrivial(nontrivial),
	  mlseq(mlseq),
	  history(0),
	  rewound_states(rewound_states),
	  overlaps(overlaps)
{
	tmppath.setLength(time);

	offset=0;

	windowsize=0;
	int k=1;
	for (int i=0; i < numstates.len; i++)
		if (numstates[i]>windowsize)
			windowsize=numstates[i];
	allocation_granularity = k*windowsize;
	windowsize *= k*2;
	mem_available=windowsize;
	matrix_start=0;
	history.setLength(windowsize*nontrivial.len); //indexing t*nontrivial+state

	rewound_trivial.setLength(trivial.len);
	rewind_trivial_amount.setLength(trivial.len);
	for (int i=0; i < trivial.len; i++) {
		rewind_trivial_amount[i] = 0;
		rewound_trivial[i]=trivial[i];
		while (istrivial(rewound_trivial[i],overlaps,numstates)) {
			rewound_trivial[i]=rewindtrivialstate(rewound_trivial[i],numstates);
			rewind_trivial_amount[i]++;
		}
	}
}

void spike_history::recalculate_size(){
	if (mem_available < allocation_granularity) {
		unsigned int newsize = windowsize+allocation_granularity;
		state_identifyer *tmp = new state_identifyer[newsize*nontrivial.len];
		for (int t=0; t < windowsize-mem_available; t++) {
			int t_mod=(t+matrix_start) % windowsize;
			for (int state=0; state < nontrivial.len; state++) {
				tmp[t*nontrivial.len+state]=history.data[t_mod*nontrivial.len+state];
			}
		}

		mem_available+=allocation_granularity;
		matrix_start=0;
		delete[] history.data;
		history.data = tmp;
		history.len=newsize*nontrivial.len;
		windowsize=newsize;
		current_time=windowsize-mem_available;
	}
#ifdef VERBOSE
	cout<<"windowssize= " << windowsize<< endl;
#endif
}



void printstate(FILE* file, state_type state, const vec<unsigned int> &numstates){
	vec<unsigned int> sstate;
	splitState(state,numstates,sstate);
	fprintf(file,"[");
	for (int i=0; i<numstates.len; i++) {
		if (sstate[i]<10)
			fprintf(file,"0%d",sstate[i]);
		else
			fprintf(file,"%d",sstate[i]);
		if (i != numstates.len-1)
			fprintf(file," ");
	}
	fprintf(file,"]\t");
}

//define dorecord
void spike_history::pack(){
	vec<state_identifyer> reference(windowsize-mem_available);
	vec<state_identifyer> current(windowsize-mem_available);

	partial_decode(reference,0,0);

	int numok=windowsize-mem_available;

#ifdef dorecord
	FILE *file = fopen("log.txt", "w");
	for (int idx=0; idx < reference.len; idx++)
		printstate(file,reference[idx],numstates);
	printstate(file,nontrivial[0],numstates);
	fprintf(file,"\n");
#endif

#ifdef VERBOSE
	cout << "partial decoding nontrivial: "<< endl;
#endif
	for (state_identifyer i=1; i < nontrivial.len; i++) {
		partial_decode(current,i,numok);
#ifdef VERBOSE
		if (i % 1000==0) {
			cout << 100.0 *(double) i / double(nontrivial.len) << "%" << endl;
		}
#endif
#ifdef dorecord
		for (int idx=0; idx < current.len; idx++)
			printstate(file,current[idx],numstates);
		printstate(file,nontrivial[i],numstates);
		fprintf(file,"\n");
#endif
		while (numok>0) {
			if (current[numok-1] == reference[numok-1]) {
				break;
			}
			numok--;
		}
		if (numok==0)
        	break;
	}
#ifdef VERBOSE
	cout << "partial decoding trivial: "<< endl;
#endif
	if (numok > 0) {
		for (state_identifyer i=0; i < trivial.len; i++) {
			partial_decode(current,i+nontrivial.len);
#ifdef VERBOSE
			if (i % 1000==0) {
				cout << 100.0 *(double) i / double(trivial.len) << "%" << endl;
			}
#endif

#ifdef dorecord
			for (int idx=0; idx < current.len; idx++)
				printstate(file,current[idx],numstates);
			printstate(file,trivial[i],numstates);
			fprintf(file,"\n");
#endif
			while (numok>0) {
				if (current[numok-1] == reference[numok-1]) {
					break;
				}
				numok--;
			}
		}
	}

#ifdef dorecord
	fclose(file);
#endif

	for (int i=0; i < numok; i++) {
		tmppath[i+offset] = reference[i];
	}
//	std::cout << numok << endl;
	offset+=numok;
	matrix_start = (matrix_start+numok) % windowsize;
	mem_available+=numok;

	recalculate_size();
}


void spike_history::finalize(state_identifyer max_state){
	vec<state_identifyer> reference(windowsize-mem_available);

	partial_decode(reference,max_state,0);
	for (int i=0; i < reference.len; i++) {
		tmppath[i+offset] = reference[i];
	}
	offset+=reference.len;
	tmppath[offset]=max_state;
	
	mlseq.setLength(tmppath.len);

	for (int i=0; i < tmppath.len; i++) {
//		cout << i << "\t" << tmppath[i] <<endl;
		mlseq[i] = calc_actual_state(tmppath[i],nontrivial,trivial);
	}
}



void spike_history::partial_decode(vec<state_identifyer> &v, state_identifyer max_state, unsigned int maxsteps, unsigned int offset){
	int path_t=windowsize-mem_available-1-offset;
	state_identifyer laststate=max_state;
	for (int t=windowsize-mem_available-1-offset; t >= 0; t--) {
		if (laststate < nontrivial.len) {  //check if it is a trivial one
			v[path_t]=history[((t+matrix_start) % windowsize)*nontrivial.len+laststate];
		} else {
			v[path_t]=rewound_states[laststate-nontrivial.len];
		}
		laststate=v[path_t];
		path_t--;
		if (maxsteps > 0) {
			maxsteps--;
			if (maxsteps==0)
			break;
		}
	}
}


void spike_history::step_time(){
	current_time++;
	current_time=current_time%windowsize;
	mem_available--;
	if (mem_available==0) {
		pack();
	}
}

state_identifyer *spike_history::getCurrentColumn(){
	return &history[current_time*nontrivial.len];
}

/******************************************************************************/

void statespace_enum::count(){
	trivialcount = 0;
	nontrivialcount = 0;
	for (int k=0;k<=overlaps;k++) {
		count_if_k_are_active(k);
	}
}

void statespace_enum::count_if_k_are_active(unsigned int k){
	if (k == 0) {
		nontrivialcount++;
	} else if (k <= numstates.len) {
		max_recursion = k;
		recursive_count(0,numstates.len-k,0,0);
	}
#ifndef GNUMEX    
     else
		throw "we did something wrong!!";
#endif
}

void statespace_enum::recursive_count(int start, int stop, unsigned int recursion,int nonones){
	if (recursion == max_recursion) {
		if (max_recursion==numstates.len) {
			trivialcount++;
		} else {
			if (overlaps-nonones == 0)
				trivialcount++;
			else
				nontrivialcount++;
		}
	} else if (recursion < max_recursion) {
		for (int i=start;i<=stop;i++) {
			recursive_count(i+1,min(stop+1,numstates.len-1),recursion+1,nonones);
			for (int j=2;j<numstates[i];j++) {
				recursive_count(i+1,min(stop+1,numstates.len-1),recursion+1,nonones+1);
			}
		}
	}
#ifndef GNUMEX    
     else
		throw "we did something wrong!!";
#endif

}

void statespace_enum::getStates(vec<state_type> &trivial,vec<state_type> &nontrivial){
	count();
	trivial.setLength(trivialcount);
	nontrivial.setLength(nontrivialcount);
	trivial_ref = &trivial;
	nontrivial_ref = &nontrivial;
	curfillpos_trivial=0;
	curfillpos_nontrivial=0;

	cur_split_state.setLength(numstates.len);
	for (int i=0;i<cur_split_state.len;i++) {
		cur_split_state[i]=0;
	}
	for (int k=0;k<=overlaps;k++) {
		fill_if_k_are_active(k);
	}
//define dostaterecord
#ifdef dostaterecord
	FILE *file = fopen("e:/trivial.txt", "w");
	for (int idx=0; idx < trivial.len; idx++){
		printstate(file,trivial[idx],numstates);
		fprintf(file,"\n");
	}
	fclose(file);

	file = fopen("e:/nontrivial.txt", "w");
	for (int idx=0; idx < nontrivial.len; idx++){
		printstate(file,nontrivial[idx],numstates);
		fprintf(file,"\n");
	}
	fclose(file);
#endif

}

void statespace_enum::recursive_fill(int start, int stop, unsigned int recursion,int nonones){
	if (recursion == max_recursion) {
		if (max_recursion==numstates.len) {
			addtrivial(combineStates(cur_split_state,numstates));
		} else {
			if (overlaps-nonones == 0)
				addtrivial(combineStates(cur_split_state,numstates));
			else
				addnontrivial(combineStates(cur_split_state,numstates));
		}
	} else if (recursion < max_recursion) {
		for (int i=start;i<=stop;i++) {
			cur_split_state[i]=1;
			recursive_fill(i+1,min(stop+1,numstates.len-1),recursion+1,nonones);
			for (int j=2;j<numstates[i];j++) {
				cur_split_state[i]=j;
				recursive_fill(i+1,min(stop+1,numstates.len-1),recursion+1,nonones+1);
			}
			cur_split_state[i]=0;
		}
	}
#ifndef GNUMEX    
     else
		throw "we did something wrong!!";
#endif

}

void statespace_enum::fill_if_k_are_active(unsigned int k){
	if (k == 0) {
		addnontrivial(0);
	} else if (k <= numstates.len) {
		max_recursion = k;
		recursive_fill(0,numstates.len-k,0,0);
	}
#ifndef GNUMEX    
     else
		throw "we did something wrong!!";
#endif
}


void statespace_enum::addtrivial(state_type state){
	trivial_ref->data[curfillpos_trivial] = state;
	curfillpos_trivial++;
}

void statespace_enum::addnontrivial(state_type state){
	nontrivial_ref->data[curfillpos_nontrivial] = state;
	curfillpos_nontrivial++;
}

/******************************************************************************/
#endif
