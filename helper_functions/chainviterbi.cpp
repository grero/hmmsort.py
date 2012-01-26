#ifndef _CHAINVITERBI_HPP_INCLUDED
#define _CHAINVITERBI_HPP_INCLUDED

#include <math.h>
#include <errno.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <assert.h>
#include <string>
#include "vec.hpp"
#include "viterbi_helpers.hpp"
#define PI 3.14159265

#if defined(WIN32)
#	include <windows.h>
#endif

#ifndef NULL
#  if defined(__cplusplus) || defined(WIN32)
#    define NULL 0
#  else
#    define NULL ((void *)0)
#  endif
#endif



template <class N>
inline unsigned int round_d2i(double d);

/******************************************************************************/


double logd(double d) {
	if (d <= 1e-304) {
		return -1e308*10.0;
	} else {
		return log(d);
	}
}

void logVec(vec<double> &v){
	for (int i=0;i<v.len;i++){
		v.data[i]=logd(v.data[i]);
	}
	return;
}

unsigned int prodVec(const vec<unsigned int> &v) {
	unsigned int result=1;
	for (int i=0;i<v.len;i++){
		result *= v.data[i];
	}
	return result;
}

inline unsigned int round_d2i(double d)
{
	return static_cast<unsigned int> (d<0?d-.5:d+.5);
}


/*static int bits_in_char[256]={0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7};

int bitcount(state_type n){

return bits_in_char [n         & 0xffu]
		  +  bits_in_char [(n >>  8) & 0xffu]
          +  bits_in_char [(n >> 16) & 0xffu]
          +  bits_in_char [(n >> 24) & 0xffu]
          +  bits_in_char [(n >> 32) & 0xffu]
          +  bits_in_char [(n >> 40) & 0xffu]
          +  bits_in_char [(n >> 48) & 0xffu]
          +  bits_in_char [(n >> 56) & 0xffu];
}
  */
/*
MIT Hackmem Count is funky. Consider a 3 bit number as being 4a+2b+c. If we shift it right 1 bit, we have 2a+b. Subtracting this from the original gives 2a+b+c. If we right-shift the original 3-bit number by two bits, we get a, and so with another subtraction we have a+b+c, which is the number of bits in the original number. How is this insight employed? The first assignment statement in the routine computes tmp. Consider the octal representation of tmp. Each digit in the octal representation is simply the number of 1's in the corresponding three bit positions in n. The last return statement sums these octal digits to produce the final answer. The key idea is to add adjacent pairs of octal digits together and then compute the remainder modulus 63. This is accomplished by right-shifting tmp by three bits, adding it to tmp itself and ANDing with a suitable mask. This yields a number in which groups of six adjacent bits (starting from the LSB) contain the number of 1's among those six positions in n. This number modulo 63 yields the final answer. For 64-bit numbers, we would have to add triples of octal digits and use modulus 1023. This is HACKMEM 169, as used in X11 sources. Source: MIT AI Lab memo, late 1970's.
*/


inline int bitcount(unsigned int n)
{
	/* works for 32-bit numbers only    */
	/* fix last line for 64-bit numbers */
	register unsigned int tmp;
	tmp = n - ((n >> 1) & 033333333333)
		- ((n >> 2) & 011111111111);
	 return ((tmp + (tmp >> 3)) & 030707070707) % 63;
}



vec<state_type> *generatelookuptable(state_type state, const vec<unsigned int> &numstates, unsigned int overlaps) {
	vec<unsigned int> sstate;
	vec<unsigned int> zero_indices(numstates.len);
	int zero_count=0;
	splitState(state,numstates,sstate);
	int fixed_zeros=0;
	for (int i=0;i<sstate.len;i++) {
		if (sstate[i] != 0) {
			sstate[i]--;
			if (sstate[i]==0) fixed_zeros++;
		} else {
			zero_indices[zero_count] = i;
			zero_count++;
		}
	}
	zero_indices.setLength(zero_count);
	// there are 2^#0 possible ways to get to that state
	// lookuptbl=zeros(1,2^numzeros);
	int lookuptbllength=(1<<zero_count);
	vec<state_type> *result = new vec<state_type>(lookuptbllength);

	unsigned int elems_count=0;
	for (unsigned int combination=0; combination<lookuptbllength; combination++){
		if (bitcount(combination)+(numstates.len-zero_count-fixed_zeros) <= overlaps) {
			unsigned int curbit = 1;
			for (int bit_idx=0; bit_idx<zero_indices.len; bit_idx++){
				if ((combination & curbit) > 0) {
					sstate[zero_indices[bit_idx]] = numstates[bit_idx]-1;
				} else {
					sstate[zero_indices[bit_idx]] = 0;
				}
				curbit = (curbit << 1);
			}
			result->data[elems_count] = combineStates(sstate,numstates);
			elems_count++;
		}
	}
	result->setLength(elems_count);
	return result;
}

double f(unsigned int instate, unsigned int outstate, double p, unsigned int numstates){
	if ((instate==outstate) && (instate==0)) {
		return (1.0-p);
	} else if ((instate==0) && (outstate==1)) {
		return p;
	} else if ((instate==(numstates-1)) && (outstate==0)) {
		return 1.0;
	} else if (outstate-instate==1) {
		return 1.0;
	} else return 0.0;
}

double f_multiring(unsigned int instate, unsigned int outstate, vec<double> &pstay, vec<double> &pback, unsigned int numstates, unsigned int numringsperneuron){
	if (instate==outstate){
		if ((instate % numringsperneuron) == 0) {
			return pstay[instate/numringsperneuron];
		} else {
			return 0.0;
		}
	}
	if (((instate % numringsperneuron)==0) && ((outstate % numringsperneuron)==0)) {
		if ((outstate-instate)==numringsperneuron) {
			return pback[instate/numringsperneuron];
		} else {
			return 0.0;
		}
	}
	if ((outstate-instate)==1) {
		if ((instate % outstate) != 0) {
			return 1.0;
		} else {
			if (instate==0) {
				return 1.0-pstay[0];
			} else {
				return 1.0-pstay[(instate/numringsperneuron)]-pback[(instate/numringsperneuron)-1];
			}
		}
	} else {
		return 0.0;
	}
}

double calctransmissionprob_multiring(	const vec<unsigned int> &sstate1,
				const vec<unsigned int> &sstate2,
				const vec<double> &pstay,
				const vec<double> &pback,
				const vec<unsigned int> &numstates,
				const vec<unsigned int> &numringsperneuron){
	double res=1.0;
	vec<double> tmp1;
	vec<double> tmp2;
	int c1=0;
	int c2=0;
	for (int i=0;i<sstate1.len;i++) {
		tmp1.setLength(numringsperneuron[i]);
		tmp2.setLength(numringsperneuron[i]-1);
		for (int j=0;j<tmp1.len;j++){
			tmp1[j]=pstay[c1];
			c1++;
		}
		for (int j=0;j<tmp2.len;j++){
			tmp2[j]=pback[c2];
			c2++;
		}
		res *= f_multiring(sstate1[i],sstate2[i],tmp1,tmp2,numstates[i],numringsperneuron[i]);
	}
	return res;
}

double calctransmissionprob(	const vec<unsigned int> &sstate1,
				const vec<unsigned int> &sstate2,
				const vec<double> &p,
				const vec<unsigned int> &numstates){
    double res=1.0;
    for (int i=0;i<sstate1.len;i++) {
        res *= f(sstate1[i],sstate2[i],p[i],numstates[i]);
    }
    return res;
}

inline double a_transpose_cinv_a(const vec<double> &a, double *cinv){
	register double result=0.0;
	for (int i=0; i < a.len; i++) {
		result += a.data[i]*a.data[i]*cinv[i*a.len+i];
	}
	for (int j=0; j < a.len; j++) {
		for (int i=j+1; i < a.len; i++) {
			result += 2.0*a.data[i]*a.data[j]*cinv[i*a.len+j];
		}
	}
	return -0.5*result;
}


#endif

/***************************************************************************************************/
