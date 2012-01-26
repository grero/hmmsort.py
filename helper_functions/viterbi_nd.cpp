/*
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2008 Joshua Herbst, Stephan Gammeter
% 
% viterbi_nd is free software: you can redistribute it and/or modify it under 
% the terms of the GNU General Public License as published by the Free Software 
% Foundation, either version 3 of the License, or (at your option) any later version.
% 
% viterbi_nd is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
% without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
% PURPOSE.  See the GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License along with 
% this program. If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*/

#define __USE_LARGEFILE64
#define _LARGEFILE_SOURCE
#define _FILE_OFFSET_BITS 64
//efine WIN32
#include <mex.h> 
#include "viterbi.cpp"

int mx_GetLen(const mxArray *a) {
    return std::max(mxGetM(a),mxGetN(a));
}

void loadArgToVec_uint(const mxArray *arg, vec<unsigned int> &v){
    int len = mx_GetLen(arg);
    v.setLength(len);
    
    double *data=mxGetPr(arg);
    for (int i=0;i<len;i++) {
        v.data[i]=round_d2i(data[i]);
    }
}

void loadArgToVec_double(const mxArray *arg,vec<double> &v){
    int len = mx_GetLen(arg);
    v.setLength(len);
    
    double *data=mxGetPr(arg);
    for (int i=0;i<len;i++) {
        v.data[i]=data[i];
    }
}

void mexFunction ( int nlhs,mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
        int m,n,size,i,j;

        if (nrhs < 8 ) {
            mexErrMsgTxt ("Funktion benoetigt mindestens 8 Argumente!");
        } else if (nrhs > 9) {
            mexErrMsgTxt ("Funktion hat hoechstens 9 Argumente!");
        } else if (nlhs > 2) {
            mexErrMsgTxt ("Funktion gibt maximal zwei Argumente zurueck!");
        }
        
        //(observerd_data,p,numstates,cinv,mu_cell,initial_guess)
        

        vec<double> observed_data;
        loadArgToVec_double(prhs[0],observed_data);
        
        vec<double> p;
        loadArgToVec_double(prhs[1],p);

        double *cinv = mxGetPr(prhs[2]);

        vec<unsigned int> numstates;
        loadArgToVec_uint(prhs[3],numstates);
                
        int mu_cell_len=mx_GetLen(prhs[4]);
        if (mu_cell_len != p.len) {
            mexErrMsgTxt ("Inconsistent Parameters");
        }
        vec<double> *mu_cell=new vec<double>[mu_cell_len];
        for (int i=0;i<mu_cell_len;i++){
            loadArgToVec_double(mxGetCell(prhs[4],i),mu_cell[i]);
        }

        vec<double> initial_guess;
        loadArgToVec_double(prhs[5],initial_guess);
        
        double overlaps = *mxGetPr(prhs[6]);
		double dim = *mxGetPr(prhs[7]);

        
        
        vec<state_type> mlseq;
        double ll;

        
        if (nrhs == 9)
        {
            /* Input must be a string. */
            if (mxIsChar(prhs[8]) != 1)
                mexErrMsgTxt("Input must be a string.");

            /* Get the length of the input string. */
            int buflen = (mxGetM(prhs[8]) * mxGetN(prhs[8])) + 1;

            /* Allocate memory for input and output strings. */
            char * filename = (char*)mxCalloc(buflen, sizeof(char));

            /* Copy the string data from prhs[0] into a C string 
             * input_buf. */
            int status = mxGetString(prhs[8], filename, buflen);
               if (status != 0) 
            mexWarnMsgTxt("Not enough space. String is truncated.");            
            
/*            savevec(observed_data,"observed_data.data");
            savevec(p,"p.data");
            savevec(numstates,"numstates.data");
        
            vec<double> tmp;
#ifndef _Windows
            std::swap(tmp.data,cinv);
#else
            swap(tmp.data,cinv);
#endif
            tmp.len=round_d2i(dim*dim);
            savevec(tmp,"cinv.data");
#ifndef _Windows
            std::swap(tmp.data,cinv);
#else
            swap(tmp.data,cinv);
#endif
            tmp.len=0;
        
            savemat(mu_cell,numstates.len,"mu_cell.data");
            saveval(overlaps,"overlaps.data");
            saveval(dim,"dim.data");*/
            #ifdef WIN32
            ll=viterbi_nd(observed_data,p,cinv,numstates,mu_cell,initial_guess,mlseq,round_d2i(overlaps),round_d2i(dim));
            #else
            ll=viterbi_nd_record(observed_data,p,cinv,numstates,mu_cell,initial_guess,mlseq,round_d2i(overlaps),round_d2i(dim),filename);
            #endif
            
/*            savevec(mlseq,"output.data");

        	FILE *file = fopen("output_states.data", "w");
        	for (int idx=0; idx < mlseq.len; idx++) {
        		printstate(file,mlseq[idx],numstates);
            	fprintf(file,"\n");
            }
            fclose(file);*/
        } else {
            ll=viterbi_nd(observed_data,p,cinv,numstates,mu_cell,initial_guess,mlseq,round_d2i(overlaps),round_d2i(dim));
        }

        if (nlhs == 2) {
            plhs[1] = mxCreateDoubleMatrix (1,1,mxREAL);
            double *pll;
            pll=mxGetPr(plhs[1]);
            *pll=ll;
        }

        
        plhs[0] = mxCreateDoubleMatrix (1,mlseq.len,mxREAL);
        double *dbl_mlseq;
        dbl_mlseq=mxGetPr(plhs[0]);
        for (int i=0;i<mlseq.len;i++){
            dbl_mlseq[i] = mlseq.data[i];
        }
}
