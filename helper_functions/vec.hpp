#ifndef _VEC_HPP_INCLUDED
#define _VEC_HPP_INCLUDED

#include<errno.h>
using namespace std;

template <class T>
class vec {
public:
	int len;
	T *data;
	vec () : len(0) {data=NULL;};
	vec (int initlen);
	vec (const vec<T>& v);
	~vec ();
	T &operator[](int i);
	const T &operator[](int i) const;
	const vec& operator=(const vec<T>& rhs);
	const vec& operator+=(const vec<T>& rhs);
	const vec& operator+=(const T &rhs);
	const vec& operator*=(const vec<T>& rhs);
	void setLength(int newLen);
};


template <class T>
struct hash_entry{
	T key;
	unsigned int pos;
};

template <class T>
class hashtable : public vec< hash_entry<T> > {
	private:
		int partition(int top, int bottom);
		int binarySearch(int first, int last, const T &key);
		void quicksort(int top, int bottom);
		void sort();

/*		void hash();
		int search(const T &key);*/
	public:
		hashtable(const vec<T> &v, bool sorted=false);
		int lookup(const T &key);
};

template <class T>
void savemat(vec<T> *v, int numvecs ,const char * filename );
template <class T>
void loadmat(vec<T> *v, int numvecs ,const char * filename );
template <class T>
void savevec(const vec<T> &v, const char * filename );
template <class T>
void loadvec(vec<T> &v, const char * filename );
template <class T>
void saveval(const T &v, const char * filename );
template <class T>
void loadval(T &v, const char * filename );
/*
template <class N>
inline N max(N x, N y)
template <class N>
inline N min(N x, N y)
*/

/******************************************************************************/

#ifdef WIN32
#ifndef GNUMEX    
template <class N>
inline void swap(N &x, N &y){
	N tmp;
	tmp = x;
	x=y;
	y=tmp;
}
#endif
#endif


/*
template <class N>
inline N max(N x, N y)
{
	if (x >= y)
	{
		return x;
	}
	return y;
}

template <class N>
inline N min(N x, N y)
{
	if (x <= y)
	{
		return x;
	}
	return y;
}
*/

template <class T>
inline void printval(std::FILE *file,const T &val){
	std::fprintf(file,"Not supported");
}

template <>
inline void printval<float>(std::FILE *file,const float &val){
	std::fprintf(file,"%f\n",val);
}

template <>
inline void printval<double>(std::FILE *file,const double &val){
	std::fprintf(file,"%f\n",val);
}

template <>
inline void printval<unsigned int>(std::FILE *file,const unsigned int &val){
	std::fprintf(file,"%d\n",val);
}

template <>
inline void printval<int>(std::FILE *file,const int &val){
	std::fprintf(file,"%d\n",val);
}


template <class T>
inline void readtval(std::FILE *file,T &val){
	std::fscanf(file,"",val);
}

template <>
inline void readtval<float>(std::FILE *file,float &val){
	std::fscanf(file,"%f",&val);
}

/*template <>
void readtval<double>(std::FILE *file,double &val){
	std::fscanf(file,"%f",&val);
}  */

template <>
inline void readtval<int>(std::FILE *file,int &val){
	std::fscanf(file,"%d",&val);
}

template <>
inline void readtval<unsigned int>(std::FILE *file,unsigned int &val){
	std::fscanf(file,"%d",&val);
}

/******************************************************************************/

#ifndef GNUMEX

template <class T>
void savevec(const vec<T> &v, const char * filename ){
	using namespace std;
	FILE *file = fopen(filename, "w");
	if (file == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return;
	}
	fprintf(file,"%d\n",v.len);
	for (int i=0;i<v.len;i++) {
		printval<T>(file,v.data[i]);
	}
	fclose(file);
}


template <class T>
void savemat(vec<T> *v, int numvecs ,const char * filename ){
	using namespace std;
	FILE *file = fopen(filename, "w");
	if (file == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return;
	}
	fprintf(file,"%d\n",numvecs);
	fprintf(file,"%d\n",v[0].len);
	for (int i=0;i<v[0].len;i++) {
		for (int j=0;j<numvecs;j++) {
			printval<T>(file,v[j].data[i]);
		}
		fprintf(file,"\n");
	}
	fclose(file);
}


template <class T>
void saveval(const T &v, const char * filename ){
	using namespace std;
	FILE *file = fopen(filename, "w");
	if (file == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return;
	}
	printval<T>(file,v);
	fclose(file);
}

template <class T>
void loadmat(vec<T> **v, int &numvecs ,const char * filename ){
	using namespace std;
	FILE *file = fopen(filename, "r");
	if (file == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return;
	}
	readtval<int>(file,numvecs);
	*v = new vec<T>[numvecs];
	int veclen;
	readtval<int>(file,veclen);
	for (int j=0;j<numvecs;j++)
		(*v)[j].setLength(veclen);

	float d2;

	for (int i=0;i<(*v)[0].len;i++) {
		for (int j=0;j<numvecs;j++) {

			readtval<float>(file,d2);
			(*v)[j].data[i] = d2;
		}
	}
	fclose(file);
}


template <class T>
void loadvec(vec<T> &v, const char * filename ){
	using namespace std;
	ifstream f;
	f.open(filename);
	if (!f.is_open()) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return;
	}
	int len;
	f >> len;
	v.setLength(len);
	for (int i=0;i<v.len;i++) {
		f >> v.data[i];
	}
	f.close();
}


template <class T>
void loadval(T &v, const char * filename ){
	using namespace std;
	FILE *file = fopen(filename, "r");
	if (file == NULL) {
		#if defined(WIN32)
			fprintf(stderr,"open error %d",GetLastError());
		#else
			fprintf(stderr,"%s\n",strerror(errno));
		#endif
		return;
	}
	readtval<T>(file,v);
	fclose(file);
}

#endif
/******************************************************************************/

template <class T>
const vec<T>& vec<T>::operator=(const vec<T>& rhs) {
	setLength(rhs.len);
	for (int i=0; i < len; i++) {
		data[i] = rhs.data[i];
	}
	return *this;
}

template <class T>
const vec<T>& vec<T>::operator+=(const vec<T>& rhs) {
		if (len != rhs.len) {
            #ifndef GNUMEX
			throw "Non equal vector length";
            #endif
		}
		for (int i=0; i < len; i++) {
			data[i] += rhs.data[i];
		}
		return *this;
}

template <class T>
const vec<T>& vec<T>::operator+=(const T &rhs) {
		for (int i=0; i < len; i++) {
			data[i] += rhs;
		}
		return *this;
}


template <class T>
const vec<T>& vec<T>::operator*=(const vec<T>& rhs) {
		if (len != rhs.len) {
            #ifndef GNUMEX
			throw "Non equal vector length";
            #endif
		}
		for (int i=0; i < len; i++) {
			data[i] *= rhs.data[i];
		}
		return *this;
}


template <class T>
inline T &vec<T>::operator[](int i){
	if ((i>=0) && (i<len)) {
		return data[i];
	} else {
        #ifndef GNUMEX
		throw "invalid subscript";
        #endif
		return data[0];
	}
}

template <class T>
const T &vec<T>::operator[](int i) const {
	if ((i>=0) && (i<len)) {
		return data[i];
	} else {
        #ifndef GNUMEX
		throw "invalid subscript";
        #endif
	}
}

template <class T>
vec<T>::vec(int initlen) {
	len = initlen;
	data = new T[len];
}

template <class T>
vec<T>::vec(const vec<T>& v) {
	len = v.len;
	data = new T[len];
	for (int i = 0; i < len; i++) {
		data[i] = v.data[i];
	}
}


template <class T>
vec<T>::~vec() {
	delete[] data;
}

template <class T>
void vec<T>::setLength(int newLen){
    T *newdata = new T[newLen];
	for (int i=0;i<min(newLen,len);i++){
        newdata[i]=data[i];
    }
    if (data != NULL) {
        delete[] data;
    }
    data = newdata;
	len = newLen;
}

/******************************************************************************/
/*
template <class T>
void hashtable<T>::hash(){

}

template <class T>
int hashtable<T>::search(const T &key){

}*/


template <class T>
hashtable<T>::hashtable(const vec<T> &v, bool sorted){
	vec< hash_entry<T> >::setLength(v.len);
	for (int i=0; i<v.len; i++) {
		//if (i%1000==0) cout << i << " of " << v.len << endl;
		vec< hash_entry<T> >::data[i].key=v.data[i];
		vec< hash_entry<T> >::data[i].pos=i;
	}
//	cout << " done " << endl;
	if (!sorted)
		sort();
}

template <class T>
void hashtable<T>::sort(){
//	cout << " sortstart " << endl;
	quicksort(0,vec< hash_entry<T> >::len-1);
//	cout << " sortend " << endl;
}

template <class T>
int hashtable<T>::lookup(const T &key){
   // function:
   //   Searches sortedArray[first]..sortedArray[last] for key.  
   // returns: index of the matching element if it finds key, 
   //         otherwise  -(index where it could be inserted)-1.
   // parameters:
   //   sortedArray in  array of sorted (ascending) values.
   //   first, last in  lower and upper subscript bounds
   //   key         in  value to search for.
   // returns:
   //   index of key, or -insertion_position -1 if key is not 
   //                 in the array. This value can easily be
   //                 transformed into the position to insert it.
	return binarySearch(0,vec<hash_entry<T> >::len-1,key);
}

template <class T>
int hashtable<T>::binarySearch(int first, int last, const T &key){
   while (first <= last) {
       int mid = (first + last) / 2;  // compute mid point.
       if (key > vec<hash_entry<T> >::data[mid].key) 
           first = mid + 1;  // repeat search in top half.
       else if (key < vec<hash_entry<T> >::data[mid].key) 
           last = mid - 1; // repeat search in bottom half.
       else
           return vec<hash_entry<T> >::data[mid].pos;     // found it. return position /////
   }
   return -1;    // failed to find key
}

//Function to determine the partitions
// partitions the array and returns the middle index (subscript)
template <class T>
int hashtable<T>::partition(int top, int bottom)
{
	T x = vec< hash_entry<T> >::data[top].key;
	int i = top - 1;
	int j = bottom + 1;
	do
	{
		do
		{
			j--;
		} while (x < vec< hash_entry<T> >::data[j].key);
		do  
		{
			i++;
		} while (x > vec< hash_entry<T> >::data[i].key);
		if (i < j)
		{
			// switch elements at positions i and j
#ifndef WIN32
			std::swap(vec< hash_entry<T> >::data[i],vec< hash_entry<T> >::data[j]);
#else
			swap(vec< hash_entry<T> >::data[i],vec< hash_entry<T> >::data[j]);
#endif
		}
	} while (i < j);
	// returns middle index
	return j;
}

//Quick Sort Functions for Descending Order
// (2 Functions)
template <class T>
void hashtable<T>::quicksort(int top, int bottom)
{
	// top = subscript of beginning of vector being considered
	// bottom = subscript of end of vector being considered
	// this process uses recursion - the process of calling itself
	int middle;
	if (top < bottom)
	{
//		cout << "sortstep\n" << top << "-" << bottom;
		middle = partition(top, bottom);
		quicksort(top, middle);   // sort top partition
		quicksort(middle+1, bottom);    // sort bottom partition
	}
	return;
}

#endif
