#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "linear.h"
#include <mpi.h>
#include <set>
#include <map>

#ifndef TIMER
#define TIMER
#define NS_PER_SEC 1000000000
double wall_time_diff(int64_t ed, int64_t st)
{
	return (double)(ed-st)/(double)NS_PER_SEC;
}

int64_t wall_clock_ns()
{
#if __unix__
	struct timespec tspec;
	clock_gettime(CLOCK_MONOTONIC, &tspec);
	return tspec.tv_sec*NS_PER_SEC + tspec.tv_nsec;
#else
#if __MACH__
	return 0;
#else
	struct timeval tv;
	gettimeofday( &tv, NULL );
	return tv.tv_sec*NS_PER_SEC + tv.tv_usec*1000;
#endif
#endif
}
#endif
double communication;
double global_n;

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class T> static inline void swap(T* a, int x, int y) { T t=a[x]; a[x]=a[y]; a[y]=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	if(mpi_get_rank()!=0)
		return;
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif
class sparse_operator
{
public:
	static double nrm2_sq(const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += x->value*x->value;
			x++;
		}
		return (ret);
	}

	static double dot(const double *s, const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += s[x->index-1]*x->value;
			x++;
		}
		return (ret);
	}

	static void axpy(const double a, const feature_node *x, double *y)
	{
		while(x->index != -1)
		{
			y[x->index-1] += a*x->value;
			x++;
		}
	}
};

#ifdef __cplusplus
extern "C" {
#endif

extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);
extern int dgemv_(char *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
extern void dgetrf_(int *, int *, double *, int *, int *, int *);
extern void dgetri_(int *, double *, int *, int *, double *, int *, int *);

#ifdef __cplusplus
}
#endif
void inverse(double* A, int N)
{
	int *IPIV = new int[N+1];
	int LWORK = N*N;
	double *WORK = new double[LWORK];
	int INFO;

	dgetrf_(&N,&N,A,&N,IPIV,&INFO);
	dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

	delete[] IPIV;
	delete[] WORK;
}


class l1r_lr_fun
{
public:
	l1r_lr_fun(const problem *prob, double C);
	virtual ~l1r_lr_fun();

	double fun(double *w);
	void loss_grad(double *w, double *g);
	void cal_pg(double *w, double *loss_g, double *pg);
	int init_step(double *w, double *loss_g, double *step, int *index);
	int setselection(double *w, double *loss_g, int *index);
	int get_nr_variable(void);
	int get_nr_data(void);
	void cal_sqrtCD(double *sqrtCD);
	double line_search(double *step, double *old_w, double *loss_g, double *pg, double *step_size, double eta,  double old_f, int *num_line_search_steps, double *w);
	double armijo_line_search(double *step, double *w, double *loss_g, double *step_size, double eta, int *num_line_search_steps, double *delta_ret, int *index = NULL, int indexlength = 0, int localstart = 0, int locallength = 0);
	//	void get_Xw(double *Xw_);
	double *get_Xw();
	virtual void Xv(double *v, double *Xv) = 0;
	virtual void subXv(double *v, int *index, int length, double *Xv) = 0;
	void Hv(double *s, double *Hs);
	double vHv(double *v);
	
protected:
	virtual void XTv(double *v, double *XTv) = 0;

	double global_l;
	double C;
	double *z;
	double *Xw;
	double *D;
	const problem *prob;
	double reg;
	double current_f;
	int start;
	int length;
};

l1r_lr_fun::l1r_lr_fun(const problem *prob, double C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	Xw = new double[l];
	D = new double[l];
	mpi_allreduce(&l, 1, MPI_INT, MPI_SUM);
	global_l = (double) l;
	this->C = C;
	int w_size = get_nr_variable();
	int nr_node = mpi_get_size();
	int rank = mpi_get_rank();
	int shift = int(ceil(double(w_size) / double(nr_node)));
	this->start = shift * rank;
	this->length = min(max(w_size - start, 0), shift);
	if (length == 0)
		start = 0;
}

l1r_lr_fun::~l1r_lr_fun()
{
	delete[] z;
	delete[] Xw;
	delete[] D;
}

double l1r_lr_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;

	reg = 0;

	Xv(w, Xw);

	for(i=start;i<start + length;i++)
		reg += fabs(w[i]);
	for(i=0;i<l;i++)
	{
		double yXw = y[i]*Xw[i];
		if (yXw >= 0)
			f += log(1 + exp(-yXw));
		else
			f += (-yXw+log(1 + exp(yXw)));
	}
	double buffer[2] = {f, reg};
	mpi_allreduce(buffer, 2, MPI_DOUBLE, MPI_SUM);
	communication += 2.0 / global_n;
	f = C * buffer[0];
	reg = buffer[1];
	f += reg;

	current_f = f;
	return(f);
}

double *l1r_lr_fun::get_Xw()
{
	return Xw;
}

void l1r_lr_fun::loss_grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int n = prob->n;

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*Xw[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C*(z[i]-1)*y[i];
	}
	XTv(z, g);
	mpi_allreduce(g, n, MPI_DOUBLE, MPI_SUM);
	communication += 1.0;
}


void l1r_lr_fun::cal_pg(double *w, double *loss_g, double *pg)
{
	int n = prob->n;
	int i;

	for (i=0; i<n; i++)
		if (w[i] > 0)
			pg[i] = 1 + loss_g[i];
		else if (w[i] < 0)
			pg[i] = -1 + loss_g[i];
		else
		{
			if (1 + loss_g[i] < 0)
				pg[i] = 1 + loss_g[i];
			else if (-1 + loss_g[i] > 0)
				pg[i] = -1 + loss_g[i];
			else
				pg[i] = 0;
		}
}

int l1r_lr_fun::init_step(double *w, double *loss_g, double *step, int *index)
{
	int i;
	int indexlength = 0;

	for (i=0; i<length; i++)
	{
		double Gp = loss_g[i] + 1;
		double Gn = loss_g[i] - 1;
		step[i] = -(Gp < 0) * Gp - (Gn > 0) * Gn;
		if (step[i] != 0)
		{
			step[indexlength] = step[i] / global_l;
			index[indexlength] = i;
			indexlength++;
		}
	}
	return indexlength;
}

int l1r_lr_fun::setselection(double *w, double *loss_g, int *index)
{
	int i;
	int indexlength = 0;

	for (i=0; i<length; i++)
		if (w[start + i] != 0 || loss_g[start + i] < -1 || loss_g[start + i] > 1)
		{
			index[indexlength] = i;
			indexlength++;
		}
	return indexlength;
}

int l1r_lr_fun::get_nr_variable(void)
{
	return prob->n;
}


int l1r_lr_fun::get_nr_data(void)
{
	return prob->l;
}

void l1r_lr_fun::cal_sqrtCD(double *sqrtCD)
{
	int l=prob->l;
	for (int i=0;i<l;i++)
		sqrtCD[i] = sqrt(C*D[i]);
}

double l1r_lr_fun::line_search(double *step, double *old_w, double *loss_g, double *pg, double *step_size, double eta,  double old_f, int *num_line_search_steps, double *w)
{
	int i;
	int w_size = get_nr_variable();
	int max_num_linesearch = 1000;
	double f;

	int num_linesearch;

	for(num_linesearch=1; num_linesearch <= max_num_linesearch; num_linesearch++)
	{
		for (i=0; i<w_size; i++)
		{
			w[i] = old_w[i] + (*step_size)*step[i];
			if (old_w[i] > 0)
				w[i] = max(w[i], 0.0);
			else if (old_w[i] < 0)
				w[i] = min(w[i], 0.0);
			else
			{
				if (-(1 + loss_g[i]) > 0)
					w[i] = max(w[i], 0.0);
				else if (-(-1 + loss_g[i]) < 0)
					w[i] = min(w[i], 0.0);
				else
					w[i] = 0;
			}
		}

		f = fun(w);
		double tmp = 0;
		for (i=start; i<start + length; i++)
			tmp += pg[i]*(w[i]-old_w[i]);
		mpi_allreduce(&tmp, 1, MPI_DOUBLE, MPI_SUM);
		communication += 1.0 / global_n;

		if (f - old_f - eta*tmp <= 0)
			break;
		(*step_size) *= 0.5;
	}
	info("step size %g\n", (*step_size));

	if (num_linesearch >= max_num_linesearch)
		return 0;

	*num_line_search_steps = num_linesearch;

	return f;
}

double l1r_lr_fun::armijo_line_search(double *step, double *w, double *loss_g, double *step_size, double eta, int *num_line_search_steps, double *delta_ret, int *index, int indexlength, int localstart, int locallength)
{
	double *y=prob->y;
	int i;
	int inc = 1;
	int l = prob->l;
	int max_num_linesearch = 100;
	double localstepsize = (*step_size);

	double reg_diff = 0;

	int num_linesearch;
	double delta = 0;
	if (indexlength > 0)
	{
		for (i=0;i<locallength;i++)
			reg_diff += fabs(w[index[localstart + i]] + step[localstart + i]) - fabs(w[index[localstart + i]]);
		for (i=0;i<locallength;i++)
			delta += loss_g[index[localstart + i]] * step[localstart + i];
	}
	else
	{
		for (i=0;i<length;i++)
			reg_diff += fabs(w[start + i] + step[start + i]) - fabs(w[start + i]);
		for (i=0;i<length;i++)
			delta += loss_g[start + i] * step[start + i];
	}
	delta += reg_diff;
	mpi_allreduce(&delta, 1, MPI_DOUBLE, MPI_SUM);
	*delta_ret = delta;
	communication += 1.0 / global_n;
	delta *= eta;
	double *Xd = new double[l];
	if (indexlength > 0)
		subXv(step, index, indexlength, Xd);
	else
		Xv(step, Xd);

	// For profiling purpose only
	int count = 0;
	for (i=0;i<l;i++)
		if (Xd[i] != 0)
			count++;
	mpi_allreduce(&count, 1, MPI_INT, MPI_SUM);
	info("Nonzero updates of Xw = %d\n",count);
	//Done profiling

	daxpy_(&l, &localstepsize, Xd, &inc, Xw, &inc);
	for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
	{
		double cond = 0;
		for(i=0; i<l; i++)
		{
			double yXw = y[i]*Xw[i];
			if (yXw >= 0)
				cond += log(1 + exp(-yXw));
			else
				cond += (-yXw+log(1 + exp(yXw)));
		}
		cond *= C;
		cond += reg_diff;
		//If profiling gets sparsity, should consider tracking loss as a sum and only update individual losses
		mpi_allreduce(&cond, 1, MPI_DOUBLE, MPI_SUM);
		communication += 1.0 / global_n;
		if (cond + reg - current_f <= delta * localstepsize)
		{
			mpi_allreduce(&reg_diff, 1, MPI_DOUBLE, MPI_SUM);
			communication += 1.0 / global_n;
			current_f = cond + reg;
			reg += reg_diff;
			break;
		}
		else
		{
			localstepsize *= 0.5;
			double factor = -localstepsize;
			daxpy_(&l, &factor, Xd, &inc, Xw, &inc);

			reg_diff = 0;
			if (indexlength > 0)
				for (i=0;i<locallength;i++)
					reg_diff += fabs(w[index[localstart + i]] + localstepsize * step[localstart + i]) - fabs(w[index[localstart + i]]);
			else
				for (i=0;i<length;i++)
					reg_diff += fabs(w[start + i] + localstepsize * step[start + i]) - fabs(w[start + i]);
		}
	}
	delete[] Xd;

	*num_line_search_steps = num_linesearch;
	*step_size = localstepsize;
	if (num_linesearch >= max_num_linesearch)
	{
		info("LINE SEARCH FAILED\n");
		*step_size = 0;
	}

	return current_f;
}

void l1r_lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	double *wa = new double[l];

	Xv(s, wa);
	for(i=0;i<l;i++)
		wa[i] = C*D[i]*wa[i];

	XTv(wa, Hs);
	mpi_allreduce(Hs, w_size, MPI_DOUBLE, MPI_SUM);
	delete[] wa;
}

double l1r_lr_fun::vHv(double *s)
{
	int i;
	int inc = 1;
	int l=prob->l;
	double *wa = new double[l];

	Xv(s, wa);
	double alpha = 0;
	for(i=0;i<l;i++)
		alpha += wa[i] * wa[i] * D[i];
	delete[] wa;
	alpha *= C;
	mpi_allreduce(&alpha, 1, MPI_DOUBLE, MPI_SUM);
	double norm = ddot_(&length, s + start, &inc, s + start, &inc);
	mpi_allreduce(&norm, 1, MPI_DOUBLE, MPI_SUM);
	alpha /= norm;
	communication += 2 / global_n;

	return alpha;
}

class l1r_lr_fun_row: public l1r_lr_fun
{
public:
	l1r_lr_fun_row(const problem *prob_col, double C);
	~l1r_lr_fun_row();

	void Xv(double *v, double *Xv);
	void subXv(double *v, int *index, int length, double *Xv);

private:
	void XTv(double *v, double *XTv);
};


l1r_lr_fun_row::l1r_lr_fun_row(const problem *prob_col, double C):
	l1r_lr_fun(prob_col, C)
{
}

l1r_lr_fun_row::~l1r_lr_fun_row()
{
}


void l1r_lr_fun_row::Xv(double *v, double *Xv)
{

	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l1r_lr_fun_row::subXv(double *v, int *index, int length, double *Xv)
{
	int w_size = get_nr_variable();
	int i;
	double *tmpv = new double[w_size];
	memset(tmpv, 0, sizeof(double) * w_size);
	for (i=0;i<length;i++)
		tmpv[index[i]] = v[i];
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=tmpv[s->index-1]*s->value;
			s++;
		}
	}
	delete[] tmpv;
}

void l1r_lr_fun_row::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}



class l1r_lr_fun_col: public l1r_lr_fun
{
public:
	l1r_lr_fun_col(const problem *prob_col, double C);
	~l1r_lr_fun_col();

	void Xv(double *v, double *Xv);
	void subXv(double *v, int *index, int length, double *Xv);

private:
	void XTv(double *v, double *XTv);
};


l1r_lr_fun_col::l1r_lr_fun_col(const problem *prob_col, double C):
	l1r_lr_fun(prob_col, C)
{
}

l1r_lr_fun_col::~l1r_lr_fun_col()
{
}


void l1r_lr_fun_col::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;
	//	int nnz = 0;
	//	int w_nnz = 0;

	for(i=0;i<l;i++)
		Xv[i]=0;
	for(i=0;i<w_size;i++)
	{
		if (v[i] != 0)
		{
			//			w_nnz++;
			feature_node *s=x[i];
			while(s->index!=-1)
			{
				Xv[s->index-1]+=v[i]*s->value;
				s++;
				//				nnz++;
			}
		}
	}
	//	info("w_nnz %d nnz %d\n", w_nnz, nnz);
}

void l1r_lr_fun_col::subXv(double *v, int *index, int locallength, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=0;
	for(i=0;i<locallength;i++)
	{
		int idx = index[i];
		feature_node *s=x[idx];
		while(s->index!=-1)
		{
			Xv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

void l1r_lr_fun_col::XTv(double *v, double *XTv)
{

	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
	{
		feature_node *s=x[i];
		XTv[i]=0;
		while(s->index!=-1)
		{
			XTv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

// begin of l1 solvers

class OWLQN
{
public:
	OWLQN(const l1r_lr_fun *fun_obj, double eps = 0.1, int m=10, double eta = 1e-4, int max_iter = 5000);
	~OWLQN();

	void owlqn(double *w);
	void set_print_string(void (*i_print) (const char *buf));

protected:
	double eps;
	double eta;
	int max_iter;
	int M;
	l1r_lr_fun *fun_obj;
	void info(const char *fmt,...);
	void (*owlqn_print_string)(const char *buf);
	int start;
	int length;
	int *recv_count;
	int *displace;
private:
	void update_inner_products(double** inner_product_matrix, int k, int DynamicM, double* s, double* y, double* pg);
	void TwoLoop(double** inner_product_matrix, double* rho, int DynamicM, int k, double* delta);
};

class PLBFGS: public OWLQN
{
public:
	PLBFGS(const l1r_lr_fun *fun_obj, double eps = 0.1, int m=10, double inner_eps = 0.01, int max_inner = 100, double eta = 1e-4, int max_iter = 5000);
	~PLBFGS();

	void plbfgs(double *w);
private:
	void prox_grad(double *w, double *g, double *local_step, double *oldd, double alpha = 0, int indexlength = -1);
	double inner_eps;
	int max_inner;
	void update_inner_products(double **inner_product_matrix, int k, int DynamicM, double *s, double *y);
	void compute_R(double *R, int DynamicM, double **inner_product_matrix, int k, double gamma);
	void SpaRSA(double *w, double *loss_g, double *R, double *s, double *y, double gamma, double *local_step, int DynamicM, int *index, int indexlength);
};

class SPARSA: public OWLQN
{
public:
	SPARSA(const l1r_lr_fun *fun_obj, double eps = 0.1, double eta = 1e-4, int max_iter = 5000);
	~SPARSA();

	void sparsa(double *w);
private:
	void prox_grad(double *w, double *g, double *local_step, double alpha, int vectorsize);
};

static void default_print(const char *buf)
{
	fputs(buf,stdout);
	fflush(stdout);
}

void OWLQN::info(const char *fmt,...)
{
	if(mpi_get_rank()!=0)
		return;
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*owlqn_print_string)(buf);
}

void OWLQN::set_print_string(void (*print_string) (const char *buf))
{
	owlqn_print_string = print_string;
}

OWLQN::OWLQN(const l1r_lr_fun *fun_obj, double eps, int m, double eta, int max_iter)
{
	this->fun_obj=const_cast<l1r_lr_fun *>(fun_obj);
	this->eps=eps;
	this->max_iter=max_iter;
	this->eta = eta;
	owlqn_print_string = default_print;
	this->M = m;
	int w_size = this->fun_obj->get_nr_variable();
	int nr_node = mpi_get_size();
	int rank = mpi_get_rank();
	int shift = int(ceil(double(w_size) / double(nr_node)));
	this->recv_count = new int[nr_node];
	this->displace = new int[nr_node];
	this->start = shift * rank;
	this->length = min(max(w_size - start, 0), shift);
	if (length == 0)
		start = 0;
	int counter = 0;
	for (int i=0;i<nr_node;i++)
	{
		recv_count[i] = shift;
		displace[i] = counter;
		counter += shift;
		if (counter >= w_size)
		{
			counter = 0;
			shift = 0;
		}
		else if (counter + shift > w_size)
			shift = w_size - counter;
	}
}

OWLQN::~OWLQN()
{
	delete[] recv_count;
	delete[] displace;
}

void OWLQN::owlqn(double *w)
{
	int n = fun_obj->get_nr_variable();
	int i;
	int k = 0;
	double f;
	int iter = 1, inc = 1;
	double *loss_g = new double[n];
	double *pg = new double[n];
	double *old_w = new double[n];
	double *old_loss_g = new double[length];
	double *step = new double[n];
	double *local_step = new double[length];
	double accumulated_time = 0;
	int nnz_ops = 0;
	double pgnorm;
	double init_step_size = 1;
	double step_size;
	int num_linesearch;
	char N[] = "N";
	double one = 1.0;
	double **inner_product_matrix = new double*[2*M+1];
	double* delta = new double[M * 2 + 1];
	for (i=0; i < 2 * M + 1; i++)
		inner_product_matrix[i] = new double[2*M+1];

	// calculate gradient norm at w=0 for stopping condition.
	double *w0 = new double[n];
	for (i=0; i<n; i++)
		w0[i] = 0;
	fun_obj->fun(w0);
	nnz_ops++;	
	fun_obj->loss_grad(w0, loss_g);
	nnz_ops++;	
	fun_obj->cal_pg(w0, loss_g, pg);
	double pgnorm0 = dnrm2_(&n, pg, &inc);
	pgnorm = pgnorm0;
	delete [] w0;

	iter = 0;

	double *s = new double[M*length];
	double *y = new double[M*length];
	double *rho = new double[M];

	global_n = (double)n;
	communication = 0;
	f = fun_obj->fun(w);
	nnz_ops++;	
	int64_t timer_st = wall_clock_ns(), timer_ed;

	while (iter < max_iter)
	{
		int DynamicM = min(iter, M);
		timer_ed = wall_clock_ns();
		accumulated_time += wall_time_diff(timer_ed, timer_st);

		nnz_ops++;		
		int nnz = 0;
		int w0_gsmall = 0, w0_glarge = 0;
		for (i = 0; i < n; i++)
		{
			old_w[i] = w[i];
			if (w[i] != 0)
				nnz++;
			else
			{
				if (fabs(loss_g[i]) > 1)
					w0_glarge++;
				else
					w0_gsmall++;
			}
		}
		info("iter=%d m=%d |pg|=%5.3e f=%.14e nnz_ops=%d nnz=%d elapsed_time=%g w0_gsmall=%d w0_glarge=%d communication=%g\n", iter, 2 * DynamicM + 1, pgnorm, f, nnz_ops, nnz, accumulated_time, w0_gsmall, w0_glarge,communication);
		timer_st = wall_clock_ns();

		fun_obj->loss_grad(w, loss_g);
		fun_obj->cal_pg(w, loss_g, pg);

		double s0y0 = 0;
		if (iter == 0)
		{
			pgnorm = ddot_(&length, pg + start, &inc, pg + start, &inc);
			mpi_allreduce(&pgnorm, 1, MPI_DOUBLE, MPI_SUM);
			communication += 1.0 / global_n;
			pgnorm = sqrt(pgnorm);
			inner_product_matrix[0][0] = pgnorm * pgnorm;
		}
		else
		{
			for (i=0;i<length;i++)
				y[k*length+i] = loss_g[start + i] - old_loss_g[i];
			update_inner_products(inner_product_matrix, k, DynamicM, s, y, pg);
			s0y0 = inner_product_matrix[2 * k][2* k + 1];
			pgnorm = sqrt(inner_product_matrix[2 * DynamicM][2 * DynamicM]);
		}

		if (pgnorm <= eps*pgnorm0)
		{
			info("final pgnorm %g\n", pgnorm);
			break;
		}

		if (iter > 0)
		{
			if (s0y0 == 0)
				break;
			rho[k] = 1.0 / s0y0;
			k = (k+1)%M;

			TwoLoop(inner_product_matrix, rho, DynamicM, k, delta);
			memset(local_step, 0, sizeof(double) * length);
			daxpy_(&length, delta + 2 * DynamicM, pg + start, &inc, local_step, &inc);
			double *tmpdelta = new double[DynamicM];
			for (i=0;i<DynamicM;i++)
				tmpdelta[i] = delta[2 * i];
			dgemv_(N, &length, &DynamicM, &one, s, &length, tmpdelta, &inc, &one, local_step, &inc);
			for (i=0;i<DynamicM;i++)
				tmpdelta[i] = delta[2 * i + 1];
			dgemv_(N, &length, &DynamicM, &one, y, &length, tmpdelta, &inc, &one, local_step, &inc);
			delete[] tmpdelta;

			for (i=0;i<length; i++)
				local_step[i] *= (local_step[i] * pg[start + i] < 0);
			MPI_Allgatherv(local_step, length, MPI_DOUBLE, step, recv_count, displace, MPI_DOUBLE, MPI_COMM_WORLD);
			communication += 1.0;
			
			init_step_size = 1;
		}
		else
		{
			for (i=0;i<n;i++)
				step[i] = -pg[i];
			init_step_size = 1/pgnorm;
		}

		if (iter == 0 || iter == 1)
			step_size = init_step_size;
		else if (iter %4 == 0)
			step_size = min(init_step_size, step_size*2);

		f = fun_obj->line_search(step, old_w, loss_g, pg, &step_size, eta, f, &num_linesearch, w);
		nnz_ops += num_linesearch;

		nnz = 0;
		for (i=0; i<length; i++)
		{
			old_loss_g[i] = loss_g[start + i];
			s[k * length + i] = w[start + i] - old_w[start + i];
			if (w[i] != 0)
				nnz++;
		}

		iter++;
	}
	delete[] s;
	delete[] y;
	delete[] loss_g;
	delete[] old_loss_g;
	delete[] pg;
	delete[] step;
	delete[] old_w;
	delete[] local_step;
	for (i=0; i < 2 * M + 1; i++)
		delete[] inner_product_matrix[i];
	delete[] inner_product_matrix;
	delete[] delta;
}

void OWLQN::update_inner_products(double** inner_product_matrix, int k, int DynamicM, double* s, double* y, double* pg)
{
	int i;
	int inc = 1;
	double* buffer = new double[DynamicM * 6 + 1];
	memset(buffer, 0, sizeof(double) * DynamicM * 6 + 1);
	for (i=0;i<DynamicM;i++)
	{
		buffer[6 * i] = ddot_(&length, s + k * length, &inc,  s + i * length, &inc);
		buffer[6 * i + 1] = ddot_(&length, s + k * length, &inc,  y + i * length, &inc);
		buffer[6 * i + 2] = ddot_(&length, y + k * length, &inc,  s + i * length, &inc);
		buffer[6 * i + 3] = ddot_(&length, y + k * length, &inc,  y + i * length, &inc);
		buffer[6 * i + 4] = ddot_(&length, pg + start, &inc,  s + i * length, &inc);
		buffer[6 * i + 5] = ddot_(&length, pg + start, &inc,  y + i * length, &inc);
	}
	buffer[6 * DynamicM] = ddot_(&length, pg + start, &inc,  pg + start, &inc);

	mpi_allreduce(buffer, 6 * DynamicM + 1, MPI_DOUBLE, MPI_SUM);
	communication += (6 * DynamicM + 1.0) / global_n;
	for (i=0;i<DynamicM;i++)
	{
		inner_product_matrix[2 * i][2 * k] = buffer[6 * i];
		inner_product_matrix[2 * i + 1][2 * k] = buffer[6 * i + 1];
		inner_product_matrix[2 * i][2 * k + 1] = buffer[6 * i + 2];
		inner_product_matrix[2 * i + 1][2 * k + 1] = buffer[6 * i + 3];
		inner_product_matrix[2 * i][2 * DynamicM] = buffer[6 * i + 4];
		inner_product_matrix[2 * i + 1][2 * DynamicM] = buffer[6 * i + 5];

		inner_product_matrix[2 * k][2 * i] = inner_product_matrix[2 * i][2 * k];
		inner_product_matrix[2 * k][2 * i + 1] = inner_product_matrix[2 * i + 1][2 * k];
		inner_product_matrix[2 * k + 1][2 * i] = inner_product_matrix[2 * i][2 * k + 1];
		inner_product_matrix[2 * k + 1][2 * i + 1] = inner_product_matrix[2 * i + 1][2 * k + 1];
		inner_product_matrix[2 * DynamicM][2 * i] = inner_product_matrix[2 * i][2 * DynamicM];
		inner_product_matrix[2 * DynamicM][2 * i + 1] = inner_product_matrix[2 * i + 1][2 * DynamicM];
	}
	inner_product_matrix[2 * DynamicM][2 * DynamicM] = buffer[6 * DynamicM];

	delete[] buffer;
}

void OWLQN::TwoLoop(double** inner_product_matrix, double* rho, int DynamicM, int k, double *delta)
{
	int i, j;
	int inc = 1;
	int start = k-1;
	double beta;
	double *alpha = new double[DynamicM];
	int tmp_length = 2 * DynamicM + 1;

	memset(delta, 0, sizeof(double) * size_t(2 * DynamicM));
	memset(alpha, 0, sizeof(double) * size_t(DynamicM));
	delta[2 * DynamicM] = -1;
	if (k < DynamicM)
		start += DynamicM;

	int lastrho = -1;
	for (i = 0; i < DynamicM; i++)
	{
		j = start % DynamicM;

		start--;
		if (rho[j] > 0)
		{
			alpha[j] = rho[j] * ddot_(&tmp_length, delta, &inc, inner_product_matrix[2 * j], &inc);
			delta[2 * j + 1] -= alpha[j];
			if (lastrho == -1)
				lastrho = j;
		}
		else
		{
			fprintf(stderr,"ERROR: rho[%d] <= 0\n",i);
			return;
		}
	}
	if (lastrho != -1)
	{
		double scal = (1.0 / (rho[lastrho] * inner_product_matrix[2 * lastrho + 1][2 * lastrho + 1]));
		dscal_(&tmp_length, &scal, delta, &inc);
		for (i = 0; i < DynamicM; i++)
		{
			start++;
			j = start % DynamicM;
			if (rho[j] <= 0)
				continue;
			beta = alpha[j] - rho[j] * ddot_(&tmp_length, delta, &inc, inner_product_matrix[2 * j + 1], &inc);
			delta[2 * j] += beta;
		}
	}
	delete[] alpha;
}

PLBFGS::PLBFGS(const l1r_lr_fun *fun_obj, double eps, int m, double inner_eps, int max_inner, double eta, int max_iter):
	OWLQN(fun_obj, eps, m, eta, max_iter)
{
	this->inner_eps = inner_eps;
	this->max_inner = max_inner;
}

PLBFGS::~PLBFGS()
{
}

void PLBFGS::plbfgs(double *w)
{
	const double update_eps = 1e-10;//Ensure PD of the LBFGS matrix
	const double sparse_factor = 0.2;//The sparsity threshold of when should we switch to sparse communication
	const double init_step_size = 1;

	int n = fun_obj->get_nr_variable();
	int inc = 1;
	double one = 1.0;
	double mone = -1.0;

	int i, k = 0;
	int iter = 0;
	int skip = 0;
	int DynamicM = 0;
	double f;
	double delta0;
	double delta = 0;
	double step_size = 1;
	double gamma = 0;
	int skip_flag;
	int num_linesearch = 0;
	int64_t timer_st, timer_ed;
	double accumulated_time = 0;
	double all_reduce_buffer[3];
	int indexlength = length;
	int newindexlength = 0;
	int nr_node = mpi_get_size();

	int *sizes = new int[nr_node];
	int *tmprecvcount = new int[nr_node];
	int *tmpdisplace = new int[nr_node];
	double *s = new double[M*length];
	double *y = new double[M*length];
	double *tmpy = new double[length];
	double *tmps = new double[length];
	double *loss_g = new double[n];
	double *local_step = new double[length];
	double *R = new double[4 * M * M];
	double *step;
	int *fullindex;
	double **inner_product_matrix = new double*[M];
	for (i=0; i < M; i++)
		inner_product_matrix[i] = new double[2*M];
	int *index = new int[length];
	for (i=0;i<length;i++)
		index[i] = i;

	// calculate delta in line search with proximal gradient at w=0 for stopping condition.
	double *w0 = new double[n];
	for (i=0; i<n; i++)
		w0[i] = 0;
	fun_obj->fun(w0);
	fun_obj->loss_grad(w0, loss_g);
	double alpha = fun_obj->vHv(loss_g);
	memset(local_step, 0, sizeof(double) * length);
	prox_grad(w0 + start, loss_g + start, local_step, local_step, alpha);
	delete [] w0;
	delta0 = ddot_(&length, loss_g + start, &inc, local_step, &inc);
	for (i=0;i<length;i++)
		delta0 += fabs(local_step[i]);
	mpi_allreduce(&delta0, 1, MPI_DOUBLE, MPI_SUM);
	delta = delta0;
	communication = 0;
	global_n = (double)n;

	f = fun_obj->fun(w);
	timer_st = wall_clock_ns();
	while (iter < max_iter)
	{
		indexlength = length;
		skip_flag = 0;
		timer_ed = wall_clock_ns();
		accumulated_time += wall_time_diff(timer_ed, timer_st);
		int nnz = 0;
		for (i = 0; i < length; i++)
			if (w[start + i] != 0)
				nnz++;
		mpi_allreduce(&nnz, 1, MPI_INT, MPI_SUM);
		info("iter=%d m=%d f=%15.20e stepsize=%g delta=%g nnz=%d elapsed_time=%g communication=%g\n", iter, DynamicM,  f, step_size, delta, nnz, accumulated_time,communication);
		if (step_size == 0 || (iter > 0 && delta / delta0 < eps))
			break;

		timer_st = wall_clock_ns(),
		fun_obj->loss_grad(w, loss_g);

		double s0y0 = 0;
		double s0s0 = 0;
		double y0y0 = 0;
		
		//Decide if we want to add the new pair of s,y and then update the inner products if added
		if (iter != 0)
		{
			daxpy_(&length, &one, loss_g + start, &inc, tmpy, &inc);
			
			all_reduce_buffer[0] = ddot_(&length, tmpy, &inc, tmps, &inc);
			all_reduce_buffer[1] = ddot_(&length, tmps, &inc, tmps, &inc);
			all_reduce_buffer[2] = ddot_(&length, tmpy, &inc, tmpy, &inc);
			mpi_allreduce(all_reduce_buffer, 3, MPI_DOUBLE, MPI_SUM);
			communication += 3.0 / n;
			s0y0 = all_reduce_buffer[0];
			s0s0 = all_reduce_buffer[1];
			y0y0 = all_reduce_buffer[2];
			if (s0y0 >= update_eps * s0s0)
			{
				memcpy(y + (k*length), tmpy, length * sizeof(double));
				memcpy(s + (k*length), tmps, length * sizeof(double));
				gamma = y0y0 / s0y0;
			}
			else
			{
				info("skip\n");
				skip_flag = 1;
				skip++;
			}
			DynamicM = min(iter - skip, M);
		}

		memset(local_step, 0, sizeof(double) * length);
		memset(tmps, 0, sizeof(double) * length);

		if (DynamicM > 0)
		{
			indexlength = fun_obj->setselection(w, loss_g, index);

			if (skip_flag == 0)
			{
				update_inner_products(inner_product_matrix, k, DynamicM, s, y);

				compute_R(R, DynamicM, inner_product_matrix, k, gamma);
				k = (k+1)%M;
			}

			SpaRSA(w + start, loss_g + start, R, s, y, gamma, local_step, DynamicM, index, indexlength);
		}
		else
		{
			alpha = fun_obj->vHv(loss_g);
			info("init alpha = %g\n",alpha);
			prox_grad(w + start, loss_g + start, local_step, local_step, alpha);
//			indexlength = fun_obj->init_step(w + start, loss_g + start, local_step, index);
			daxpy_(&length, &mone, w + start, &inc, local_step, &inc);
		}

		for (i=0;i<indexlength;i++)
			index[i] = index[i] + start;
		newindexlength = indexlength;
		mpi_allreduce(&indexlength, 1, MPI_INT, MPI_SUM);
		communication += 1.0 / n;
		step_size = init_step_size;
		if (indexlength < n * sparse_factor) // conduct sparse communication
		{
			step = new double[indexlength];
			fullindex = new int[indexlength];
			//First decide the size to communicate from each machine
			//Then another round to communicate the sparse vector
			//Finally the whole vectors
			//cost: 2 rounds of sparse communication + 2 sweep of the original sparse index
			memset(sizes, 0, sizeof(int) * nr_node);
			MPI_Allgather(&newindexlength, 1, MPI_INT, sizes, 1, MPI_INT, MPI_COMM_WORLD);
			communication += nr_node / (double) n;
			tmprecvcount[0] = sizes[0];
			tmpdisplace[0] = 0;
			int rank = mpi_get_rank();
			for (i=1;i<nr_node;i++)
			{
				tmprecvcount[i] = sizes[i];
				tmpdisplace[i] = tmpdisplace[i-1] + sizes[i-1];
			}

			MPI_Allgatherv(index, newindexlength, MPI_INT, fullindex, tmprecvcount, tmpdisplace, MPI_INT, MPI_COMM_WORLD);
			MPI_Allgatherv(local_step, newindexlength, MPI_DOUBLE, step, tmprecvcount, tmpdisplace, MPI_DOUBLE, MPI_COMM_WORLD);
			communication += 2.0 * indexlength / n;
			f = fun_obj->armijo_line_search(step, w, loss_g, &step_size, eta, &num_linesearch, &delta, fullindex, indexlength, tmpdisplace[rank], tmprecvcount[rank]);
			delete[] fullindex;
			delete[] step;
			for (i=0;i<newindexlength;i++)
			{
				w[index[i]] += local_step[i] * step_size;
				tmps[index[i] - start] = local_step[i] * step_size;
			}
		}
		else
		{
			step = new double[n];

			if (newindexlength < length)
				for (i=newindexlength-1;i >= 0; i--)
				{
					double tmp = local_step[i];
					local_step[i] = 0;
					local_step[index[i] - start] = tmp;
				}
			MPI_Allgatherv(local_step, length, MPI_DOUBLE, step, recv_count, displace, MPI_DOUBLE, MPI_COMM_WORLD);
			communication += 1.0;
			f = fun_obj->armijo_line_search(step, w, loss_g, &step_size, eta, &num_linesearch, &delta);
			delete[] step;
			for (i=0;i<length;i++)
			{
				w[start + i] += local_step[i] * step_size;
				tmps[i] = local_step[i] * step_size;
			}
		}
		if (iter == 0)
			delta0 = delta;

		memcpy(tmpy, loss_g + start, sizeof(double) * length);
		dscal_(&length, &mone, tmpy, &inc);
		iter++;
	}

	delete[] sizes;
	delete[] tmprecvcount;
	delete[] tmpdisplace;
	delete[] s;
	delete[] y;
	delete[] tmpy;
	delete[] tmps;
	delete[] loss_g;
	delete[] local_step;
	delete[] index;
	for (i=0; i < M; i++)
		delete[] inner_product_matrix[i];
	delete[] inner_product_matrix;
	delete[] R;
}

void PLBFGS::update_inner_products(double **inner_product_matrix, int k, int DynamicM, double *s, double *y)
{
	int i;
	int inc = 1;
	char T[] = "T";
	double zero = 0;
	double one = 1.0;

	double *buffer = new double[DynamicM * 2];
	dgemv_(T, &length, &DynamicM, &one, s, &length, s + k * length, &inc, &zero, buffer, &inc);
	dgemv_(T, &length, &DynamicM, &one, y, &length, s + k * length, &inc, &zero, buffer + DynamicM, &inc);

	mpi_allreduce(buffer, 2 * DynamicM, MPI_DOUBLE, MPI_SUM);
	communication += 2 * DynamicM / global_n;
	for (i=0;i<DynamicM;i++)
	{
		inner_product_matrix[k][2 * i] = buffer[i];
		inner_product_matrix[i][2 * k] = buffer[i];
		inner_product_matrix[k][2 * i + 1] = buffer[DynamicM + i];
	}
	delete[] buffer;
}

void PLBFGS::compute_R(double *R, int DynamicM, double **inner_product_matrix, int k, double gamma)
{
	int i,j;
	int size = 2 * DynamicM;
	int sizesquare = size * size;

	memset(R, 0, sizeof(double) * sizesquare);

	//R is a symmetric matrix
	//(1,1) block, S^T S
	for (i=0;i<DynamicM;i++)
		for (j=0;j<DynamicM;j++)
			R[i * size + j] = gamma * inner_product_matrix[i][2 * j];

	//(2,2) block, D = diag(s_i^T y_i)
	for (i=0;i<DynamicM;i++)
		R[(DynamicM + i) * (size + 1)] = -inner_product_matrix[i][2 * i + 1];

	//(1,2) block, L = tril(S^T Y, -1), and (2,1) block, L^T
	for (i=1;i<DynamicM;i++)
	{
		int idx = (k + 1 + i) % DynamicM;
		for (j=0;j<i;j++)
		{
			int idxj = (k + 1 + j) % DynamicM;
			R[(DynamicM + idxj) * size + idx] = inner_product_matrix[idx][2 * idxj + 1];
			R[idx * size + DynamicM + idxj] = inner_product_matrix[idx][2 * idxj + 1];
		}
	}

	//compute_inverse(R, size);
	inverse(R, size);
}

void PLBFGS::SpaRSA(double *w, double *loss_g, double *R, double *s, double *y, double gamma, double *local_step, int DynamicM, int *index, int indexlength)
{
	const double eta = .01 / 2;
	const double ALPHA_MAX = 1e30;
	const double ALPHA_MIN = 1e-4;
	//Fixed parameters (except max_iter) from Taedong's code

	info("inner_eps = %g\n",inner_eps);
	int i,j;
	double one = 1.0;
	int inc = 1;
	double mgamma = -gamma;
	double mone = -one;
	double zero = 0;
	double dnorm0;
	double dnorm;
	char N[] = "N";
	char T[] = "T";
	//parameters for blas

	int iter = 0;
	int Rsize = 2 * DynamicM;

	double *oldd = new double[indexlength];
	double *oldg = new double[indexlength];
	double *g = new double[indexlength];
	double *subg = new double[indexlength];
	double *subw = new double[indexlength];
	double *ddiff = new double[indexlength];//Note that ddiff is always oldd - newd
	double *SYTd = new double[Rsize + 3];
	double *tmp = new double[Rsize];
	double *subs = new double[indexlength * DynamicM];
	double *suby = new double[indexlength * DynamicM];

	double oldquadratic = 0;
	double alpha = 0;
	double all_reduce_buffer;
	int proxgradcounts = 0;
	if (indexlength == 0)
	{
		while (iter < max_inner)
		{
			if (iter != 0)
			{
				all_reduce_buffer = 0;
				mpi_allreduce(&all_reduce_buffer, 1, MPI_DOUBLE, MPI_SUM);
				communication += 1.0 / global_n;
				alpha = all_reduce_buffer / SYTd[Rsize + 2];
			}
			else
				alpha = gamma;

			alpha = max(ALPHA_MIN, alpha);
			double fun_improve = 0.0;
			double rhs = 0.0;
			double tmpquadratic = 0;
			int times = 0;

			//Line search stopping: f^+ - f < -eta * alpha * |x^+ - x|^2
			while (fun_improve >= rhs)
			{
				times++;
				if (alpha > ALPHA_MAX)
					break;
				proxgradcounts++;

				memset(SYTd, 0, (Rsize + 3) * sizeof(double));
				mpi_allreduce(SYTd, Rsize + 3, MPI_DOUBLE, MPI_SUM);
				communication += (Rsize + 3) / global_n;

				for (i=0;i<Rsize;i++)
				{
					tmp[i] = 0;
					for (j=0;j<Rsize;j++)
						tmp[i] += R[i * Rsize + j] * SYTd[j];
				}
				tmpquadratic = gamma * SYTd[Rsize] - ddot_(&Rsize, SYTd, &inc, tmp, &inc);
				fun_improve = SYTd[Rsize + 1] + (tmpquadratic - oldquadratic) / 2;
				rhs = -eta * alpha * SYTd[Rsize + 2];

				alpha *= 2.0;
			}
			dnorm = sqrt(SYTd[Rsize + 2]);
			if (iter == 0)
				dnorm0 = dnorm;
			oldquadratic = tmpquadratic;

			if (alpha > ALPHA_MAX)
				break;
			if (iter > 0 && dnorm / dnorm0 < inner_eps)
				break;
			iter++;
		}
	}
	else
	{
		for (i=0;i<indexlength;i++)
		{
			subg[i] = loss_g[index[i]];
			subw[i] = w[index[i]];
		}
		for (i=0;i<DynamicM;i++)
		{
			double *tmpsubs = subs + (indexlength * i);
			double *tmps = s + (length * i);
			for (j=0;j<indexlength;j++)
				tmpsubs[j] = tmps[index[j]];
		}
		for (i=0;i<DynamicM;i++)
		{
			double *tmpsuby = suby + (indexlength * i);
			double *tmpy = y + (length * i);
			for (j=0;j<indexlength;j++)
				tmpsuby[j] = tmpy[index[j]];
		}

		memset(oldd, 0, sizeof(double) * indexlength);
		memcpy(oldg, subg, indexlength * sizeof(double));
		memset(local_step, 0, sizeof(double) * indexlength);
		memset(SYTd, 0, (Rsize + 3) * sizeof(double));


		while (iter < max_inner)
		{
			memcpy(g, subg, indexlength * sizeof(double));
			if (iter != 0)
			{
				//compute the gradient of the sub-problem: g + gamma * d - Q R Q^T d
				//Note that RQ^Td is already obtained from the previous round in computing obj value
				daxpy_(&indexlength, &gamma, local_step, &inc, g, &inc);
				dgemv_(N, &indexlength, &DynamicM, &mgamma, subs, &indexlength, tmp, &inc, &one, g, &inc);
				dgemv_(N, &indexlength, &DynamicM, &mone, suby, &indexlength, tmp + DynamicM, &inc, &one, g, &inc);

				//Now grad is ready, compute alpha = y^T s / s^T s, of the subprob
				daxpy_(&indexlength, &mone, g, &inc, oldg, &inc);//get -y of the subprob

				all_reduce_buffer = ddot_(&indexlength, ddiff, &inc, oldg, &inc);
				mpi_allreduce(&all_reduce_buffer, 1, MPI_DOUBLE, MPI_SUM);
				communication += 1.0 / global_n;
				alpha = all_reduce_buffer / SYTd[Rsize + 2];

				memcpy(oldg, g, sizeof(double) * indexlength);
			}
			else
				alpha = gamma;

			alpha = max(ALPHA_MIN, alpha);
			double fun_improve = 0.0;
			double rhs = 0.0;
			double tmpquadratic = 0;
			int times = 0;
			memset(SYTd, 0, (Rsize + 3) * sizeof(double));

			//Line search stopping: f^+ - f < -eta * alpha * |x^+ - x|^2
			while (fun_improve >= rhs)
			{
				times++;
				if (alpha > ALPHA_MAX)
					break;
				prox_grad(subw, g, local_step, oldd, alpha, indexlength);
				proxgradcounts++;
				daxpy_(&indexlength, &mone, subw, &inc, local_step, &inc);
				memcpy(ddiff, oldd, sizeof(double) * indexlength);
				daxpy_(&indexlength, &mone, local_step, &inc, ddiff, &inc);

				//SYTd[Rsize + 2] records ||x^+ - x||^2
				SYTd[Rsize + 2] = ddot_(&indexlength, ddiff, &inc, ddiff, &inc);

				//SYTd[Rsize + 1] records func diff in the terms linear to d: l1 regularizer and g^T d
				SYTd[Rsize + 1] = -ddot_(&indexlength, subg, &inc, ddiff, &inc);
				for (i=0;i<indexlength;i++)
					SYTd[Rsize + 1] += fabs(subw[i] + local_step[i]) - fabs(subw[i] + oldd[i]);
				//SYTd[Rsize] records d^T d
				SYTd[Rsize] = ddot_(&indexlength, local_step, &inc, local_step, &inc);

				dgemv_(T, &indexlength, &DynamicM, &gamma, subs, &indexlength, local_step, &inc, &zero, SYTd, &inc);
				dgemv_(T, &indexlength, &DynamicM, &one, suby, &indexlength, local_step, &inc, &zero, SYTd + DynamicM, &inc);

				mpi_allreduce(SYTd, Rsize + 3, MPI_DOUBLE, MPI_SUM);
				communication += (Rsize + 3) / global_n;

				for (i=0;i<Rsize;i++)
				{
					tmp[i] = 0;
					for (j=0;j<Rsize;j++)
						tmp[i] += R[i * Rsize + j] * SYTd[j];
				}
				//			dsymv_(U, &Rsize, &one, R, &Rsize, SYTd, &inc, &zero, tmp, &inc);
				tmpquadratic = gamma * SYTd[Rsize] - ddot_(&Rsize, SYTd, &inc, tmp, &inc);
				fun_improve = SYTd[Rsize + 1] + (tmpquadratic - oldquadratic) / 2;
				rhs = -eta * alpha * SYTd[Rsize + 2];

				alpha *= 2.0;
			}
			dnorm = sqrt(SYTd[Rsize + 2]);
			if (iter == 0)
				dnorm0 = dnorm;
			oldquadratic = tmpquadratic;

			if (alpha > ALPHA_MAX)
			{
				memcpy(local_step, oldd, sizeof(double) * indexlength);
				break;
			}
			memcpy(oldd, local_step, sizeof(double) * indexlength);

			if (iter > 0 && dnorm / dnorm0 < inner_eps)
				break;
			iter++;
		}
	}
	info("iters = %d, prox_grad = %d\n",iter, proxgradcounts);

	if (iter == 1)
		inner_eps /= 4.0;
}

void PLBFGS::prox_grad(double *w, double *g, double *local_step, double *oldd, double alpha, int indexlength)
{
	if (alpha <= 0)
	{
		int inc = 1;
		alpha = ddot_(&length, g, &inc, g, &inc);
		mpi_allreduce(&alpha, 1, MPI_DOUBLE, MPI_SUM);
		communication += 1.0 / global_n;
		alpha = sqrt(alpha);
	}
	if (indexlength < 0)
		indexlength = length;
	for (int i=0;i<indexlength;i++)
	{
		double u = w[i] + oldd[i] - g[i] / alpha;
		local_step[i] = max(fabs(u) - 1.0 / alpha, 0.0);
		local_step[i] *= ((u > 0) - (u<0));
	}
}

SPARSA::SPARSA(const l1r_lr_fun *fun_obj, double eps, double eta, int max_iter):
	OWLQN(fun_obj, eps, 0, eta, max_iter)
{
}

SPARSA::~SPARSA()
{
}

void SPARSA::sparsa(double *w)
{
	const double ALPHA_MAX = 1e30;
	const double ALPHA_MIN = 1e-4;
	int n = fun_obj->get_nr_variable();

	int i;
	int iter = 0;
	double f = 0;
	double newf = 0;
	double s0y0;
	double dnorm0;
	double dnorm = 0;
	int64_t timer_st, timer_ed;
	double accumulated_time = 0;
	double *oldg = new double[length];

	double *loss_g = new double[n];
	double *step = new double[n];
	double *old_step = new double[length];
	int *index = new int[length];
	int indexlength;
	double alpha = 0;
	int times = 0;

	communication = 0;
	global_n = (double)n;

	f = fun_obj->fun(w);
	timer_st = wall_clock_ns();
	indexlength = length;
	while (iter < max_iter)
	{
		timer_ed = wall_clock_ns();
		accumulated_time += wall_time_diff(timer_ed, timer_st);
		int nnz = 0;
		for (i = 0; i < length; i++)
			if (w[start + i] != 0)
				nnz++;
		mpi_allreduce(&nnz, 1, MPI_INT, MPI_SUM);
		info("iter=%d f=%15.20e alpha=%g #backtracking=%d dnorm=%g nnz=%d elapsed_time=%g communication=%g\n", iter, f, alpha / 2, times, dnorm, nnz, accumulated_time, communication);
		if (iter > 0 && (alpha > ALPHA_MAX || dnorm / dnorm0< eps))
			break;

		timer_st = wall_clock_ns(),
		fun_obj->loss_grad(w, loss_g);

		if (iter != 0)
		{
			s0y0 = 0;
			for (i=0;i<indexlength;i++)
				s0y0 += old_step[i] * (loss_g[index[i]] - oldg[index[i] - start]);
			mpi_allreduce(&s0y0, 1, MPI_DOUBLE, MPI_SUM);
			communication += 1.0 / n;
			alpha = s0y0/(dnorm * dnorm);
		}
		else
			alpha = fun_obj->vHv(loss_g);

		alpha = max(ALPHA_MIN, alpha);
		double fun_improve = 0.0;
		double rhs = 0.0;

		//Line search stopping: f^+ - f < -eta * alpha * |x^+ - x|^2
		times = 0;
		while (fun_improve >= rhs)
		{
			times++;
			if (alpha > ALPHA_MAX)
				break;
			prox_grad(w, loss_g, step, alpha, n);
			//Do not distributedly compute coordinates, because we only compute once and then function value is computed
			newf = fun_obj->fun(step);
			fun_improve = newf - f;
			indexlength = 0;
			dnorm = 0;
			for (i=0;i<length;i++)
			{
				int idx = start + i;
				double tmp = step[idx] - w[idx];
				if (tmp != 0)
				{
					old_step[indexlength] = tmp;
					dnorm += tmp * tmp;
					index[indexlength] = idx;
					indexlength++;
				}
			}
			mpi_allreduce(&dnorm, 1, MPI_DOUBLE, MPI_SUM);
			communication += 1.0 / global_n;
			rhs = -eta * alpha * dnorm / 2;
			alpha *= 2.0;
		}
		f = newf;
		dnorm = sqrt(dnorm);
		if (iter == 0)
			dnorm0 = dnorm;
		memcpy(oldg, loss_g + start, sizeof(double) * length);
		memcpy(w, step, sizeof(double) * n);
		iter++;
	}
	delete[] loss_g;
	delete[] oldg;
	delete[] old_step;
	delete[] step;
	delete[] index;
}

void SPARSA::prox_grad(double *w, double *g, double *step, double alpha, int vectorsize)
{
	for (int i=0;i<vectorsize;i++)
	{
		double u = w[i] - g[i] / alpha;
		step[i] = max(fabs(u) - 1.0 / alpha, 0.0);
		step[i] *= ((u > 0) - (u<0));
	}
}

// end of l1 solvers

static void transpose(const problem *prob, feature_node **x_space_ret, problem *prob_col)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	size_t nnz = 0;
	size_t *col_ptr = new size_t [n+1];
	feature_node *x_space;
	prob_col->l = l;
	prob_col->n = n;
	prob_col->y = new double[l];
	prob_col->x = new feature_node*[n];

	for(i=0; i<l; i++)
		prob_col->y[i] = prob->y[i];

	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			nnz++;
			col_ptr[x->index]++;
			x++;
		}
	}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;

	x_space = new feature_node[nnz+n];
	for(i=0; i<n; i++)
		prob_col->x[i] = &x_space[col_ptr[i]];

	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x_space[col_ptr[ind]].index = i+1; // starts from 1
			x_space[col_ptr[ind]].value = x->value;
			col_ptr[ind]++;
			x++;
		}
	}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;

	*x_space_ret = x_space;

	delete [] col_ptr;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int i;

	std::set<int> label_set;
	for(i=0;i<prob->l;i++)
		label_set.insert((int)prob->y[i]);
	
	int label_size = (int)label_set.size();
	int num_machines = mpi_get_size();
	int max_label_size;
	MPI_Allreduce(&label_size, &max_label_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

	std::vector<int> global_label_sets((max_label_size+1)*num_machines);
	std::vector<int> label_buff(max_label_size+1);

	label_buff[0] = label_size;
	i = 1;
	for(std::set<int>::iterator this_label=label_set.begin();
			this_label!=label_set.end(); this_label++)
	{
		label_buff[i] = (*this_label);
		i++;
	}
	
	MPI_Allgather(label_buff.data(), max_label_size+1, MPI_INT, global_label_sets.data(), max_label_size+1, MPI_INT, MPI_COMM_WORLD);

	for(i=0; i<num_machines; i++)
	{
		int offset = i*(max_label_size+1);
		int size = global_label_sets[offset];
		for(int j=1; j<=size; j++)
			label_set.insert(global_label_sets[offset+j]);
	}

	int nr_class = (int)label_set.size();

	std::map<int, int> label_map;
	int *label = Malloc(int, nr_class);
	i = 0;
	for(std::set<int>::iterator this_label=label_set.begin();
			this_label!=label_set.end(); this_label++)
	{
		label[i] = (*this_label);
		i++;
	}
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
		swap(label[0], label[1]);
	for(i=0;i<nr_class;i++)
		label_map[label[i]] = i;


	// The following codes are similar to the original LIBLINEAR
	int *start = Malloc(int, nr_class);
	int *count = Malloc(int, nr_class);
	for(i=0;i<nr_class;i++)
		count[i] = 0;
	for(i=0;i<prob->l;i++)
		count[label_map[(int)prob->y[i]]]++;

	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[label_map[(int)prob->y[i]]]] = i;
		++start[label_map[(int)prob->y[i]]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
}

static void train_one(const problem *prob, const parameter *param, double *w)
{
	double eps = param->eps;

	int l = prob->l;
	int pos = 0;
	int neg = 0;
	for(int i=0;i<l;i++)
		if(prob->y[i] > 0)
			pos++;
	mpi_allreduce(&pos, 1, MPI_INT, MPI_SUM);
	mpi_allreduce(&l, 1, MPI_INT, MPI_SUM);
	neg = l - pos;

	double primal_solver_tol = (eps*max(min(pos,neg), 1))/l;

	l1r_lr_fun *l1r_fun_obj=NULL;
	problem prob_col;
	feature_node *x_space = NULL;
	transpose(prob, &x_space ,&prob_col);
	l1r_fun_obj = new l1r_lr_fun_col(&prob_col, param->C);

	switch(param->solver_type)
	{
		case L1R_LR_OWLQN:
		{
			OWLQN owlqn_obj(l1r_fun_obj, primal_solver_tol, param->m, param->eta);
			owlqn_obj.set_print_string(liblinear_print_string);
			owlqn_obj.owlqn(w);
			break;
		}
		case L1R_LR_BFGS:
		{
			PLBFGS lbfgs_obj(l1r_fun_obj, primal_solver_tol, param->m, param->inner_eps, param->max_inner_iter, param->eta);
			lbfgs_obj.set_print_string(liblinear_print_string);
			lbfgs_obj.plbfgs(w);
			break;
		}
		case L1R_LR_SPARSA:
		{
			SPARSA sparsa_obj(l1r_fun_obj, primal_solver_tol, param->eta);
			sparsa_obj.set_print_string(liblinear_print_string);
			sparsa_obj.sparsa(w);
			break;
		}
		default:
			if(mpi_get_rank() == 0)
				fprintf(stderr, "ERROR: unknown solver_type\n");
	}
	delete l1r_fun_obj;
	delete [] prob_col.y;
	delete [] prob_col.x;
	delete [] x_space;
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i,j;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	int nr_class;
	int *label = NULL;
	int *start = NULL;
	int *count = NULL;
	int *perm = Malloc(int,l);

	// group training data of the same class
	group_classes(prob,&nr_class,&label,&start,&count,perm);

	model_->nr_class=nr_class;
	model_->label = Malloc(int,nr_class);
	for(i=0;i<nr_class;i++)
		model_->label[i] = label[i];

	// constructing the subproblem
	feature_node **x = Malloc(feature_node *,l);
	for(i=0;i<l;i++)
		x[i] = prob->x[perm[i]];

	int k;
	problem sub_prob;
	sub_prob.l = l;
	sub_prob.n = n;
	sub_prob.x = Malloc(feature_node *,sub_prob.l);
	sub_prob.y = Malloc(double,sub_prob.l);

	for(k=0; k<sub_prob.l; k++)
		sub_prob.x[k] = x[k];

	if(nr_class == 2)
	{
		model_->w=Malloc(double, w_size);

		int e0 = start[0]+count[0];
		k=0;
		for(; k<e0; k++)
			sub_prob.y[k] = +1;
		for(; k<sub_prob.l; k++)
			sub_prob.y[k] = -1;

		for(i=0;i<w_size;i++)
			model_->w[i] = 0;

		train_one(&sub_prob, param, model_->w);
	}
	else
	{
		model_->w=Malloc(double, w_size*nr_class);
		double *w=Malloc(double, w_size);
		for(i=0;i<nr_class;i++)
		{
			int si = start[i];
			int ei = si+count[i];

			k=0;
			for(; k<si; k++)
				sub_prob.y[k] = -1;
			for(; k<ei; k++)
				sub_prob.y[k] = +1;
			for(; k<sub_prob.l; k++)
				sub_prob.y[k] = -1;

			for(j=0;j<w_size;j++)
				w[j] = 0;

			train_one(&sub_prob, param, w);

			for(j=0;j<w_size;j++)
				model_->w[j*nr_class+i] = w[j];
		}
		free(w);
	}

	free(x);
	free(label);
	free(start);
	free(count);
	free(perm);
	free(sub_prob.x);
	free(sub_prob.y);
	return model_;
}


double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	if(nr_class==2)
		return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

static const char *solver_type_table[]=
{
	"L1R_LR", NULL
};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	int nr_w;
	if(model_->nr_class==2)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[0]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var)do\
{\
	if (fscanf(_stream, _format, _var) != 1)\
	{\
		fprintf(stderr, "ERROR: fscanf failed to read the model\n");\
		EXIT_LOAD_MODEL()\
	}\
}while(0)
// EXIT_LOAD_MODEL should NOT end with a semicolon.
#define EXIT_LOAD_MODEL()\
{\
	free(model_->label);\
	free(model_);\
	return NULL;\
}
struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char cmd[81];
	while(1)
	{
		FSCANF(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"[rank %d] unknown solver type.\n", mpi_get_rank());
				EXIT_LOAD_MODEL()
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			FSCANF(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			FSCANF(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			FSCANF(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				FSCANF(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"[rank %d] unknown text in model file: [%s]\n",mpi_get_rank(),cmd);
			EXIT_LOAD_MODEL()
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			FSCANF(fp, "%lf ", &model_->w[i*nr_w+j]);
	}


	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

// use inline here for better performance (around 20% faster than the non-inline one)
static inline double get_w_value(const struct model *model_, int idx, int label_idx)
{
	int nr_class = model_->nr_class;
	const double *w = model_->w;

	if(idx < 0 || idx > model_->nr_feature)
		return 0;
	if(label_idx < 0 || label_idx >= nr_class)
		return 0;
	if(nr_class == 2)
		if(label_idx == 0)
			return w[idx];
		else
			return -w[idx];
	else
		return w[idx*nr_class+label_idx];
}

// feat_idx: starting from 1 to nr_feature
// label_idx: starting from 0 to nr_class-1 for classification models;
//            for regression models, label_idx is ignored.

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->solver_type != L1R_LR_OWLQN
	&& param->solver_type != L1R_LR_BFGS
	&& param->solver_type != L1R_LR_SPARSA)
		return "unknown solver type";

	return NULL;
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

int mpi_get_rank()
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return rank;	
}

int mpi_get_size()
{
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	return size;	
}

void mpi_exit(const int status)
{
	MPI_Finalize();
	exit(status);
}
