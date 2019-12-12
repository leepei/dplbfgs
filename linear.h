#include <mpi.h>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int l, n;
	double *y;
	struct feature_node **x;
	double bias;            /* < 0 if no bias term */
	int global_l;
};

/* solver_type */
enum { L1R_LR_OWLQN, L1R_LR_BFGS, L1R_LR_SPARSA, L2R_L2_BFGS, L2R_L2_BDA, L2R_L2_ADN, L2R_L2_BDA_CATALYST};

struct parameter
{
	int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	int m;
	double eta;
	double kappa;
	double beta;
	double inner_eps;
	int max_inner_iter;
	int minswitch;
};

struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	double *w;
	int *label;		/* label of each class */
	double bias;
};

struct model* train(const struct problem *prob, const struct parameter *param);

double predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
double predict(const struct model *model_, const struct feature_node *x);

int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, int* label);

void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);

const char *check_parameter(const struct problem *prob, const struct parameter *param);
void set_print_string_function(void (*print_func) (const char*));

#ifdef __cplusplus
}
#endif

int mpi_get_rank();

int mpi_get_size();

template<typename T>
void mpi_allreduce(T *buf, const int count, MPI_Datatype type, MPI_Op op)
{
	std::vector<T> buf_reduced(count);
	MPI_Allreduce(buf, buf_reduced.data(), count, type, op, MPI_COMM_WORLD);
	for(int i=0;i<count;i++)
		buf[i] = buf_reduced[i];
}

void mpi_exit(const int status);

