#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "linear.h"
#include <math.h>

int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;

struct feature_node *x;
int max_nr_attr = 64;

struct model* model_;

void exit_input_error(int line_num)
{
	fprintf(stderr,"[rank %d] Wrong input format at line %d\n", mpi_get_rank(), line_num);
	mpi_exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void do_predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;

	int n;
	int nr_feature=get_nr_feature(model_);
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = 0; // strtol gives 0 if wrong format

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-2)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct feature_node *) realloc(x,max_nr_attr*sizeof(struct feature_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			// feature indices larger than those in training are not used
			if(x[i].index <= nr_feature)
				++i;
		}

		if(model_->bias>=0)
		{
			x[i].index = n;
			x[i].value = model_->bias;
			i++;
		}
		x[i].index = -1;

		predict_label = predict(model_,x);
		fprintf(output,"%g\n",predict_label);

		if(predict_label == target_label)
			++correct;
		++total;
	}
	mpi_allreduce(&total, 1, MPI_INT, MPI_SUM);
	mpi_allreduce(&correct, 1, MPI_INT, MPI_SUM);
	if(mpi_get_rank() == 0)
		info("Accuracy = %g%% (%d/%d)\n",(double) correct/total*100,correct,total);
}

void exit_with_help()
{
	if(mpi_get_rank() != 0)
		mpi_exit(1);
	printf(
	"Usage: predict [options] test_file model_file output_file\n"
	"options:\n"
	"-q : quiet mode (no outputs)\n"
	);
	mpi_exit(1);
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	
	FILE *input, *output;
	int i;
	char input_file_name[1024];
	char output_file_name[1024];

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'q':
				info = &print_null;
				i--;
				break;
			default:
				if(mpi_get_rank() == 0)
					fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}
	if(i>=argc)
		exit_with_help();

	int multiple = 1;
	if (mpi_get_size() == 1)
		multiple = 0;
	if (multiple)
	{
		char tmp_fmt[8];
		sprintf(tmp_fmt,"%%s.%%0%dd", int(log10(mpi_get_size()))+1);
		sprintf(input_file_name,tmp_fmt, argv[i], mpi_get_rank());
		sprintf(output_file_name,tmp_fmt, argv[i + 2], mpi_get_rank());
	}
	else
	{
		strcpy(input_file_name, argv[i]);
		strcpy(output_file_name, argv[i + 2]);
	}

	input = fopen(input_file_name,"r");

	if(input == NULL)
	{
		fprintf(stderr,"[rank %d] can't open input file %s\n",mpi_get_rank(),input_file_name);
		mpi_exit(1);
	}

	output = fopen(output_file_name,"w");
	if(output == NULL)
	{
		fprintf(stderr,"[rank %d] can't open output file %s\n",mpi_get_rank(),output_file_name);
		mpi_exit(1);
	}

	if((model_=load_model(argv[i+1]))==0)
	{
		fprintf(stderr,"[rank %d] can't open model file %s\n",mpi_get_rank(),argv[i+1]);
		mpi_exit(1);
	}

	x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));
	do_predict(input, output);
	free_and_destroy_model(&model_);
	free(line);
	free(x);
	fclose(input);
	fclose(output);

	MPI_Finalize();
	return 0;
}

