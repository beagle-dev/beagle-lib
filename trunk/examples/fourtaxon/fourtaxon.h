#include <vector>
#include <string>
#include "libhmsbeagle/beagle.h"

typedef std::vector<int> CodedSequence;
typedef std::vector<double> PartialVector;

/*-----------------------------------------------------------------------------
|	FourTaxonExample reads in DNA sequence data for four taxa and simply 
|	recomputes the likelihood of the following unrooted tree numerous times:
|	
|	(1:0.01, 2:0.02, (3:0.03, 4:0.04):0.05)
|	
|	Future improvements:
|	- estimate something
*/
class FourTaxonExample
	{
	public:
		FourTaxonExample();
		void interpretCommandLineParameters(int argc, char* argv[]);
		void run();
		
	private:
		void helpMessage();
		void abort(std::string msg);
		void initBeagleLib();
		void readData();
		void writeData();
		void updateBrlen(unsigned brlen_index);
		void defineOperations();
		double calcLnL(int return_value);
		
	private:
		bool						quiet;
		unsigned 					niters;
		unsigned 					like_root_node;
		unsigned 					like_parent_index;
		unsigned 					like_child_index;
		unsigned 					transmat_index;
		std::string					data_file_name;
        bool                        scaling;
        bool                        single;
        bool                        require_double;
		const unsigned 				ntaxa;
		unsigned 					nsites;
        unsigned                    nrates;
		double						delta;
		double						mu;
		unsigned					seed;
		std::vector<std::string>	taxon_name;
		std::vector<CodedSequence> 	data;
		std::vector<PartialVector> 	partial;
		std::vector<int>			transition_matrix_index;
		std::vector<double>			brlens;
		std::vector<int>			operations;
        std::vector<int>            scaleIndices;
		int							instance_handle;
		int							rsrc_number;
		bool						use_tip_partials;
        bool                        accumulate_on_the_fly;
        bool                        dynamic_scaling;
        bool                        do_rescaling;
        bool                        auto_scaling;
		int                         calculate_derivatives;
        bool                        empirical_derivatives;
        bool                        sse_vectorization;
	};
