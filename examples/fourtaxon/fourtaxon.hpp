#include <vector>
#include <string>
#include "beagle.h"

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
		void abort(std::string msg);
		void initBeagleLib();
		void readData();
		void writeData();
		double calcLnL();
		
	private:
		unsigned 					niters;
		std::string					data_file_name;
		const unsigned 				ntaxa;
		unsigned 					nsites;
		std::vector<std::string>	taxon_name;
		std::vector<CodedSequence> 	data;
		std::vector<PartialVector> 	partial;
		std::vector<int>			transition_matrix_index;
		std::vector<double>			brlens;
		std::vector<int>			operations;
		int							instance_handle;
	};
