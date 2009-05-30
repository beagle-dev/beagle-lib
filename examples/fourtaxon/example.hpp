#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "beagle.h"

typedef std::vector<int> CodedSequence;
typedef std::vector<double> PartialVector;

class GPUExample
	{
	public:
		GPUExample();
		void abort(std::string msg);
		void init();
		void run();
		
	private:
		void readData(const std::string file_name);
		void writeData(const std::string file_name);
		double calcLnL();
		
	private:
		unsigned 					ntaxa;
		unsigned 					nsites;
		std::vector<std::string>	taxon_name;
		std::vector<CodedSequence> 	data;
		std::vector<PartialVector> 	partial;
		std::vector<int>			transition_matrix_index;
		std::vector<double>			brlens;
		std::vector<int>			operations;
		int							instance_handle;
	};
