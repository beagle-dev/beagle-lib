// cudaBuildHelper : a small program to reformat .ptx files into string constants
//                   that can be included in a C++ header file.  Always appends to a file.
// @author Aaron Darling
// see BEAGLE license for copyright permissions
#include <string>
#include <iostream>
#include <fstream>
using namespace std;

string escape( string& s, char c )
{
	string output;
	string::size_type prev = 0;
	for( string::size_type cur = s.find(c, prev); cur != string::npos; cur = s.find(c,prev) )
	{
		output += s.substr(prev, cur-prev) + "\\" + c;
		prev = cur+1;
	}
	output += s.substr( prev );
	return output;
}

string LFCR(string& s) {
	string output;
	string::size_type prev = 0;
	for( string::size_type cur = s.find('\r'); cur != string::npos; cur = s.find('\r',prev) ) {
		output += s.substr(prev,cur-prev) + "\\n";
		prev = cur+1;
	}
	output += s.substr(prev);
	return output;
}

int main(int argc, char* argv[])
{
	if(argc != 4){
		cerr << "Usage: cudaBuildHelper <ptx source file> <kernel name> <output header file>";
		return -1;
	}
	string inputFileName = argv[1];
	string kernelName = argv[2];
	string outputFileName = argv[3];
	ifstream infile( inputFileName.c_str() );
	ofstream outfile( outputFileName.c_str(), ios::app );
	string cur_line;
	size_t charCount = 0;
	int sectionCount = 0;
	outfile << "#include <string>\n";
	outfile << "\nstatic const char* " << kernelName << "_0 =\"\"\n";
	while( getline( infile, cur_line ) ){
		cur_line = escape(cur_line, '\\');
		cur_line = escape(cur_line, '"');
		cur_line = LFCR(cur_line);
		charCount += cur_line.size();
		if(charCount > 60000){
			// start a new constant variable to circumvent
			sectionCount++;
			charCount = cur_line.size();
			outfile << ";\n";
			outfile << "\nstatic const char* " << kernelName << "_" << sectionCount << " =\"\"\n";
		}
		outfile << '"' << cur_line << "\"\n";
	}
	// write out the function to munge the string variables together
	outfile << ";\n";
	outfile << "static const char* merge"<<kernelName<<"(void){\n"
		"\tstatic std::string abracadabra;\n";
	for(int i=0; i<=sectionCount; i++)
		outfile << "\tabracadabra+=" << kernelName << "_" << i << ";\n";
	outfile << "\treturn abracadabra.c_str();\n"
		"}\n"
		"static const char* " << kernelName << " = merge" << kernelName << "();\n";


	outfile.close();
	return 0;
}

