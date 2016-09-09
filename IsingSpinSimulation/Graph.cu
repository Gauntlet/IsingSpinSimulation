#include "Graph.h"
#include <fstream>
#include "File_IO.h"
#include <map>

#define MAJOR_VERSION 1
#define MINOR_VERSION 0

using namespace kspace::Graph;

GraphHeader Graph::readHeader(FileHandle &file)
{
	GraphHeader hdr;

	fread(&hdr.type, sizeof(char), 8, file());
	if ("xKGRAPHx" == hdr.type) {

		//Check file format version
		fread(&hdr.version[0], sizeof(std::uint8_t), 2, file());
		if (hdr.version[0] <= MAJOR_VERSION && hdr.version[1] <= MINOR_VERSION)
		{
			fread(&hdr.data_offset, sizeof(std::uint16_t), 1, file());				//Data offset
			fread(&hdr.data_size, sizeof(std::uint32_t), 1, file());				//Data size (B)
			fread(&hdr.type, sizeof(uint16_t), 1, file());							//Graph type
			fread(&hdr.num_of_nodes_in_graph, sizeof(std::uint32_t), 1, file());	//Number of nodes in the graph
			fread(&hdr.num_of_nodes_in_file, sizeof(std::uint32_t), 1, file());		//Number of nodes in the file
			fread(&hdr.num_of_neighbours, sizeof(std::uint32_t), 1, file());		//Number of neighbours in the file

			//Get the graph parameters from the file. It will depend on the graph type.
			switch ((std::uint16_t) hdr.type) {
			case (std::uint16_t) Type::Rectangular_Lattice:
				std::uint32_t width, height;

				fread(&width, sizeof(std::uint32_t), 1, file());
				fread(&height, sizeof(std::uint32_t), 1, file());

				hdr.parameters = new Parameters::Rectangular_Lattice(width, height);
				break;

			case (std::uint16_t) Type::Circular_Lattice:
				std::uint32_t numOfDegreesPerNode;
				fread(&numOfDegreesPerNode, sizeof(std::uint32_t), 1, file());

				hdr.parameters = new Parameters::Circular_Lattice(hdr.num_of_nodes_in_graph, numOfDegreesPerNode);
				break;

			case (std::uint16_t) Type::Erdos_Renyi:
				std::uint32_t seed;
				float wiringProbability;

				fread(&seed, sizeof(std::uint32_t), 1, file());
				fread(&wiringProbability, sizeof(float), 1, file());

				hdr.parameters = new Parameters::Erdos_Renyi(hdr.num_of_nodes_in_graph, wiringProbability, seed);
				break;

			case (std::uint16_t) Type::Watts_Strogatz:
				std::uint32_t seed, numOfDegreesPerNode;
				float rewiringProbability;

				fread(&seed, sizeof(std::uint32_t), 1, file());
				fread(&numOfDegreesPerNode, sizeof(std::uint32_t), 1, file());
				fread(&rewiringProbability, sizeof(float), 1, file());

				hdr.parameters = new Parameters::Watts_Strogatz(hdr.num_of_nodes_in_graph, numOfDegreesPerNode, rewiringProbability, seed);
				break;

			case (std::uint16_t) Type::Barabasi_Albert:
				std::uint32_t seed, initNumOfNodes, numOfDegreesPerNode;

				fread(&seed, sizeof(std::uint32_t), 1, file());
				fread(&initNumOfNodes, sizeof(std::uint32_t), 1, file());
				fread(&numOfDegreesPerNode, sizeof(std::uint32_t), 1, file());

				hdr.parameters = new Parameters::Barabasi_Albert(hdr.num_of_nodes_in_graph, initNumOfNodes, numOfDegreesPerNode, seed);
				break;
			}
		}
		else
		{
			throw std::runtime_error("File Format Version: The format version is higher than this library can read. Update library to latest.");
		}
	}
	else
	{
		throw std::runtime_error("Incorrect File Format: File is not a .kgraph");
	}
}

GraphData Graph::readData(const FileHandle &file, const GraphHeader &hdr)
{
	int seeksuccess = fseek(file(), 42, SEEK_SET);
	GraphData data;
	if (!seeksuccess)
	{
		data.degrees = new std::uint32_t[hdr.num_of_nodes_in_graph]();
		
	}
	else
	{
		throw std::runtime_error("fseek to beginning of graph data failed: " + std::to_string(seeksuccess));
	}

}

Graph::Graph(const std::string fname, const MemoryLocation memloc)
{
	FileHandle file(fname, "rb");

	GraphHeader hdr;
	
	//Check the file type.
	

			//move to beginning of data

			if (MemoryLocation::host == memloc)
			{
				parameters = hdr.parameters;
			}
			else if (MemoryLocation::device == memloc)
			{

			}
		}
		
	}
	
}

uint32_t Graph::numOfNodes() const
{
	return parameters->numOfNodes();
}

uint32_t Graph::degree(const uint32_t v) const
{
	return _degrees[v];
}

bool Graph::is_connected(const uint32_t v, const uint32_t w) const
{
	return _adjmat[v*numOfNodes() + w];
}

uint32_t Graph::neighbour(const uint32_t v, const uint32_t kth_neighbour) const
{
	return _adjlist[_offsets[v] + kth_neighbour];
}

kspace::MemoryLocation Graph::memory_location() const
{
	return *_memLoc;
}
