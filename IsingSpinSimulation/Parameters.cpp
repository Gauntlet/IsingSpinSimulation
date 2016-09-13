#include "Graph.h"

using namespace kspace::Graph;

Parameters::Graph_Parameters::Graph_Parameters(const std::string filename) {
	FileHandle file(filename, "rb");
	//Check the file type.
	char filetype[8];
	fread(filetype, sizeof(char), 8, file());
	if ("xKGRAPHx" == filetype) {
		std::uint8_t fileversion[2];
		//Check file format version
		fread(fileversion, sizeof(std::uint8_t), 2, file());
		if (fileversion[0] <= MAJOR_VERSION && fileversion[1] <= MINOR_VERSION)
		{
			std::uint16_t parameter_offset, data_offset, graphtype;
			std::uint32_t data_size, number_of_nodes_in_graph, number_of_nodes_in_file, number_of_neighbours;

			fread(&parameter_offset, sizeof(std::uint16_t), 1, file());			//parameter offset
			fread(&data_offset, sizeof(std::uint16_t), 1, file());				//Data offset
			fread(&data_size, sizeof(std::uint32_t), 1, file());				//Data size (B)
			fread(&graphtype, sizeof(uint16_t), 1, file());						//Graph type
			fread(&number_of_nodes_in_graph, sizeof(std::uint32_t), 1, file());	//Number of nodes in the graph

			type = (GraphType) graphtype;

			int seeksuccess = fseek(file(), parameter_offset, SEEK_SET);
			if (!seeksuccess)
			{
				clear();
				//Get the graph parameters from the file. It will depend on the graph type.
				switch (type) {
				case (std::uint16_t) GraphType::RECTANGULAR_LATTICE:
					std::int32_t width, height;

					fread(&width, sizeof(std::int32_t), 1, file());
					fread(&height, sizeof(std::int32_t), 1, file());
					rectangular_lattice(width, height);
					break;

				case (std::uint16_t) GraphType::CIRCULAR_LATTICE:
					std::int32_t numOfDegreesPerNode;
					fread(&numOfDegreesPerNode, sizeof(std::int32_t), 1, file());

					circular_lattice(number_of_nodes_in_graph, numOfDegreesPerNode);
					break;

				case (std::uint16_t) GraphType::ERDOS_RENYI:
					std::uint32_t seed;
					float wiringProbability;

					fread(&seed, sizeof(std::uint32_t), 1, file());
					fread(&wiringProbability, sizeof(float), 1, file());

					erdos_renyi(number_of_nodes_in_graph, wiringProbability, seed);
					break;

				case (std::uint16_t) GraphType::WATTS_STROGATZ:
					std::uint32_t seed;
					std::int32_t numOfDegreesPerNode;
					float rewiringProbability;

					fread(&seed, sizeof(std::uint32_t), 1, file());
					fread(&numOfDegreesPerNode, sizeof(std::int32_t), 1, file());
					fread(&rewiringProbability, sizeof(float), 1, file());

					watts_strogatz(number_of_nodes_in_graph, numOfDegreesPerNode, rewiringProbability, seed);
					break;

				case (std::uint16_t) GraphType::BARABASI_ALBERT:
					std::uint32_t seed;
					std::int32_t initNumOfNodes, numOfDegreesPerNode;

					fread(&seed, sizeof(std::uint32_t), 1, file());
					fread(&initNumOfNodes, sizeof(std::int32_t), 1, file());
					fread(&numOfDegreesPerNode, sizeof(std::int32_t), 1, file());

					barabasi_albert(number_of_nodes_in_graph, initNumOfNodes, numOfDegreesPerNode, seed);
					break;
				}
			}
			else
			{
				throw std::runtime_error("fseek to beginning of parameter data failed: " + std::to_string(seeksuccess));
			}
		}
		else
		{
			throw std::runtime_error("File Format Version: The format version is higher than this library can read. Update library to latest release.");
		}
	}
	else
	{
		throw std::runtime_error("Incorrect File Format: File is not a .kgraph");
	}
}

void Parameters::Rectangular_Lattice::operator()(const std::int32_t width, const std::int32_t height)
{
	if (width < 0 || height < 0 || width * height < 0)
	{
		throw std::runtime_error("Invalid Graph Parameters");
	}

	this->number_of_nodes = width * height;
	this->width = width;
	this->height = height;
}

void Parameters::Circular_Lattice::operator()(const std::int32_t number_of_nodes, const std::int32_t number_of_degrees)
{
	if (number_of_nodes < 0 || number_of_degrees < 0)
	{
		throw std::runtime_error("Invalid Graph Parameters");
	}

	this->number_of_nodes = number_of_nodes;
	this->number_of_degrees = number_of_degrees;
}

void Parameters::Erdos_Renyi::operator()(const std::int32_t number_of_nodes, const float wiring_probability, const uint32_t seed)
{
	if (number_of_nodes < 0 || wiring_probability < 0 || wiring_probability > 1)
	{
		throw std::runtime_error("Invalid Graph Parameters");
	}

	this->number_of_nodes = number_of_nodes;
	this->wiring_probability = wiring_probability;
	this->seed = seed;
}

void Parameters::Watts_Strogatz::operator()(const std::int32_t number_of_nodes, const std::int32_t number_of_degrees, const float rewiring_probability, const uint32_t seed)
{
	if (number_of_nodes < 0 || number_of_degrees < 0 || rewiring_probability < 0 || rewiring_probability > 1)
	{
		throw std::runtime_error("Invalid Graph Parameters");
	}

	this->number_of_nodes = number_of_nodes;
	this->number_of_degrees = number_of_degrees;
	this->rewiring_probability = rewiring_probability;
	this->seed = seed;
}

void Parameters::Barabasi_Albert::operator()(const std::int32_t init_number_of_nodes, const std::int32_t final_number_of_nodes, const std::int32_t number_of_degrees, const uint32_t seed)
{
	if (init_number_of_nodes < 0 || final_number_of_nodes < 0 || number_of_degrees < 0)
	{
		throw std::runtime_error("Invalid Graph Parameters");
	}

	this->initial_number_of_nodes = init_number_of_nodes;
	this->final_number_of_nodes = final_number_of_nodes;
	this->number_of_degrees = number_of_degrees;
	this->seed = seed;
}