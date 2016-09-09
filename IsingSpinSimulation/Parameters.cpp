#include "Graph.h"

using namespace kspace::Graph;


Type type;
std::int32_t number_of_nodes;
std::uint32_t width;
std::uint32_t height;
std::int32_t number_of_degrees;
float wiring_probability;
float rewiring_probability;
std::int32_t initial_number_of_nodes;

parameters_t::parameters_t(const Type type)
{
	this->type = type;
	_is_set[0] = true;
	_is_set
	switch (type)
	{
	case Type::Rectangular_Lattice:

		break;
	}
}

void read(FileHandle &file);
void write(FileHandle &file);