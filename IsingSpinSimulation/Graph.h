#ifndef GRAPH_H
#define GRAPH_H
#include "DataStructures.h"
namespace kspace
{
	class Graph
	{
	private:

	public:
		enum class Type 
		{
			Rectangular_Lattice, Circular_Lattice, Erdos_Renyi, Watts_Strogatz, Barabasi_Albert
		};

		struct Parameters
		{

		} parameters;

		static class Generator
		{
			static class Parameters
			{
			public:
				static struct Rectangular_Lattice
				{
					size_t W;
					size_t H;
				};

				static struct Circular_Lattice
				{
					size_t N;
					size_t K;
				};

				static struct Erdos_Renyi
				{
					size_t N;
					double P;
					size_t seed;
				};

				static struct Watts_Strogatz
				{
					size_t N;
					size_t K;
					double P;
					size_t seed;
				};

				static struct Barabasi_Albert
				{
					size_t N;
					size_t M;
					size_t K;
					size_t seed;
				};
			};
		};
	};
}

#endif