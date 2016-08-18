#ifndef GRAPH_H
#define GRAPH_H
#include "DataStructures.h"
namespace kspace {
	class Graph {
	public:
		static class Parameters {
			public:
			static struct Rectangular_Lattice {
				size_t W;
				size_t H;
			};

			static struct Circular_Lattice {
				size_t N;
				size_t K;
			};

			static struct Erdos_Renyi {
				size_t N;
				double P;
				size_t seed;
			};

			static struct Watts_Strogatz {
				size_t N;
				size_t K;
				double P;
			};
		};
	};
}

#endif