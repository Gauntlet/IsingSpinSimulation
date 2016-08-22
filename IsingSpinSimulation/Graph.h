#ifndef GRAPH_H
#define GRAPH_H
#include "DataStructures.h"
#include <cstdint>
#include <cassert>

namespace kspace
{
	namespace Graph
	{
		enum class Type
		{
			Rectangular_Lattice, Circular_Lattice, Erdos_Renyi, Watts_Strogatz, Barabasi_Albert
		};

		namespace Parameters
		{
			class parameters_t
			{
			private:
				Type _type;

			protected:
				uint32_t _numOfNodes;

			public:
				parameters_t(const Type type, const uint32_t numOfNodes) : _type(type), _numOfNodes(numOfNodes) {};

				Type type() const;
				uint32_t numOfNodes() const;
			};


			class Rectangular_Lattice : public parameters_t
			{
			private:
				uint32_t _width;
				uint32_t _height;

			public:
				Rectangular_Lattice::Rectangular_Lattice(const uint32_t width, const uint32_t height) : parameters_t(Type::Rectangular_Lattice, width * height), _width(width), _height(height) {};

				uint32_t width() const;
				uint32_t height() const;
			};


			class Circular_Lattice : public parameters_t
			{
			private:
				uint32_t _numOfDegreesPerNode;

			public:
				Circular_Lattice(const uint32_t numOfNodes, const uint32_t numOfDegreesPerNode);

				uint32_t numOfDegreesPerNode() const;
			};


			static struct Erdos_Renyi : public parameters_t
			{
			private:
				double _wiringProbability;
				uint32_t _seed;

			public:
				Erdos_Renyi(const uint32_t numOfnodes, const double wiringProbability, const uint32_t seed) : parameters_t(Type::Erdos_Renyi, numOfNodes), _wiringProbability(wiringProbability), _seed(seed) {};
				
				double wiringProbability() const;
				uint32_t seed() const;
			};


			class Watts_Strogatz : public parameters_t
			{
			private:
				uint32_t _numOfDegreesPerNode;
				double _rewiringProbability;
				uint32_t _seed;

			public:
				Watts_Strogatz(const uint32_t numOfNodes, const uint32_t numOfDegreesPerNode, const double rewiringProbability, const uint32_t seed);
				
				uint32_t numOfDegreesPerNode() const;
				double rewiringProbability() const;
				uint32_t seed() const;
			};


			class Barabasi_Albert : public parameters_t
			{
			private:
				uint32_t _initNumOfNodes;
				uint32_t _numOfDegreesPerNode;
				uint32_t _seed;
			public:
				Barabasi_Albert(const uint32_t finNumOfNodes, const uint32_t initNumOfNodes, const uint32_t numOfDegreesPerNode, const uint32_t seed);

				uint32_t initNumOfNodes() const;
				uint32_t numOfDegreesPerNode() const;
				uint32_t seed() const;
			};
		};


		class Graph
		{
		private:
			Matrix<std::uint8_t> _adjmat;
			JaggedList<std::uint32_t> _adjlist;
		public:
			Parameters::parameters_t parameters;

			bool is_connected(const uint32_t v, const uint32_t w) const;

			uint32_t neighbour_k(const uint32_t v, const uint32_t k) const;


		};
	}
	
}

#endif