#ifndef ISING_SPIN_MODEL_H
#define ISING_SPIN_MODEL_H

#include "Physical_Model.h"
#include "DataStructures.h"
#include "Graph.h"

namespace kspace
{
	class IsingSpinModelBase : public Physical_Model
	{
	private:
		MatrixShared<std::uint8_t> _spin_states;
		Graph::GraphShared _graph;
		std::uint32_t _current_time;
	public:
		IsingSpinModelBase(Graph::GraphShared &graphshared, const std::uint32_t run_time) : _spin_states(graphshared.host->number_of_nodes(), run_time), _graph(graphshared) {};
	};

	class IsingSpinModel : public IsingSpinModelBase
	{
	public:
		IsingSpinModel(Graph::GraphShared &graphshared, const std::uint32_t run_time) : IsingSpinModelBase(graphshared, run_time) {};

		void run(const std::uint32_t temperature);
	};

	class IsingSpinModel_MetropolisHastings : public IsingSpinModelBase
	{
	public:
		IsingSpinModel_MetropolisHastings(Graph::GraphShared &graphshared, const std::uint32_t run_time) : IsingSpinModelBase(graphshared, run_time) {};

		void run(const std::uint32_t temperature, const std::uint32_t time_step);
	};
}

#endif